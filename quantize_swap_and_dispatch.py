from fnmatch import fnmatch
from typing import List, Optional, Union

import torch
from click import secho
from cublas_ops import CublasLinear

from quanto.nn import QModuleMixin, quantize_module, QLinear, QConv2d, QLayerNorm
from quanto.tensor import Optimizer, qtype, qfloat8
from torch import nn


def _set_module_by_name(parent_module, name, child_module):
    module_names = name.split(".")
    if len(module_names) == 1:
        setattr(parent_module, name, child_module)
    else:
        parent_module_name = name[: name.rindex(".")]
        parent_module = parent_module.get_submodule(parent_module_name)
        setattr(parent_module, module_names[-1], child_module)


def _quantize_submodule(
    model: torch.nn.Module,
    name: str,
    module: torch.nn.Module,
    weights: Optional[Union[str, qtype]] = None,
    activations: Optional[Union[str, qtype]] = None,
    optimizer: Optional[Optimizer] = None,
):
    if isinstance(module, CublasLinear):
        return 0
    num_quant = 0
    qmodule = quantize_module(
        module, weights=weights, activations=activations, optimizer=optimizer
    )
    if qmodule is not None:
        _set_module_by_name(model, name, qmodule)
        # num_quant += 1
        qmodule.name = name
        for name, param in module.named_parameters():
            # Save device memory by clearing parameters
            setattr(module, name, None)
            del param
        num_quant += 1

    return num_quant


def _quantize(
    model: torch.nn.Module,
    weights: Optional[Union[str, qtype]] = None,
    activations: Optional[Union[str, qtype]] = None,
    optimizer: Optional[Optimizer] = None,
    include: Optional[Union[str, List[str]]] = None,
    exclude: Optional[Union[str, List[str]]] = None,
):
    """Quantize the specified model submodules

    Recursively quantize the submodules of the specified parent model.

    Only modules that have quantized counterparts will be quantized.

    If include patterns are specified, the submodule name must match one of them.

    If exclude patterns are specified, the submodule must not match one of them.

    Include or exclude patterns are Unix shell-style wildcards which are NOT regular expressions. See
    https://docs.python.org/3/library/fnmatch.html for more details.

    Note: quantization happens in-place and modifies the original model and its descendants.

    Args:
        model (`torch.nn.Module`): the model whose submodules will be quantized.
        weights (`Optional[Union[str, qtype]]`): the qtype for weights quantization.
        activations (`Optional[Union[str, qtype]]`): the qtype for activations quantization.
        include (`Optional[Union[str, List[str]]]`):
            Patterns constituting the allowlist. If provided, module names must match at
            least one pattern from the allowlist.
        exclude (`Optional[Union[str, List[str]]]`):
            Patterns constituting the denylist. If provided, module names must not match
            any patterns from the denylist.
    """
    num_quant = 0
    if include is not None:
        include = [include] if isinstance(include, str) else exclude
    if exclude is not None:
        exclude = [exclude] if isinstance(exclude, str) else exclude
    for name, m in model.named_modules():
        if include is not None and not any(
            fnmatch(name, pattern) for pattern in include
        ):
            continue
        if exclude is not None and any(fnmatch(name, pattern) for pattern in exclude):
            continue
        num_quant += _quantize_submodule(
            model,
            name,
            m,
            weights=weights,
            activations=activations,
            optimizer=optimizer,
        )
    return num_quant


def _freeze(model):
    for name, m in model.named_modules():
        if isinstance(m, QModuleMixin):
            m.freeze()


def _is_block_compilable(module: nn.Module) -> bool:
    for module in module.modules():
        if _is_quantized(module):
            return False
    if _is_quantized(module):
        return False
    return True


def _simple_swap_linears(model: nn.Module, root_name: str = ""):
    for name, module in model.named_children():
        if _is_linear(module):
            weights = module.weight.data
            bias = None
            if module.bias is not None:
                bias = module.bias.data
            with torch.device(module.weight.device):
                new_cublas = CublasLinear(
                    module.in_features,
                    module.out_features,
                    bias=bias is not None,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
            new_cublas.weight.data = weights
            if bias is not None:
                new_cublas.bias.data = bias
            setattr(model, name, new_cublas)
            if root_name == "":
                secho(f"Replaced {name} with CublasLinear", fg="green")
            else:
                secho(f"Replaced {root_name}.{name} with CublasLinear", fg="green")
        else:
            if root_name == "":
                _simple_swap_linears(module, str(name))
            else:
                _simple_swap_linears(module, str(root_name) + "." + str(name))


def _full_quant(
    model, max_quants=24, current_quants=0, quantization_dtype: qtype = qfloat8
):
    if current_quants < max_quants:
        current_quants += _quantize(model, quantization_dtype)
        _freeze(model)
        print(f"Quantized {current_quants} modules")
    return current_quants


def _is_linear(module: nn.Module) -> bool:
    return not isinstance(
        module, (QLinear, QConv2d, QLayerNorm, CublasLinear)
    ) and isinstance(module, nn.Linear)


def _is_quantized(module: nn.Module) -> bool:
    return isinstance(module, (QLinear, QConv2d, QLayerNorm))


def quantize_and_dispatch_to_device(
    flow_model: nn.Module,
    flux_device: torch.device = torch.device("cuda"),
    flux_dtype: torch.dtype = torch.float16,
    num_layers_to_quantize: int = 20,
    quantization_dtype: qtype = qfloat8,
    compile_blocks: bool = True,
    compile_extras: bool = True,
    quantize_extras: bool = False,
):
    num_quanted = 0
    flow_model = flow_model.requires_grad_(False).eval().type(flux_dtype)
    for block in flow_model.single_blocks:
        block.cuda(flux_device)
        if num_quanted < num_layers_to_quantize:
            num_quanted = _full_quant(
                block,
                num_layers_to_quantize,
                num_quanted,
                quantization_dtype=quantization_dtype,
            )

    for block in flow_model.double_blocks:
        block.cuda(flux_device)
        if num_quanted < num_layers_to_quantize:
            num_quanted = _full_quant(
                block,
                num_layers_to_quantize,
                num_quanted,
                quantization_dtype=quantization_dtype,
            )

    to_gpu_extras = [
        "vector_in",
        "img_in",
        "txt_in",
        "time_in",
        "guidance_in",
        "final_layer",
        "pe_embedder",
    ]

    if compile_blocks:
        for i, block in enumerate(flow_model.single_blocks):
            if _is_block_compilable(block):
                block.compile()
                secho(f"Compiled block {i}", fg="green")
        for i, block in enumerate(flow_model.double_blocks):
            if _is_block_compilable(block):
                block.compile()
                secho(f"Compiled block {i}", fg="green")

    _simple_swap_linears(flow_model)
    for extra in to_gpu_extras:
        m_extra = getattr(flow_model, extra).cuda(flux_device).type(flux_dtype)
        if compile_blocks:
            if extra in ["time_in", "vector_in", "guidance_in", "final_layer"]:
                m_extra.compile()
                secho(
                    f"Compiled extra {extra} -- {m_extra.__class__.__name__}",
                    fg="green",
                )
        elif quantize_extras:
            _full_quant(
                m_extra,
                current_quants=num_quanted,
                max_quants=num_layers_to_quantize,
                quantization_dtype=quantization_dtype,
            )
    return flow_model

import re
from typing import Optional, OrderedDict, Tuple, TypeAlias, Union
import torch
from loguru import logger
from safetensors.torch import load_file
from tqdm import tqdm
from torch import nn

try:
    from cublas_ops import CublasLinear
except Exception as e:
    CublasLinear = type(None)
from float8_quantize import F8Linear
from modules.flux_model import Flux

path_regex = re.compile(r"\/|\\")

StateDict: TypeAlias = OrderedDict[str, torch.Tensor]


class LoraWeights:
    def __init__(
        self,
        weights: StateDict,
        path: str,
        name: str = None,
        scale: float = 1.0,
    ) -> None:
        self.path = path
        self.weights = weights
        self.name = name if name else path_regex.split(path)[-1]
        self.scale = scale


def swap_scale_shift(weight):
    scale, shift = weight.chunk(2, dim=0)
    new_weight = torch.cat([shift, scale], dim=0)
    return new_weight


def check_if_lora_exists(state_dict, lora_name):
    subkey = lora_name.split(".lora_A")[0].split(".lora_B")[0].split(".weight")[0]
    for key in state_dict.keys():
        if subkey in key:
            return subkey
    return False


def convert_if_lora_exists(new_state_dict, state_dict, lora_name, flux_layer_name):
    if (original_stubkey := check_if_lora_exists(state_dict, lora_name)) != False:
        weights_to_pop = [k for k in state_dict.keys() if original_stubkey in k]
        for key in weights_to_pop:
            key_replacement = key.replace(
                original_stubkey, flux_layer_name.replace(".weight", "")
            )
            new_state_dict[key_replacement] = state_dict.pop(key)
        return new_state_dict, state_dict
    else:
        return new_state_dict, state_dict


def convert_diffusers_to_flux_transformer_checkpoint(
    diffusers_state_dict,
    num_layers,
    num_single_layers,
    has_guidance=True,
    prefix="",
):
    original_state_dict = {}

    # time_text_embed.timestep_embedder -> time_in
    original_state_dict, diffusers_state_dict = convert_if_lora_exists(
        original_state_dict,
        diffusers_state_dict,
        f"{prefix}time_text_embed.timestep_embedder.linear_1.weight",
        "time_in.in_layer.weight",
    )
    # time_text_embed.text_embedder -> vector_in
    original_state_dict, diffusers_state_dict = convert_if_lora_exists(
        original_state_dict,
        diffusers_state_dict,
        f"{prefix}time_text_embed.text_embedder.linear_1.weight",
        "vector_in.in_layer.weight",
    )

    original_state_dict, diffusers_state_dict = convert_if_lora_exists(
        original_state_dict,
        diffusers_state_dict,
        f"{prefix}time_text_embed.text_embedder.linear_2.weight",
        "vector_in.out_layer.weight",
    )

    if has_guidance:
        original_state_dict, diffusers_state_dict = convert_if_lora_exists(
            original_state_dict,
            diffusers_state_dict,
            f"{prefix}time_text_embed.guidance_embedder.linear_1.weight",
            "guidance_in.in_layer.weight",
        )

        original_state_dict, diffusers_state_dict = convert_if_lora_exists(
            original_state_dict,
            diffusers_state_dict,
            f"{prefix}time_text_embed.guidance_embedder.linear_2.weight",
            "guidance_in.out_layer.weight",
        )

    # context_embedder -> txt_in
    original_state_dict, diffusers_state_dict = convert_if_lora_exists(
        original_state_dict,
        diffusers_state_dict,
        f"{prefix}context_embedder.weight",
        "txt_in.weight",
    )

    # x_embedder -> img_in
    original_state_dict, diffusers_state_dict = convert_if_lora_exists(
        original_state_dict,
        diffusers_state_dict,
        f"{prefix}x_embedder.weight",
        "img_in.weight",
    )
    # double transformer blocks
    for i in range(num_layers):
        block_prefix = f"transformer_blocks.{i}."
        # norms
        original_state_dict, diffusers_state_dict = convert_if_lora_exists(
            original_state_dict,
            diffusers_state_dict,
            f"{prefix}{block_prefix}norm1.linear.weight",
            f"double_blocks.{i}.img_mod.lin.weight",
        )

        original_state_dict, diffusers_state_dict = convert_if_lora_exists(
            original_state_dict,
            diffusers_state_dict,
            f"{prefix}{block_prefix}norm1_context.linear.weight",
            f"double_blocks.{i}.txt_mod.lin.weight",
        )

        sample_q_A = diffusers_state_dict.pop(
            f"{prefix}{block_prefix}attn.to_q.lora_A.weight"
        )
        sample_q_B = diffusers_state_dict.pop(
            f"{prefix}{block_prefix}attn.to_q.lora_B.weight"
        )

        sample_k_A = diffusers_state_dict.pop(
            f"{prefix}{block_prefix}attn.to_k.lora_A.weight"
        )
        sample_k_B = diffusers_state_dict.pop(
            f"{prefix}{block_prefix}attn.to_k.lora_B.weight"
        )

        sample_v_A = diffusers_state_dict.pop(
            f"{prefix}{block_prefix}attn.to_v.lora_A.weight"
        )
        sample_v_B = diffusers_state_dict.pop(
            f"{prefix}{block_prefix}attn.to_v.lora_B.weight"
        )

        context_q_A = diffusers_state_dict.pop(
            f"{prefix}{block_prefix}attn.add_q_proj.lora_A.weight"
        )
        context_q_B = diffusers_state_dict.pop(
            f"{prefix}{block_prefix}attn.add_q_proj.lora_B.weight"
        )

        context_k_A = diffusers_state_dict.pop(
            f"{prefix}{block_prefix}attn.add_k_proj.lora_A.weight"
        )
        context_k_B = diffusers_state_dict.pop(
            f"{prefix}{block_prefix}attn.add_k_proj.lora_B.weight"
        )
        context_v_A = diffusers_state_dict.pop(
            f"{prefix}{block_prefix}attn.add_v_proj.lora_A.weight"
        )
        context_v_B = diffusers_state_dict.pop(
            f"{prefix}{block_prefix}attn.add_v_proj.lora_B.weight"
        )

        original_state_dict[f"double_blocks.{i}.img_attn.qkv.lora_A.weight"] = (
            torch.cat([sample_q_A, sample_k_A, sample_v_A], dim=0)
        )
        original_state_dict[f"double_blocks.{i}.img_attn.qkv.lora_B.weight"] = (
            torch.cat([sample_q_B, sample_k_B, sample_v_B], dim=0)
        )
        original_state_dict[f"double_blocks.{i}.txt_attn.qkv.lora_A.weight"] = (
            torch.cat([context_q_A, context_k_A, context_v_A], dim=0)
        )
        original_state_dict[f"double_blocks.{i}.txt_attn.qkv.lora_B.weight"] = (
            torch.cat([context_q_B, context_k_B, context_v_B], dim=0)
        )

        # qk_norm
        original_state_dict, diffusers_state_dict = convert_if_lora_exists(
            original_state_dict,
            diffusers_state_dict,
            f"{prefix}{block_prefix}attn.norm_q.weight",
            f"double_blocks.{i}.img_attn.norm.query_norm.scale",
        )

        original_state_dict, diffusers_state_dict = convert_if_lora_exists(
            original_state_dict,
            diffusers_state_dict,
            f"{prefix}{block_prefix}attn.norm_k.weight",
            f"double_blocks.{i}.img_attn.norm.key_norm.scale",
        )
        original_state_dict, diffusers_state_dict = convert_if_lora_exists(
            original_state_dict,
            diffusers_state_dict,
            f"{prefix}{block_prefix}attn.norm_added_q.weight",
            f"double_blocks.{i}.txt_attn.norm.query_norm.scale",
        )
        original_state_dict, diffusers_state_dict = convert_if_lora_exists(
            original_state_dict,
            diffusers_state_dict,
            f"{prefix}{block_prefix}attn.norm_added_k.weight",
            f"double_blocks.{i}.txt_attn.norm.key_norm.scale",
        )

        # ff img_mlp

        original_state_dict, diffusers_state_dict = convert_if_lora_exists(
            original_state_dict,
            diffusers_state_dict,
            f"{prefix}{block_prefix}ff.net.0.proj.weight",
            f"double_blocks.{i}.img_mlp.0.weight",
        )
        original_state_dict, diffusers_state_dict = convert_if_lora_exists(
            original_state_dict,
            diffusers_state_dict,
            f"{prefix}{block_prefix}ff.net.2.weight",
            f"double_blocks.{i}.img_mlp.2.weight",
        )
        original_state_dict, diffusers_state_dict = convert_if_lora_exists(
            original_state_dict,
            diffusers_state_dict,
            f"{prefix}{block_prefix}ff_context.net.0.proj.weight",
            f"double_blocks.{i}.txt_mlp.0.weight",
        )

        original_state_dict, diffusers_state_dict = convert_if_lora_exists(
            original_state_dict,
            diffusers_state_dict,
            f"{prefix}{block_prefix}ff_context.net.2.weight",
            f"double_blocks.{i}.txt_mlp.2.weight",
        )
        # output projections
        original_state_dict, diffusers_state_dict = convert_if_lora_exists(
            original_state_dict,
            diffusers_state_dict,
            f"{prefix}{block_prefix}attn.to_out.0.weight",
            f"double_blocks.{i}.img_attn.proj.weight",
        )

        original_state_dict, diffusers_state_dict = convert_if_lora_exists(
            original_state_dict,
            diffusers_state_dict,
            f"{prefix}{block_prefix}attn.to_add_out.weight",
            f"double_blocks.{i}.txt_attn.proj.weight",
        )

    # single transformer blocks
    for i in range(num_single_layers):
        block_prefix = f"single_transformer_blocks.{i}."
        # norm.linear -> single_blocks.0.modulation.lin
        original_state_dict, diffusers_state_dict = convert_if_lora_exists(
            original_state_dict,
            diffusers_state_dict,
            f"{prefix}{block_prefix}norm.linear.weight",
            f"single_blocks.{i}.modulation.lin.weight",
        )

        # Q, K, V, mlp
        q_A = diffusers_state_dict.pop(f"{prefix}{block_prefix}attn.to_q.lora_A.weight")
        q_B = diffusers_state_dict.pop(f"{prefix}{block_prefix}attn.to_q.lora_B.weight")
        k_A = diffusers_state_dict.pop(f"{prefix}{block_prefix}attn.to_k.lora_A.weight")
        k_B = diffusers_state_dict.pop(f"{prefix}{block_prefix}attn.to_k.lora_B.weight")
        v_A = diffusers_state_dict.pop(f"{prefix}{block_prefix}attn.to_v.lora_A.weight")
        v_B = diffusers_state_dict.pop(f"{prefix}{block_prefix}attn.to_v.lora_B.weight")
        mlp_A = diffusers_state_dict.pop(
            f"{prefix}{block_prefix}proj_mlp.lora_A.weight"
        )
        mlp_B = diffusers_state_dict.pop(
            f"{prefix}{block_prefix}proj_mlp.lora_B.weight"
        )
        original_state_dict[f"single_blocks.{i}.linear1.lora_A.weight"] = torch.cat(
            [q_A, k_A, v_A, mlp_A], dim=0
        )
        original_state_dict[f"single_blocks.{i}.linear1.lora_B.weight"] = torch.cat(
            [q_B, k_B, v_B, mlp_B], dim=0
        )

        # output projections
        original_state_dict, diffusers_state_dict = convert_if_lora_exists(
            original_state_dict,
            diffusers_state_dict,
            f"{prefix}{block_prefix}proj_out.weight",
            f"single_blocks.{i}.linear2.weight",
        )

    original_state_dict, diffusers_state_dict = convert_if_lora_exists(
        original_state_dict,
        diffusers_state_dict,
        f"{prefix}proj_out.weight",
        "final_layer.linear.weight",
    )
    original_state_dict, diffusers_state_dict = convert_if_lora_exists(
        original_state_dict,
        diffusers_state_dict,
        f"{prefix}proj_out.bias",
        "final_layer.linear.bias",
    )
    original_state_dict, diffusers_state_dict = convert_if_lora_exists(
        original_state_dict,
        diffusers_state_dict,
        f"{prefix}norm_out.linear.weight",
        "final_layer.adaLN_modulation.1.weight",
    )
    if len(list(diffusers_state_dict.keys())) > 0:
        logger.warning("Unexpected keys:", diffusers_state_dict.keys())

    return original_state_dict


def convert_from_original_flux_checkpoint(
    original_state_dict,
):
    sd = {
        k.replace("lora_unet_", "")
        .replace("double_blocks_", "double_blocks.")
        .replace("single_blocks_", "single_blocks.")
        .replace("_img_attn_", ".img_attn.")
        .replace("_txt_attn_", ".txt_attn.")
        .replace("_img_mod_", ".img_mod.")
        .replace("_txt_mod_", ".txt_mod.")
        .replace("_img_mlp_", ".img_mlp.")
        .replace("_txt_mlp_", ".txt_mlp.")
        .replace("_linear1", ".linear1")
        .replace("_linear2", ".linear2")
        .replace("_modulation_", ".modulation.")
        .replace("lora_up", "lora_B")
        .replace("lora_down", "lora_A"): v
        for k, v in original_state_dict.items()
        if "lora" in k
    }
    return sd


def get_module_for_key(
    key: str, model: Flux
) -> F8Linear | torch.nn.Linear | CublasLinear:
    parts = key.split(".")
    module = model
    for part in parts:
        module = getattr(module, part)
    return module


def get_lora_for_key(key: str, lora_weights: dict):
    prefix = key.split(".lora")[0]
    lora_A = lora_weights[f"{prefix}.lora_A.weight"]
    lora_B = lora_weights[f"{prefix}.lora_B.weight"]
    alpha = lora_weights.get(f"{prefix}.alpha", None)
    return lora_A, lora_B, alpha


def calculate_lora_weight(
    lora_weights: Tuple[torch.Tensor, torch.Tensor, Union[torch.Tensor, float]],
    rank: Optional[int] = None,
    lora_scale: float = 1.0,
    device: Optional[Union[torch.device, int, str]] = None,
):
    lora_A, lora_B, alpha = lora_weights
    if device is None:
        device = lora_A.device

    uneven_rank = lora_B.shape[1] != lora_A.shape[0]
    rank_diff = lora_A.shape[0] / lora_B.shape[1]

    if rank is None:
        rank = lora_B.shape[1]
    if alpha is None:
        alpha = rank

    dtype = torch.float32
    w_up = lora_A.to(dtype=dtype, device=device)
    w_down = lora_B.to(dtype=dtype, device=device)

    if alpha != rank:
        w_up = w_up * (alpha / rank)

    if uneven_rank:
        fused_lora = lora_scale * torch.mm(
            w_down.repeat_interleave(int(rank_diff), dim=1), w_up
        )
    else:
        fused_lora = lora_scale * torch.mm(w_down, w_up)
    return fused_lora


@torch.inference_mode()
def unfuse_lora_weight_from_module(
    fused_weight: torch.Tensor,
    lora_weights: dict,
    rank: Optional[int] = None,
    lora_scale: float = 1.0,
):
    w_dtype = fused_weight.dtype
    dtype = torch.float32
    device = fused_weight.device

    fused_weight = fused_weight.to(dtype=dtype, device=device)
    fused_lora = calculate_lora_weight(lora_weights, rank, lora_scale, device=device)
    module_weight = fused_weight - fused_lora
    return module_weight.to(dtype=w_dtype, device=device)


@torch.inference_mode()
def apply_lora_weight_to_module(
    module_weight: torch.Tensor,
    lora_weights: dict,
    rank: int = None,
    lora_scale: float = 1.0,
):
    w_dtype = module_weight.dtype
    dtype = torch.float32
    device = module_weight.device

    fused_lora = calculate_lora_weight(lora_weights, rank, lora_scale, device=device)
    fused_weight = module_weight.to(dtype=dtype) + fused_lora
    return fused_weight.to(dtype=w_dtype, device=device)


def resolve_lora_state_dict(lora_weights, has_guidance: bool = True):
    check_if_starts_with_transformer = [
        k for k in lora_weights.keys() if k.startswith("transformer.")
    ]
    if len(check_if_starts_with_transformer) > 0:
        lora_weights = convert_diffusers_to_flux_transformer_checkpoint(
            lora_weights, 19, 38, has_guidance=has_guidance, prefix="transformer."
        )
    else:
        lora_weights = convert_from_original_flux_checkpoint(lora_weights)
    logger.info("LoRA weights loaded")
    logger.debug("Extracting keys")
    keys_without_ab = [
        key.replace(".lora_A.weight", "")
        .replace(".lora_B.weight", "")
        .replace(".lora_A", "")
        .replace(".lora_B", "")
        .replace(".alpha", "")
        for key in lora_weights.keys()
    ]
    logger.debug("Keys extracted")
    keys_without_ab = list(set(keys_without_ab))
    keys_without_ab = list(
        set(
            [
                key.replace(".lora_A.weight", "")
                .replace(".lora_B.weight", "")
                .replace(".lora_A", "")
                .replace(".lora_B", "")
                .replace(".alpha", "")
                for key in keys_without_ab
            ]
        )
    )
    return keys_without_ab, lora_weights


def get_lora_weights(lora_path: str | StateDict):
    if isinstance(lora_path, dict):
        return lora_path, True
    else:
        return load_file(lora_path, "cpu"), False


def extract_weight_from_linear(linear: Union[nn.Linear, CublasLinear, F8Linear]):
    dtype = linear.weight.dtype
    weight_is_f8 = False
    if isinstance(linear, F8Linear):
        weight_is_f8 = True
        weight = (
            linear.float8_data.clone()
            .detach()
            .float()
            .mul(linear.scale_reciprocal)
            .to(linear.weight.device)
        )
    elif isinstance(linear, torch.nn.Linear):
        weight = linear.weight.clone().detach().float()
    elif isinstance(linear, CublasLinear):
        weight = linear.weight.clone().detach().float()
    return weight, weight_is_f8, dtype


@torch.inference_mode()
def apply_lora_to_model(
    model: Flux,
    lora_path: str | StateDict,
    lora_scale: float = 1.0,
    return_lora_resolved: bool = False,
) -> Flux:
    has_guidance = model.params.guidance_embed
    logger.info(f"Loading LoRA weights for {lora_path}")
    lora_weights, _ = get_lora_weights(lora_path)

    keys_without_ab, lora_weights = resolve_lora_state_dict(lora_weights, has_guidance)

    for key in tqdm(keys_without_ab, desc="Applying LoRA", total=len(keys_without_ab)):
        module = get_module_for_key(key, model)
        weight, is_f8, dtype = extract_weight_from_linear(module)
        lora_sd = get_lora_for_key(key, lora_weights)
        weight = apply_lora_weight_to_module(weight, lora_sd, lora_scale=lora_scale)
        if is_f8:
            module.set_weight_tensor(weight.type(dtype))
        else:
            module.weight.data = weight.type(dtype)
    logger.success("Lora applied")
    if return_lora_resolved:
        return model, lora_weights
    return model


def remove_lora_from_module(
    model: Flux,
    lora_path: str | StateDict,
    lora_scale: float = 1.0,
):
    has_guidance = model.params.guidance_embed
    logger.info(f"Loading LoRA weights for {lora_path}")
    lora_weights = get_lora_weights(lora_path)
    lora_weights, _ = get_lora_weights(lora_path)

    keys_without_ab, lora_weights = resolve_lora_state_dict(lora_weights, has_guidance)

    for key in tqdm(keys_without_ab, desc="Unfusing LoRA", total=len(keys_without_ab)):
        module = get_module_for_key(key, model)
        weight, is_f8, dtype = extract_weight_from_linear(module)
        lora_sd = get_lora_for_key(key, lora_weights)
        weight = unfuse_lora_weight_from_module(weight, lora_sd, lora_scale=lora_scale)
        if is_f8:
            module.set_weight_tensor(weight.type(dtype))
        else:
            module.weight.data = weight.type(dtype)
    logger.success("Lora unfused")
    return model

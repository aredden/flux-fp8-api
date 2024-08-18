import json
from pathlib import Path
from typing import Optional

import torch
from modules.autoencoder import AutoEncoder, AutoEncoderParams
from modules.conditioner import HFEmbedder
from modules.flux_model import Flux, FluxParams

from safetensors.torch import load_file as load_sft
from enum import StrEnum
from pydantic import BaseModel, ConfigDict
from loguru import logger


class ModelVersion(StrEnum):
    flux_dev = "flux-dev"
    flux_schnell = "flux-schnell"


class QuantizationDtype(StrEnum):
    qfloat8 = "qfloat8"
    qint2 = "qint2"
    qint4 = "qint4"
    qint8 = "qint8"


class ModelSpec(BaseModel):
    version: ModelVersion
    params: FluxParams
    ae_params: AutoEncoderParams
    ckpt_path: str | None
    ae_path: str | None
    repo_id: str | None
    repo_flow: str | None
    repo_ae: str | None
    text_enc_max_length: int = 512
    text_enc_path: str | None
    text_enc_device: str | torch.device | None = "cuda:0"
    ae_device: str | torch.device | None = "cuda:0"
    flux_device: str | torch.device | None = "cuda:0"
    flow_dtype: str = "float16"
    ae_dtype: str = "bfloat16"
    text_enc_dtype: str = "bfloat16"
    num_to_quant: Optional[int] = 20
    quantize_extras: bool = False
    compile_extras: bool = False
    compile_blocks: bool = False
    flow_quantization_dtype: Optional[QuantizationDtype] = QuantizationDtype.qfloat8
    text_enc_quantization_dtype: Optional[QuantizationDtype] = QuantizationDtype.qfloat8
    ae_quantization_dtype: Optional[QuantizationDtype] = None
    clip_quantization_dtype: Optional[QuantizationDtype] = None
    offload_text_encoder: bool = False
    offload_vae: bool = False
    offload_flow: bool = False

    model_config: ConfigDict = {
        "arbitrary_types_allowed": True,
        "use_enum_values": True,
    }


def load_models(config: ModelSpec) -> tuple[Flux, AutoEncoder, HFEmbedder, HFEmbedder]:
    flow = load_flow_model(config)
    ae = load_autoencoder(config)
    clip, t5 = load_text_encoders(config)
    return flow, ae, clip, t5


def parse_device(device: str | torch.device | None) -> torch.device:
    if isinstance(device, str):
        return torch.device(device)
    elif isinstance(device, torch.device):
        return device
    else:
        return torch.device("cuda:0")


def into_dtype(dtype: str) -> torch.dtype:
    if dtype == "float16":
        return torch.float16
    elif dtype == "bfloat16":
        return torch.bfloat16
    elif dtype == "float32":
        return torch.float32
    else:
        raise ValueError(f"Invalid dtype: {dtype}")


def into_device(device: str | torch.device | None) -> torch.device:
    if isinstance(device, str):
        return torch.device(device)
    elif isinstance(device, torch.device):
        return device
    elif isinstance(device, int):
        return torch.device(f"cuda:{device}")
    else:
        return torch.device("cuda:0")


def load_config(
    name: ModelVersion = ModelVersion.flux_dev,
    flux_path: str | None = None,
    ae_path: str | None = None,
    text_enc_path: str | None = None,
    text_enc_device: str | torch.device | None = None,
    ae_device: str | torch.device | None = None,
    flux_device: str | torch.device | None = None,
    flow_dtype: str = "float16",
    ae_dtype: str = "bfloat16",
    text_enc_dtype: str = "bfloat16",
    num_to_quant: Optional[int] = 20,
    compile_extras: bool = False,
    compile_blocks: bool = False,
):
    text_enc_device = str(parse_device(text_enc_device))
    ae_device = str(parse_device(ae_device))
    flux_device = str(parse_device(flux_device))
    return ModelSpec(
        version=name,
        repo_id=(
            "black-forest-labs/FLUX.1-dev"
            if name == ModelVersion.flux_dev
            else "black-forest-labs/FLUX.1-schnell"
        ),
        repo_flow=(
            "flux1-dev.sft" if name == ModelVersion.flux_dev else "flux1-schnell.sft"
        ),
        repo_ae="ae.sft",
        ckpt_path=flux_path,
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_path=ae_path,
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
        text_enc_path=text_enc_path,
        text_enc_device=text_enc_device,
        ae_device=ae_device,
        flux_device=flux_device,
        flow_dtype=flow_dtype,
        ae_dtype=ae_dtype,
        text_enc_dtype=text_enc_dtype,
        text_enc_max_length=512 if name == ModelVersion.flux_dev else 256,
        num_to_quant=num_to_quant,
        compile_extras=compile_extras,
        compile_blocks=compile_blocks,
    )


def load_config_from_path(path: str) -> ModelSpec:
    path_path = Path(path)
    if not path_path.exists():
        raise ValueError(f"Path {path} does not exist")
    if not path_path.is_file():
        raise ValueError(f"Path {path} is not a file")
    return ModelSpec(**json.loads(path_path.read_text()))


def print_load_warning(missing: list[str], unexpected: list[str]) -> None:
    if len(missing) > 0 and len(unexpected) > 0:
        logger.warning(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        logger.warning("\n" + "-" * 79 + "\n")
        logger.warning(
            f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected)
        )
    elif len(missing) > 0:
        logger.warning(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    elif len(unexpected) > 0:
        logger.warning(
            f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected)
        )


def load_flow_model(config: ModelSpec) -> Flux:
    ckpt_path = config.ckpt_path

    with torch.device("meta"):
        model = Flux(config.params, dtype=into_dtype(config.flow_dtype)).type(
            into_dtype(config.flow_dtype)
        )

    if ckpt_path is not None:
        # load_sft doesn't support torch.device
        sd = load_sft(ckpt_path, device="cpu")
        missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
        print_load_warning(missing, unexpected)
    return model


def load_text_encoders(config: ModelSpec) -> tuple[HFEmbedder, HFEmbedder]:
    clip = HFEmbedder(
        "openai/clip-vit-large-patch14",
        max_length=77,
        torch_dtype=into_dtype(config.text_enc_dtype),
        device=into_device(config.text_enc_device).index or 0,
        quantization_dtype=config.clip_quantization_dtype,
    )
    t5 = HFEmbedder(
        config.text_enc_path,
        max_length=config.text_enc_max_length,
        torch_dtype=into_dtype(config.text_enc_dtype),
        device=into_device(config.text_enc_device).index or 0,
        quantization_dtype=config.text_enc_quantization_dtype,
    )
    return clip, t5


def load_autoencoder(config: ModelSpec) -> AutoEncoder:
    ckpt_path = config.ae_path
    with torch.device("meta" if ckpt_path is not None else config.ae_device):
        ae = AutoEncoder(config.ae_params).to(into_dtype(config.ae_dtype))

    if ckpt_path is not None:
        sd = load_sft(ckpt_path, device=str(config.ae_device))
        missing, unexpected = ae.load_state_dict(sd, strict=False, assign=True)
        print_load_warning(missing, unexpected)
    if config.ae_quantization_dtype is not None:
        from quantize_swap_and_dispatch import _full_quant, into_qtype

        ae.to(into_device(config.ae_device))
        _full_quant(
            ae,
            max_quants=8000,
            current_quants=0,
            quantization_dtype=into_qtype(config.ae_quantization_dtype),
        )
        if config.offload_vae:
            ae.to("cpu")
            torch.cuda.empty_cache()
    return ae


class LoadedModels(BaseModel):
    flow: Flux
    ae: AutoEncoder
    clip: HFEmbedder
    t5: HFEmbedder
    config: ModelSpec

    model_config = {
        "arbitrary_types_allowed": True,
        "use_enum_values": True,
    }


def load_models_from_config_path(
    path: str,
) -> LoadedModels:
    config = load_config_from_path(path)
    clip, t5 = load_text_encoders(config)
    return LoadedModels(
        flow=load_flow_model(config),
        ae=load_autoencoder(config),
        clip=clip,
        t5=t5,
        config=config,
    )


def load_models_from_config(config: ModelSpec) -> LoadedModels:
    clip, t5 = load_text_encoders(config)
    return LoadedModels(
        flow=load_flow_model(config),
        ae=load_autoencoder(config),
        clip=clip,
        t5=t5,
        config=config,
    )


if __name__ == "__main__":
    p = "/big/generator-ui/flux-testing/flux/model-dir/flux1-dev.sft"
    ae_p = "/big/generator-ui/flux-testing/flux/model-dir/ae.sft"

    config = load_config(
        ModelVersion.flux_dev,
        flux_path=p,
        ae_path=ae_p,
        text_enc_path="city96/t5-v1_1-xxl-encoder-bf16",
        text_enc_device="cuda:0",
        ae_device="cuda:0",
        flux_device="cuda:0",
        flow_dtype="float16",
        ae_dtype="bfloat16",
        text_enc_dtype="bfloat16",
        num_to_quant=20,
    )
    with open("configs/config-dev-cuda0.json", "w") as f:
        json.dump(config.model_dump(), f, indent=2)
    print(config)

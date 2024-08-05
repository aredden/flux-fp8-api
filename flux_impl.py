import io
from typing import List

import torch
from torch import nn

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.benchmark_limit = 20
torch.set_float32_matmul_precision("high")
from torch._dynamo import config
from torch._inductor import config as ind_config

config.cache_size_limit = 10000000000
ind_config.force_fuse_int_mm_with_mul = True

from loguru import logger
from torchao.quantization.quant_api import int8_weight_only, quantize_

from cublas_linear import CublasLinear as F16Linear
from modules.flux_model import RMSNorm
from sampling import denoise, get_noise, get_schedule, prepare, unpack
from turbojpeg_imgs import TurboImage
from util import (
    ModelSpec,
    into_device,
    into_dtype,
    load_config_from_path,
    load_models_from_config,
)


class Model:
    def __init__(
        self,
        name,
        offload=False,
        clip=None,
        t5=None,
        model=None,
        ae=None,
        dtype=torch.bfloat16,
        verbose=False,
        flux_device="cuda:0",
        ae_device="cuda:1",
        clip_device="cuda:1",
        t5_device="cuda:1",
    ):

        self.name = name
        self.device_flux = (
            flux_device
            if isinstance(flux_device, torch.device)
            else torch.device(flux_device)
        )
        self.device_ae = (
            ae_device
            if isinstance(ae_device, torch.device)
            else torch.device(ae_device)
        )
        self.device_clip = (
            clip_device
            if isinstance(clip_device, torch.device)
            else torch.device(clip_device)
        )
        self.device_t5 = (
            t5_device
            if isinstance(t5_device, torch.device)
            else torch.device(t5_device)
        )
        self.dtype = dtype
        self.offload = offload
        self.clip = clip
        self.t5 = t5
        self.model = model
        self.ae = ae
        self.rng = torch.Generator(device="cpu")
        self.turbojpeg = TurboImage()
        self.verbose = verbose

    @torch.inference_mode()
    def generate(
        self,
        prompt,
        width=720,
        height=1023,
        num_steps=24,
        guidance=3.5,
        seed=None,
    ):
        if num_steps is None:
            num_steps = 4 if self.name == "flux-schnell" else 50

        # allow for packing and conversion to latent space
        height = 16 * (height // 16)
        width = 16 * (width // 16)

        if seed is None:
            seed = self.rng.seed()
        logger.info(f"Generating with:\nSeed: {seed}\nPrompt: {prompt}")

        x = get_noise(
            1,
            height,
            width,
            device=self.device_t5,
            dtype=torch.bfloat16,
            seed=seed,
        )
        inp = prepare(self.t5, self.clip, x, prompt=prompt)
        timesteps = get_schedule(
            num_steps, inp["img"].shape[1], shift=(self.name != "flux-schnell")
        )
        for k in inp:
            inp[k] = inp[k].to(self.device_flux).type(self.dtype)

        # denoise initial noise
        x = denoise(
            self.model,
            **inp,
            timesteps=timesteps,
            guidance=guidance,
            dtype=self.dtype,
            device=self.device_flux,
        )
        inp.clear()
        timesteps.clear()
        torch.cuda.empty_cache()
        x = x.to(self.device_ae)

        # decode latents to pixel space
        x = unpack(x.float(), height, width)
        with torch.autocast(
            device_type=self.device_ae.type, dtype=torch.bfloat16, cache_enabled=False
        ):
            x = self.ae.decode(x)

        # bring into PIL format and save
        x = x.clamp(-1, 1)
        num_images = x.shape[0]
        images: List[torch.Tensor] = []
        for i in range(num_images):
            x = x[i].permute(1, 2, 0).add(1.0).mul(127.5).type(torch.uint8).contiguous()
            images.append(x)
        if len(images) == 1:
            im = images[0]
        else:
            im = torch.vstack(images)

        im = self.turbojpeg.encode_torch(im, quality=95)
        images.clear()
        return io.BytesIO(im)


def quant_module(module, running_sum_quants=0, device_index=0):
    if isinstance(module, nn.Linear) and not isinstance(module, F16Linear):
        module.cuda(device_index)
        module.compile()
        quantize_(module, int8_weight_only())
        running_sum_quants += 1
    elif isinstance(module, F16Linear):
        module.cuda(device_index)
    elif isinstance(module, nn.Conv2d):
        module.cuda(device_index)
    elif isinstance(module, nn.Embedding):
        module.cuda(device_index)
    elif isinstance(module, nn.ConvTranspose2d):
        module.cuda(device_index)
    elif isinstance(module, nn.Conv1d):
        module.cuda(device_index)
    elif isinstance(module, nn.Conv3d):
        module.cuda(device_index)
    elif isinstance(module, nn.ConvTranspose3d):
        module.cuda(device_index)
    elif isinstance(module, nn.RMSNorm):
        module.cuda(device_index)
    elif isinstance(module, RMSNorm):
        module.cuda(device_index)
    elif isinstance(module, nn.LayerNorm):
        module.cuda(device_index)
    return running_sum_quants


def full_quant(model, max_quants=24, current_quants=0, device_index=0):
    for module in model.modules():
        if current_quants < max_quants:
            current_quants = quant_module(
                module, current_quants, device_index=device_index
            )
    return current_quants


@torch.inference_mode()
def load_pipeline_from_config_path(path: str) -> Model:
    config = load_config_from_path(path)
    return load_pipeline_from_config(config)


@torch.inference_mode()
def load_pipeline_from_config(config: ModelSpec) -> Model:
    models = load_models_from_config(config)
    config = models.config
    num_quanted = 0
    max_quanted = config.num_to_quant
    flux_device = into_device(config.flux_device)
    ae_device = into_device(config.ae_device)
    clip_device = into_device(config.text_enc_device)
    t5_device = into_device(config.text_enc_device)
    flux_dtype = into_dtype(config.flow_dtype)
    device_index = flux_device.index or 0
    flow_model = models.flow.requires_grad_(False).eval().type(flux_dtype)
    for block in flow_model.single_blocks:
        block.cuda(flux_device)
        if num_quanted < max_quanted:
            num_quanted = quant_module(
                block.linear1, num_quanted, device_index=device_index
            )

    for block in flow_model.double_blocks:
        block.cuda(flux_device)
        if num_quanted < max_quanted:
            num_quanted = full_quant(
                block, max_quanted, num_quanted, device_index=device_index
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
    for extra in to_gpu_extras:
        getattr(flow_model, extra).cuda(flux_device).type(flux_dtype)
    return Model(
        name=config.version,
        clip=models.clip,
        t5=models.t5,
        model=flow_model,
        ae=models.ae,
        dtype=flux_dtype,
        verbose=False,
        flux_device=flux_device,
        ae_device=ae_device,
        clip_device=clip_device,
        t5_device=t5_device,
    )


if __name__ == "__main__":
    pipe = load_pipeline_from_config_path("config-dev.json")
    o = pipe.generate(
        prompt="a beautiful asian woman in traditional clothing with golden hairpin and blue eyes, wearing a red kimono with dragon patterns",
        height=1024,
        width=1024,
        seed=13456,
        num_steps=24,
        guidance=3.0,
    )
    open("out.jpg", "wb").write(o.read())
    o = pipe.generate(
        prompt="a beautiful asian woman in traditional clothing with golden hairpin and blue eyes, wearing a red kimono with dragon patterns",
        height=1024,
        width=1024,
        seed=7,
        num_steps=24,
        guidance=3.0,
    )
    open("out2.jpg", "wb").write(o.read())

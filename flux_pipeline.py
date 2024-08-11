import base64
import io
import math
from typing import TYPE_CHECKING, Callable, List
from PIL import Image
from einops import rearrange, repeat
import numpy as np

import torch

from flux_emphasis import get_weighted_text_embeddings_flux

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.benchmark_limit = 20
torch.set_float32_matmul_precision("high")
from torch._dynamo import config
from torch._inductor import config as ind_config
from pybase64 import standard_b64decode

config.cache_size_limit = 10000000000
ind_config.force_fuse_int_mm_with_mul = True

from loguru import logger
from turbojpeg_imgs import TurboImage
from torchvision.transforms import functional as TF
from tqdm import tqdm
from util import (
    ModelSpec,
    into_device,
    into_dtype,
    load_config_from_path,
    load_models_from_config,
)


if TYPE_CHECKING:
    from modules.conditioner import HFEmbedder
    from modules.flux_model import Flux
    from modules.autoencoder import AutoEncoder


class FluxPipeline:
    def __init__(
        self,
        name: str,
        offload: bool = False,
        clip: "HFEmbedder" = None,
        t5: "HFEmbedder" = None,
        model: "Flux" = None,
        ae: "AutoEncoder" = None,
        dtype: torch.dtype = torch.bfloat16,
        verbose: bool = False,
        flux_device: torch.device | str = "cuda:0",
        ae_device: torch.device | str = "cuda:1",
        clip_device: torch.device | str = "cuda:1",
        t5_device: torch.device | str = "cuda:1",
        config: ModelSpec = None,
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
        self.clip: "HFEmbedder" = clip
        self.t5: "HFEmbedder" = t5
        self.model: "Flux" = model
        self.ae: "AutoEncoder" = ae
        self.rng = torch.Generator(device="cpu")
        self.turbojpeg = TurboImage()
        self.verbose = verbose
        self.ae_dtype = torch.bfloat16
        self.config = config

    @torch.inference_mode()
    def prepare(
        self,
        img: torch.Tensor,
        prompt: str | list[str],
        target_device: torch.device = torch.device("cuda:0"),
        target_dtype: torch.dtype = torch.float16,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bs, c, h, w = img.shape
        if bs == 1 and not isinstance(prompt, str):
            bs = len(prompt)
        img = img.unfold(2, 2, 2).unfold(3, 2, 2).permute(0, 2, 3, 1, 4, 5)
        img = img.reshape(img.shape[0], -1, img.shape[3] * img.shape[4] * img.shape[5])
        assert img.shape == (
            bs,
            (h // 2) * (w // 2),
            c * 2 * 2,
        ), f"{img.shape} != {(bs, (h//2)*(w//2), c*2*2)}"
        if img.shape[0] == 1 and bs > 1:
            img = img[None].repeat_interleave(bs, dim=0)

        img_ids = torch.zeros(
            h // 2, w // 2, 3, device=target_device, dtype=target_dtype
        )
        img_ids[..., 1] = (
            img_ids[..., 1]
            + torch.arange(h // 2, device=target_device, dtype=target_dtype)[:, None]
        )
        img_ids[..., 2] = (
            img_ids[..., 2]
            + torch.arange(w // 2, device=target_device, dtype=target_dtype)[None, :]
        )

        img_ids = img_ids[None].repeat(bs, 1, 1, 1).flatten(1, 2)
        vec, txt, txt_ids = get_weighted_text_embeddings_flux(
            self,
            prompt,
            num_images_per_prompt=bs,
            device=self.device_clip,
            target_device=target_device,
            target_dtype=target_dtype,
        )
        return img, img_ids, vec, txt, txt_ids

    @torch.inference_mode()
    def time_shift(self, mu: float, sigma: float, t: torch.Tensor):
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    def get_lin_function(
        self, x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
    ) -> Callable[[float], float]:
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return lambda x: m * x + b

    @torch.inference_mode()
    def get_schedule(
        self,
        num_steps: int,
        image_seq_len: int,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
        shift: bool = True,
    ) -> list[float]:
        # extra step for zero
        timesteps = torch.linspace(1, 0, num_steps + 1)

        # shifting the schedule to favor high timesteps for higher signal images
        if shift:
            # eastimate mu based on linear estimation between two points
            mu = self.get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
            timesteps = self.time_shift(mu, 1.0, timesteps)

        return timesteps.tolist()

    @torch.inference_mode()
    def get_noise(
        self,
        num_samples: int,
        height: int,
        width: int,
        generator: torch.Generator,
        dtype=None,
        device=None,
    ):
        if device is None:
            device = self.device_flux
        if dtype is None:
            dtype = self.dtype
        return torch.randn(
            num_samples,
            16,
            # allow for packing
            2 * math.ceil(height / 16),
            2 * math.ceil(width / 16),
            device=device,
            dtype=dtype,
            generator=generator,
            requires_grad=False,
        )

    @torch.inference_mode()
    def into_bytes(self, x: torch.Tensor) -> io.BytesIO:
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

    @torch.inference_mode()
    def vae_decode(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        x = x.to(self.device_ae)
        x = self.unpack(x.float(), height, width)
        with torch.autocast(
            device_type=self.device_ae.type, dtype=torch.bfloat16, cache_enabled=False
        ):
            x = self.ae.decode(x)
        return x

    def unpack(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        return rearrange(
            x,
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=math.ceil(height / 16),
            w=math.ceil(width / 16),
            ph=2,
            pw=2,
        )

    @torch.inference_mode()
    def resize_center_crop(
        self, img: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        img = TF.resize(img, min(width, height))
        img = TF.center_crop(img, (height, width))
        return img

    @torch.inference_mode()
    def preprocess_latent(
        self,
        init_image: torch.Tensor | np.ndarray = None,
        height: int = 720,
        width: int = 1024,
        num_steps: int = 20,
        strength: float = 1.0,
        generator: torch.Generator = None,
        num_images: int = 1,
    ) -> tuple[torch.Tensor, List[float]]:
        # prepare input

        if init_image is not None:
            if isinstance(init_image, np.ndarray):
                init_image = torch.from_numpy(init_image)

            init_image = (
                init_image.permute(2, 0, 1)
                .contiguous()
                .to(self.device_ae, dtype=self.ae_dtype)
                .div(127.5)
                .sub(1)[None, ...]
            )
            init_image = self.resize_center_crop(init_image, height, width)
            with torch.autocast(
                device_type=self.device_ae.type,
                dtype=torch.bfloat16,
                cache_enabled=False,
            ):
                init_image = (
                    self.ae.encode(init_image)
                    .to(dtype=self.dtype, device=self.device_flux)
                    .repeat(num_images, 1, 1, 1)
                )

        x = self.get_noise(
            num_images,
            height,
            width,
            device=self.device_flux,
            dtype=self.dtype,
            generator=generator,
        )
        timesteps = self.get_schedule(
            num_steps=num_steps,
            image_seq_len=x.shape[-1] * x.shape[-2] // 4,
            shift=(self.name != "flux-schnell"),
        )
        if init_image is not None:
            t_idx = int((1 - strength) * num_steps)
            t = timesteps[t_idx]
            timesteps = timesteps[t_idx:]
            x = t * x + (1.0 - t) * init_image
        return x, timesteps

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        width: int = 720,
        height: int = 1024,
        num_steps: int = 24,
        guidance: float = 3.5,
        seed: int | None = None,
        init_image: torch.Tensor | str | None = None,
        strength: float = 1.0,
        silent: bool = False,
        num_images: int = 1,
        return_seed: bool = False,
    ) -> io.BytesIO:
        num_steps = 4 if self.name == "flux-schnell" else num_steps

        if isinstance(init_image, str):
            try:
                init_image = Image.open(init_image)
            except Exception as e:
                init_image = Image.open(io.BytesIO(standard_b64decode(init_image)))
            init_image = torch.from_numpy(np.array(init_image)).type(torch.uint8)

        # allow for packing and conversion to latent space
        height = 16 * (height // 16)
        width = 16 * (width // 16)
        if isinstance(seed, str):
            seed = int(seed)
        if seed is None:
            seed = self.rng.seed()
        logger.info(f"Generating with:\nSeed: {seed}\nPrompt: {prompt}")

        generator = torch.Generator(device=self.device_flux).manual_seed(seed)
        img, timesteps = self.preprocess_latent(
            init_image=init_image,
            height=height,
            width=width,
            num_steps=num_steps,
            strength=strength,
            generator=generator,
            num_images=num_images,
        )
        img, img_ids, vec, txt, txt_ids = self.prepare(
            img=img,
            prompt=prompt,
            target_device=self.device_flux,
            target_dtype=self.dtype,
        )

        # this is ignored for schnell
        guidance_vec = torch.full(
            (img.shape[0],), guidance, device=self.device_flux, dtype=self.dtype
        )
        t_vec = None
        for t_curr, t_prev in tqdm(
            zip(timesteps[:-1], timesteps[1:]), total=len(timesteps) - 1, disable=silent
        ):
            if t_vec is None:
                t_vec = torch.full(
                    (img.shape[0],),
                    t_curr,
                    dtype=self.dtype,
                    device=self.device_flux,
                )
            else:
                t_vec = t_vec.reshape((img.shape[0],)).fill_(t_curr)
            pred = self.model.forward(
                img=img,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_vec,
                guidance=guidance_vec,
            )

            img = img + (t_prev - t_curr) * pred

        torch.cuda.empty_cache()

        # decode latents to pixel space
        img = self.vae_decode(img, height, width)

        if return_seed:
            return self.into_bytes(img), seed
        return self.into_bytes(img)

    @classmethod
    def load_pipeline_from_config_path(cls, path: str) -> "FluxPipeline":
        with torch.inference_mode():
            config = load_config_from_path(path)
            return cls.load_pipeline_from_config(config)

    @classmethod
    def load_pipeline_from_config(cls, config: ModelSpec) -> "FluxPipeline":
        from quantize_swap_and_dispatch import quantize_and_dispatch_to_device

        with torch.inference_mode():

            models = load_models_from_config(config)
            config = models.config
            num_layers_to_quantize = config.num_to_quant
            flux_device = into_device(config.flux_device)
            ae_device = into_device(config.ae_device)
            clip_device = into_device(config.text_enc_device)
            t5_device = into_device(config.text_enc_device)
            flux_dtype = into_dtype(config.flow_dtype)
            flow_model = models.flow

            flow_model = quantize_and_dispatch_to_device(
                flow_model=flow_model,
                flux_device=flux_device,
                flux_dtype=flux_dtype,
                num_layers_to_quantize=num_layers_to_quantize,
                compile_extras=config.compile_extras,
                compile_blocks=config.compile_blocks,
                quantize_extras=config.quantize_extras,
            )

        return cls(
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
            config=config,
        )


if __name__ == "__main__":
    pipe = FluxPipeline.load_pipeline_from_config_path(
        "configs/config-dev-gigaquant.json"
    )
    o = pipe.generate(
        prompt="Street photography portrait of a beautiful asian woman in traditional clothing with golden hairpin and blue eyes, wearing a red kimono with dragon patterns",
        height=1024,
        width=1024,
        num_steps=24,
        guidance=3.0,
    )
    open("out.jpg", "wb").write(o.read())
    o = pipe.generate(
        prompt="Street photography portrait of a beautiful asian woman in traditional clothing with golden hairpin and blue eyes, wearing a red kimono with dragon patterns",
        height=1024,
        width=1024,
        num_steps=24,
        guidance=3.0,
    )
    open("out2.jpg", "wb").write(o.read())
    o = pipe.generate(
        prompt="Street photography portrait of a beautiful asian woman in traditional clothing with golden hairpin and blue eyes, wearing a red kimono with dragon patterns",
        height=1024,
        width=1024,
        num_steps=24,
        guidance=3.0,
    )
    open("out3.jpg", "wb").write(o.read())

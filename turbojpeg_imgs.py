import numpy as np
import torch
from turbojpeg import (
    TurboJPEG,
    TJPF_GRAY,
    TJFLAG_PROGRESSIVE,
    TJFLAG_FASTUPSAMPLE,
    TJFLAG_FASTDCT,
    TJPF_RGB,
    TJPF_BGR,
    TJSAMP_GRAY,
    TJSAMP_411,
    TJSAMP_420,
    TJSAMP_422,
    TJSAMP_444,
    TJSAMP_440,
    TJSAMP_441,
)


class Subsampling:
    S411 = TJSAMP_411
    S420 = TJSAMP_420
    S422 = TJSAMP_422
    S444 = TJSAMP_444
    S440 = TJSAMP_440
    S441 = TJSAMP_441
    GRAY = TJSAMP_GRAY


class Flags:
    PROGRESSIVE = TJFLAG_PROGRESSIVE
    FASTUPSAMPLE = TJFLAG_FASTUPSAMPLE
    FASTDCT = TJFLAG_FASTDCT


class PixelFormat:
    GRAY = TJPF_GRAY
    RGB = TJPF_RGB
    BGR = TJPF_BGR


class TurboImage:
    def __init__(self):
        self.tj = TurboJPEG()
        self.flags = Flags.PROGRESSIVE

        self.subsampling_gray = Subsampling.GRAY
        self.pixel_format_gray = PixelFormat.GRAY
        self.subsampling_rgb = Subsampling.S420
        self.pixel_format_rgb = PixelFormat.RGB

    def set_subsampling_gray(self, subsampling):
        self.subsampling_gray = subsampling

    def set_subsampling_rgb(self, subsampling):
        self.subsampling_rgb = subsampling

    def set_pixel_format_gray(self, pixel_format):
        self.pixel_format_gray = pixel_format

    def set_pixel_format_rgb(self, pixel_format):
        self.pixel_format_rgb = pixel_format

    def set_flags(self, flags):
        self.flags = flags

    def encode(
        self,
        img,
        subsampling,
        pixel_format,
        quality=90,
    ):
        return self.tj.encode(
            img,
            quality=quality,
            flags=self.flags,
            pixel_format=pixel_format,
            jpeg_subsample=subsampling,
        )

    @torch.inference_mode()
    def encode_torch(self, img: torch.Tensor, quality=90):
        if img.ndim == 2:
            subsampling = self.subsampling_gray
            pixel_format = self.pixel_format_gray
            img = img.clamp(0, 255).cpu().contiguous().numpy().astype(np.uint8)
        elif img.ndim == 3:
            subsampling = self.subsampling_rgb
            pixel_format = self.pixel_format_rgb
            if img.shape[0] == 3:
                img = (
                    img.permute(1, 2, 0)
                    .clamp(0, 255)
                    .cpu()
                    .contiguous()
                    .numpy()
                    .astype(np.uint8)
                )
            elif img.shape[2] == 3:
                img = img.clamp(0, 255).cpu().contiguous().numpy().astype(np.uint8)
            else:
                raise ValueError(f"Unsupported image shape: {img.shape}")
        else:
            raise ValueError(f"Unsupported image num dims: {img.ndim}")

        return self.encode(
            img,
            quality=quality,
            subsampling=subsampling,
            pixel_format=pixel_format,
        )

    def encode_numpy(self, img: np.ndarray, quality=90):
        if img.ndim == 2:
            subsampling = self.subsampling_gray
            pixel_format = self.pixel_format_gray
        elif img.ndim == 3:
            if img.shape[0] == 3:
                img = np.ascontiguousarray(img.transpose(1, 2, 0))
            elif img.shape[2] == 3:
                img = np.ascontiguousarray(img)
            else:
                raise ValueError(f"Unsupported image shape: {img.shape}")
            subsampling = self.subsampling_rgb
            pixel_format = self.pixel_format_rgb
        else:
            raise ValueError(f"Unsupported image num dims: {img.ndim}")

        img = img.clip(0, 255).astype(np.uint8)
        return self.encode(
            img, quality=quality, subsampling=subsampling, pixel_format=pixel_format
        )

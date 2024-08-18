import io
from PIL import Image
import numpy as np
import torch


class ImageEncoder:

    @torch.inference_mode()
    def encode_torch(self, img: torch.Tensor, quality=90):
        if img.ndim == 2:
            img = (
                img[None]
                .contiguous()
                .repeat_interleave(3, dim=0)
                .contiguous()
                .clamp(0, 255)
                .type(torch.uint8)
            )
            print(img.shape)
        elif img.ndim == 3:
            if img.shape[0] == 3:
                img = img.contiguous().clamp(0, 255).type(torch.uint8)

            elif img.shape[2] == 3:
                img = img.permute(2, 0, 1).contiguous().clamp(0, 255).type(torch.uint8)
            else:
                raise ValueError(f"Unsupported image shape: {img.shape}")
        else:
            raise ValueError(f"Unsupported image num dims: {img.ndim}")

        img = (
            img.permute(1, 2, 0)
            .contiguous()
            .to(torch.uint8)
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        im = Image.fromarray(img)
        iob = io.BytesIO()
        im.save(iob, format="JPEG", quality=95)
        iob.seek(0)
        return iob.getvalue()


def test_real_img():
    from PIL import Image
    import numpy as np

    im = "out.jpg"
    im = Image.open(im)
    im = np.array(im)
    img_hwc = torch.from_numpy(im).cuda().type(torch.float32)
    img_chw = img_hwc.permute(2, 0, 1).contiguous()
    img_gray = img_hwc.mean(dim=2, keepdim=False).contiguous().clamp(0, 255)
    tj = ImageEncoder()
    o = tj.encode_torch(img_chw)
    o2 = tj.encode_torch(img_hwc)
    o3 = tj.encode_torch(img_gray)
    with open("out_chw.jpg", "wb") as f:
        f.write(o2)
    with open("out_hwc.jpg", "wb") as f:
        f.write(o)
    with open("out_gray.jpg", "wb") as f:
        f.write(o3)


if __name__ == "__main__":
    test_real_img()

import os

import torch
from pydash import max_
from quanto import freeze, qfloat8, qint2, qint4, qint8, quantize
from quanto.nn.qmodule import _QMODULE_TABLE
from safetensors.torch import load_file, load_model, save_model
from torch import Tensor, nn
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    T5EncoderModel,
    T5Tokenizer,
    __version__,
)
from transformers.utils.quantization_config import QuantoConfig

CACHE_DIR = os.environ.get("HF_HOME", "~/.cache/huggingface")


def into_quantization_name(quantization_dtype: str) -> str:
    if quantization_dtype == "qfloat8":
        return "float8"
    elif quantization_dtype == "qint4":
        return "int4"
    elif quantization_dtype == "qint8":
        return "int8"
    elif quantization_dtype == "qint2":
        return "int2"
    else:
        raise ValueError(f"Unsupported quantization dtype: {quantization_dtype}")


class HFEmbedder(nn.Module):
    def __init__(
        self,
        version: str,
        max_length: int,
        device: torch.device | int,
        quantization_dtype: str | None = None,
        **hf_kwargs,
    ):
        super().__init__()
        self.is_clip = version.startswith("openai")
        self.max_length = max_length
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"
        quant_name = (
            into_quantization_name(quantization_dtype) if quantization_dtype else None
        )

        if self.is_clip:
            self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(
                version, max_length=max_length
            )

            self.hf_module: CLIPTextModel = CLIPTextModel.from_pretrained(
                version,
                **hf_kwargs,
                quantization_config=(
                    QuantoConfig(
                        weights=quant_name,
                    )
                    if quant_name
                    else None
                ),
                device_map={"": device},
            )

        else:
            self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(
                version, max_length=max_length
            )
            self.hf_module: T5EncoderModel = T5EncoderModel.from_pretrained(
                version,
                device_map={"": device},
                **hf_kwargs,
                quantization_config=(
                    QuantoConfig(
                        weights=quant_name,
                    )
                    if quant_name
                    else None
                ),
            )

    def forward(self, text: list[str]) -> Tensor:
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        outputs = self.hf_module(
            input_ids=batch_encoding["input_ids"].to(self.hf_module.device),
            attention_mask=None,
            output_hidden_states=False,
        )
        return outputs[self.output_key]


if __name__ == "__main__":
    model = HFEmbedder(
        "city96/t5-v1_1-xxl-encoder-bf16",
        max_length=512,
        device=0,
        quantization_dtype="qfloat8",
    )
    o = model(["hello"])
    print(o)

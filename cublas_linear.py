import math
from typing import Literal, Optional

import torch
from torch.nn import functional as F

from cublas_ops_ext import _simt_hgemv
from cublas_ops_ext import cublas_hgemm_axbT as _cublas_hgemm_axbT
from cublas_ops_ext import cublas_hgemm_batched_simple as _cublas_hgemm_batched_simple
from cublas_ops_ext import (
    cublaslt_hgemm_batched_simple as _cublaslt_hgemm_batched_simple,
)
from cublas_ops_ext import cublaslt_hgemm_simple as _cublaslt_hgemm_simple
from torch import Tensor, nn

global has_moved
has_moved = {idx: False for idx in range(torch.cuda.device_count())}


class StaticState:
    workspace = {
        idx: torch.empty((1024 * 1024 * 8,), dtype=torch.uint8)
        for idx in range(torch.cuda.device_count())
    }
    workspace_size = workspace[0].nelement()
    bias_g = {
        idx: torch.tensor([], dtype=torch.float16)
        for idx in range(torch.cuda.device_count())
    }

    @classmethod
    def get(cls, __name: str, device: torch.device) -> torch.Any:
        global has_moved
        idx = device.index if device.index is not None else 0
        if not has_moved[idx]:
            cls.workspace[idx] = cls.workspace[idx].cuda(idx)
            cls.bias_g[idx] = cls.bias_g[idx].cuda(idx)
            has_moved[idx] = True
        if "bias" in __name:
            return cls.bias_g[idx]
        if "workspace" in __name:
            return cls.workspace[idx]
        if "workspace_size" in __name:
            return cls.workspace_size


@torch.no_grad()
def hgemv_simt(vec: torch.HalfTensor, mat: torch.HalfTensor, block_dim_x: int = 32):
    prev_dims = vec.shape[:-1]
    out = _simt_hgemv(mat, vec.view(-1, 1), block_dim_x=block_dim_x).view(
        *prev_dims, -1
    )
    return out


@torch.no_grad()
def cublas_half_matmul_batched_simple(a: torch.Tensor, b: torch.Tensor):
    out = _cublas_hgemm_batched_simple(a, b)
    return out


@torch.no_grad()
def cublas_half_matmul_simple(a: torch.Tensor, b: torch.Tensor):
    out = _cublas_hgemm_axbT(b, a)
    return out


@torch.no_grad()
def cublaslt_fused_half_matmul_simple(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    epilogue_str: Optional[Literal["NONE", "RELU", "GELU"]] = "NONE",
):
    if bias is None:
        bias = StaticState.get("bias", a.device)
    out = _cublaslt_hgemm_simple(
        a, b, bias, epilogue_str, StaticState.get("workspace", a.device)
    )
    return out


@torch.no_grad()
def cublaslt_fused_half_matmul_batched_simple(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    epilogue_str: Optional[Literal["NONE", "RELU", "GELU"]] = "NONE",
):
    if bias is None:
        bias = StaticState.get("bias", a.device)
    out = _cublaslt_hgemm_batched_simple(
        a, b, bias, epilogue_str, StaticState.get("workspace", a.device)
    )
    return out


class CublasLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        device=None,
        dtype=torch.float16,
        epilogue_str="NONE",
    ):
        super().__init__(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )
        self._epilogue_str = epilogue_str
        self.has_bias = bias
        self.has_checked_weight = False

    def forward(self, x: Tensor) -> Tensor:
        if not self.has_checked_weight:
            if not self.weight.dtype == torch.float16:
                self.to(dtype=torch.float16)
            self.has_checked_weight = True
        out_dtype = x.dtype
        needs_convert = out_dtype != torch.float16
        if needs_convert:
            x = x.type(torch.float16)

        use_cublasLt = self.has_bias or self._epilogue_str != "NONE"
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if math.prod(x.shape) == x.shape[-1]:
            out = F.linear(x, self.weight, bias=self.bias)
            if self._epilogue_str == "RELU":
                return F.relu(out)
            elif self._epilogue_str == "GELU":
                return F.gelu(out)
            if needs_convert:
                return out.type(out_dtype)
            return out
        if use_cublasLt:
            leading_dims = x.shape[:-1]
            x = x.reshape(-1, x.shape[-1])
            out = cublaslt_fused_half_matmul_simple(
                x, self.weight, bias=self.bias.data, epilogue_str=self._epilogue_str
            )
            if needs_convert:
                return out.view(*leading_dims, out.shape[-1]).type(out_dtype)
            return out.view(*leading_dims, out.shape[-1])
        else:
            leading_dims = x.shape[:-1]
            x = x.reshape(-1, x.shape[-1])
            out = cublas_half_matmul_simple(x, self.weight)
            if needs_convert:
                return out.view(*leading_dims, out.shape[-1]).type(out_dtype)
            return out.view(*leading_dims, out.shape[-1])


from __future__ import annotations

import torch


def integer_pixel_center_grid(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    vv, uu = torch.meshgrid(
        torch.arange(int(height), device=device, dtype=dtype),
        torch.arange(int(width), device=device, dtype=dtype),
        indexing="ij",
    )
    return uu, vv


def scale_intrinsics_align_corners_false(
    k3: torch.Tensor,
    *,
    sx: float,
    sy: float,
) -> torch.Tensor:
    kk = k3.clone()
    kk[..., 0, 0] = kk[..., 0, 0] * float(sx)
    kk[..., 1, 1] = kk[..., 1, 1] * float(sy)
    kk[..., 0, 2] = (kk[..., 0, 2] + 0.5) * float(sx) - 0.5
    kk[..., 1, 2] = (kk[..., 1, 2] + 0.5) * float(sy) - 0.5
    return kk


def normalized_intrinsics_to_integer_pixel_k(
    fx_norm: torch.Tensor,
    fy_norm: torch.Tensor,
    cx_norm: torch.Tensor,
    cy_norm: torch.Tensor,
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    intr = torch.eye(3, dtype=torch.float32, device=fx_norm.device).unsqueeze(0).repeat(int(fx_norm.shape[0]), 1, 1)
    intr[:, 0, 0] = fx_norm.to(torch.float32) * float(width)
    intr[:, 1, 1] = fy_norm.to(torch.float32) * float(height)
    intr[:, 0, 2] = cx_norm.to(torch.float32) * float(width) - 0.5
    intr[:, 1, 2] = cy_norm.to(torch.float32) * float(height) - 0.5
    return intr

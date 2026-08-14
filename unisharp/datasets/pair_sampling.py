
from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from unisharp.utils.pixel_convention import scale_intrinsics_align_corners_false


def resize_k3_align_corners_false(k: torch.Tensor, *, sx: float, sy: float) -> torch.Tensor:
    return scale_intrinsics_align_corners_false(k, sx=float(sx), sy=float(sy))


def resize_rgb_u8_chw_high_quality(image: torch.Tensor, *, size: tuple[int, int]) -> torch.Tensor:
    if not torch.is_tensor(image) or image.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got {tuple(image.shape) if torch.is_tensor(image) else type(image)}")
    dst_h, dst_w = int(size[0]), int(size[1])
    if tuple(image.shape[-2:]) == (dst_h, dst_w):
        return image.contiguous()
    resized = F.interpolate(
        image.unsqueeze(0).to(torch.float32),
        size=(dst_h, dst_w),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    return resized[0].round().clamp(0, 255).to(torch.uint8).contiguous()


def project_overlap_ratio(
    src_w2c: torch.Tensor,
    tgt_w2c: torch.Tensor,
    src_k: torch.Tensor,
    tgt_k: torch.Tensor,
    h: int,
    w: int,
    src_hw: tuple[int, int] | None = None,
    tgt_hw: tuple[int, int] | None = None,
    sample_h: int = 32,
    sample_w: int = 56,
    proxy_depth: float = 1.0,
) -> float:
    device = src_w2c.device
    src_h, src_w = tuple(int(v) for v in (src_hw or (h, w)))
    tgt_h, tgt_w = tuple(int(v) for v in (tgt_hw or (h, w)))
    ys = torch.linspace(0, src_h - 1, steps=sample_h, device=device)
    xs = torch.linspace(0, src_w - 1, steps=sample_w, device=device)
    vv, uu = torch.meshgrid(ys, xs, indexing="ij")
    u = uu.reshape(-1)
    v = vv.reshape(-1)

    fx, fy = src_k[0, 0], src_k[1, 1]
    cx, cy = src_k[0, 2], src_k[1, 2]
    x = (u - cx) / fx
    y = (v - cy) / fy
    z = torch.ones_like(x)
    rays = torch.stack([x, y, z], dim=-1)
    rays = rays / torch.norm(rays, dim=-1, keepdim=True).clamp(min=1e-6)
    pts_src = rays * float(proxy_depth)

    src_c2w = torch.linalg.inv(src_w2c)
    pts_src_h = torch.cat([pts_src, torch.ones_like(pts_src[:, :1])], dim=-1)
    pts_w = (src_c2w @ pts_src_h.T).T
    pts_tgt = (tgt_w2c @ pts_w.T).T
    xt, yt, zt = pts_tgt[:, 0], pts_tgt[:, 1], pts_tgt[:, 2].clamp(min=1e-6)
    ut = tgt_k[0, 0] * (xt / zt) + tgt_k[0, 2]
    vt = tgt_k[1, 1] * (yt / zt) + tgt_k[1, 2]
    inside = (zt > 0.0) & (ut >= 0.0) & (ut <= float(tgt_w - 1)) & (vt >= 0.0) & (vt <= float(tgt_h - 1))
    return float(inside.float().mean().item())


def select_targets_for_source(
    *,
    src_idx: int,
    candidate_indices: list[int],
    centers: torch.Tensor,
    min_index_gap: int,
    max_index_gap: int,
    pair_max_translation_m: float,
    pair_min_overlap: float,
    overlap_score_fn: Callable[[int, int], float],
) -> list[int]:
    src_c = centers[int(src_idx)]
    tgt_cands: list[int] = []
    for j in candidate_indices:
        j = int(j)
        if j == int(src_idx):
            continue
        gap = abs(int(j) - int(src_idx))
        if gap < int(min_index_gap) or gap > int(max_index_gap):
            continue
        trans = float(torch.norm(centers[j] - src_c, p=2).item())
        if trans > float(pair_max_translation_m):
            continue
        if float(overlap_score_fn(int(src_idx), j)) >= float(pair_min_overlap):
            tgt_cands.append(j)
    return tgt_cands

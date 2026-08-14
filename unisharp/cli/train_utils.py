
from __future__ import annotations

import numpy as np
import torch


def quat_mul_wxyz(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


def rotmat_to_quat_wxyz(Rm: torch.Tensor) -> torch.Tensor:
    m00, m01, m02 = Rm[0, 0], Rm[0, 1], Rm[0, 2]
    m10, m11, m12 = Rm[1, 0], Rm[1, 1], Rm[1, 2]
    m20, m21, m22 = Rm[2, 0], Rm[2, 1], Rm[2, 2]
    tr = m00 + m11 + m22
    if tr > 0.0:
        s = torch.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif (m00 > m11) and (m00 > m22):
        s = torch.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = torch.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = torch.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    q = torch.stack([w, x, y, z])
    return q / q.norm().clamp(min=1e-8)


def to_k4(k3: torch.Tensor) -> torch.Tensor:
    b = k3.shape[0]
    out = torch.eye(4, dtype=k3.dtype, device=k3.device).unsqueeze(0).repeat(b, 1, 1)
    out[:, :3, :3] = k3
    return out


def warmup_cosine_lr(step: int, warmup: int, total: int, lr0: float, lr1: float) -> float:
    if step <= warmup:
        return lr0 * float(step) / float(max(1, warmup))
    t = (step - warmup) / float(max(1, total - warmup))
    cos = 0.5 * (1 + np.cos(np.pi * t))
    return lr1 + (lr0 - lr1) * cos


@torch.no_grad()
def compute_frustum_mask(
    depth: torch.Tensor,
    tgt_w2c: torch.Tensor,
    src_w2c: torch.Tensor,
    src_k3: torch.Tensor,
    tgt_k3: torch.Tensor,
    img_h: int,
    img_w: int,
    source_img_h: int | None = None,
    source_img_w: int | None = None,
    depth_min: float = 0.05,
    margin: float = 0.05,
) -> torch.Tensor:
    dev = depth.device
    f32 = torch.float32
    src_h = int(img_h if source_img_h is None else source_img_h)
    src_w = int(img_w if source_img_w is None else source_img_w)

    d = depth[0, 0].to(f32)
    valid = d > depth_min

    vy, vx = torch.meshgrid(
        torch.arange(img_h, device=dev, dtype=f32),
        torch.arange(img_w, device=dev, dtype=f32),
        indexing="ij",
    )

    fx_t = tgt_k3[0, 0, 0].to(f32)
    fy_t = tgt_k3[0, 1, 1].to(f32)
    cx_t = tgt_k3[0, 0, 2].to(f32)
    cy_t = tgt_k3[0, 1, 2].to(f32)
    X_t = (vx - cx_t) / fx_t * d
    Y_t = (vy - cy_t) / fy_t * d
    Z_t = d
    pts_t = torch.stack([X_t, Y_t, Z_t], dim=-1).reshape(-1, 3)

    c2w_t = torch.linalg.inv(tgt_w2c[0].to(f32))
    pts_w = pts_t @ c2w_t[:3, :3].T + c2w_t[:3, 3][None, :]

    w2c_s = src_w2c[0].to(f32)
    pts_s = pts_w @ w2c_s[:3, :3].T + w2c_s[:3, 3][None, :]

    Z_s = pts_s[:, 2].clamp(min=1e-4)
    fx_s = src_k3[0, 0, 0].to(f32)
    fy_s = src_k3[0, 1, 1].to(f32)
    cx_s = src_k3[0, 0, 2].to(f32)
    cy_s = src_k3[0, 1, 2].to(f32)
    u_s = pts_s[:, 0] / Z_s * fx_s + cx_s
    v_s = pts_s[:, 1] / Z_s * fy_s + cy_s

    half_w = (src_w - 1) * 0.5
    half_h = (src_h - 1) * 0.5
    x_ndc = (u_s - half_w) / half_w
    y_ndc = (v_s - half_h) / half_h

    in_frust = (
        (x_ndc.abs() <= 1.0 + margin)
        & (y_ndc.abs() <= 1.0 + margin)
        & (pts_s[:, 2] > 0)
    )

    mask = in_frust.reshape(img_h, img_w).float()
    mask = mask * valid.float()
    return mask[None, None]

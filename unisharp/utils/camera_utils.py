from __future__ import annotations

from enum import Enum
from typing import Any

import torch
import torch.nn.functional as F


def reproject_pinhole_z_depth_same_pose(
    z_depth: torch.Tensor | None,
    src_k3: torch.Tensor | None,
    dst_k3: torch.Tensor | None,
    *,
    dst_hw: tuple[int, int] | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not (torch.is_tensor(z_depth) and torch.is_tensor(src_k3) and torch.is_tensor(dst_k3)):
        return None, None
    depth = z_depth
    if depth.ndim == 3:
        depth = depth.unsqueeze(1)
    if depth.ndim != 4 or int(depth.shape[1]) != 1:
        raise ValueError(f"Expected z_depth shape (B,1,H,W), got {tuple(depth.shape)}")
    device = depth.device
    dtype = torch.float32
    depth = depth.to(device=device, dtype=dtype)
    if src_k3.ndim == 2:
        src_k3 = src_k3.unsqueeze(0)
    if dst_k3.ndim == 2:
        dst_k3 = dst_k3.unsqueeze(0)
    src_k = src_k3.to(device=device, dtype=dtype)
    dst_k = dst_k3.to(device=device, dtype=dtype)
    bsz, _, src_h, src_w = depth.shape
    if int(src_k.shape[0]) == 1 and bsz > 1:
        src_k = src_k.expand(bsz, -1, -1)
    if int(dst_k.shape[0]) == 1 and bsz > 1:
        dst_k = dst_k.expand(bsz, -1, -1)
    if int(src_k.shape[0]) != bsz or int(dst_k.shape[0]) != bsz:
        raise ValueError(
            f"Batch mismatch: depth B={bsz}, src_k={tuple(src_k.shape)}, dst_k={tuple(dst_k.shape)}"
        )
    dst_h, dst_w = (
        (int(dst_hw[0]), int(dst_hw[1]))
        if dst_hw is not None
        else (int(src_h), int(src_w))
    )

    yy, xx = torch.meshgrid(
        torch.arange(src_h, device=device, dtype=dtype),
        torch.arange(src_w, device=device, dtype=dtype),
        indexing="ij",
    )
    xx_flat = xx.reshape(-1)
    yy_flat = yy.reshape(-1)
    out_depth: list[torch.Tensor] = []
    out_valid: list[torch.Tensor] = []
    inf = torch.tensor(float("inf"), device=device, dtype=dtype)

    for b in range(bsz):
        z = depth[b, 0].reshape(-1)
        valid = torch.isfinite(z) & (z > 0.0)
        if not bool(valid.any()):
            z_out = torch.zeros((dst_h * dst_w,), device=device, dtype=dtype)
            v_out = torch.zeros_like(z_out, dtype=torch.bool)
            out_depth.append(z_out.reshape(1, dst_h, dst_w))
            out_valid.append(v_out.reshape(1, dst_h, dst_w))
            continue

        fx_s = src_k[b, 0, 0].clamp(min=1e-6)
        fy_s = src_k[b, 1, 1].clamp(min=1e-6)
        cx_s = src_k[b, 0, 2]
        cy_s = src_k[b, 1, 2]
        fx_d = dst_k[b, 0, 0].clamp(min=1e-6)
        fy_d = dst_k[b, 1, 1].clamp(min=1e-6)
        cx_d = dst_k[b, 0, 2]
        cy_d = dst_k[b, 1, 2]

        z_v = z[valid]
        x = (xx_flat[valid] - cx_s) * z_v / fx_s
        y = (yy_flat[valid] - cy_s) * z_v / fy_s
        u = fx_d * (x / z_v.clamp(min=1e-6)) + cx_d
        v = fy_d * (y / z_v.clamp(min=1e-6)) + cy_d

        u0 = torch.floor(u)
        v0 = torch.floor(v)
        lin_parts: list[torch.Tensor] = []
        z_parts: list[torch.Tensor] = []
        for du in (0.0, 1.0):
            for dv in (0.0, 1.0):
                ui = (u0 + du).to(torch.long)
                vi = (v0 + dv).to(torch.long)
                in_bounds = (
                    torch.isfinite(u)
                    & torch.isfinite(v)
                    & (ui >= 0)
                    & (ui < dst_w)
                    & (vi >= 0)
                    & (vi < dst_h)
                )
                if bool(in_bounds.any()):
                    lin_parts.append(vi[in_bounds] * dst_w + ui[in_bounds])
                    z_parts.append(z_v[in_bounds])

        zbuf = torch.full((dst_h * dst_w,), inf, device=device, dtype=dtype)
        if lin_parts:
            lin = torch.cat(lin_parts, dim=0)
            vals = torch.cat(z_parts, dim=0)
            if hasattr(zbuf, "scatter_reduce_"):
                zbuf.scatter_reduce_(0, lin, vals, reduce="amin", include_self=True)
            else:
                order = torch.argsort(vals, descending=True)
                zbuf[lin[order]] = vals[order]
        valid_out = torch.isfinite(zbuf)
        zbuf = torch.where(valid_out, zbuf, torch.zeros_like(zbuf))
        out_depth.append(zbuf.reshape(1, dst_h, dst_w))
        out_valid.append(valid_out.reshape(1, dst_h, dst_w))

    return torch.stack(out_depth, dim=0), torch.stack(out_valid, dim=0)


class CameraType(Enum):
    PINHOLE = "pinhole"
    SPHERICAL = "spherical"


def detect_camera_type(camera_intrinsics: torch.Tensor | None) -> CameraType:
    return CameraType.SPHERICAL if camera_intrinsics is None else CameraType.PINHOLE


def transform_gaussians_to_world(
    gaussians: Any,
    src_w2c: torch.Tensor,
) -> Any:
    c2w = torch.linalg.inv(src_w2c).to(torch.float32)
    r = c2w[:3, :3]
    t = c2w[:3, 3]
    
    means_world = gaussians.mean_vectors.to(torch.float32) @ r.T + t[None, None, :]
    
    q_r = rotmat_to_quat_wxyz(r)
    q_world = quat_mul_wxyz(
        q_r[None, None, :].expand_as(gaussians.quaternions),
        gaussians.quaternions.to(torch.float32)
    )
    q_world = q_world / q_world.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    
    return type(gaussians)(
        mean_vectors=means_world.to(gaussians.mean_vectors.dtype),
        singular_values=gaussians.singular_values,
        quaternions=q_world.to(gaussians.quaternions.dtype),
        colors=gaussians.colors,
        opacities=gaussians.opacities,
    )


def rotmat_to_quat_wxyz(R: torch.Tensor) -> torch.Tensor:
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    
    if trace > 0:
        s = 0.5 / torch.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    return torch.stack([w, x, y, z], dim=0)


def quat_mul_wxyz(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return torch.stack([w, x, y, z], dim=-1)


def to_k4(k3: torch.Tensor) -> torch.Tensor:
    if k3.ndim == 2:
        k4 = torch.eye(4, dtype=k3.dtype, device=k3.device)
        k4[:3, :3] = k3
        return k4
    else:
        B = k3.shape[0]
        k4 = torch.eye(4, dtype=k3.dtype, device=k3.device)[None].expand(B, -1, -1).contiguous()
        k4 = k4.clone()
        k4[:, :3, :3] = k3
        return k4


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
    frustum_margin: float = 1.05,
    source_depth: torch.Tensor | None = None,
    source_occlusion_tolerance_m: float = 0.0,
    source_occlusion_tolerance_ratio: float = 0.10,
    source_visibility_radius_px: int = 0,
) -> torch.Tensor:
    device = depth.device
    src_h = int(img_h if source_img_h is None else source_img_h)
    src_w = int(img_w if source_img_w is None else source_img_w)
    
    y_coords, x_coords = torch.meshgrid(
        torch.arange(img_h, device=device, dtype=torch.float32),
        torch.arange(img_w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    
    fx_t = tgt_k3[0, 0, 0]
    fy_t = tgt_k3[0, 1, 1]
    cx_t = tgt_k3[0, 0, 2]
    cy_t = tgt_k3[0, 1, 2]
    
    z = depth[0, 0]
    x_cam = (x_coords - cx_t) * z / fx_t
    y_cam = (y_coords - cy_t) * z / fy_t
    
    pts_tgt_cam = torch.stack([x_cam, y_cam, z, torch.ones_like(z)], dim=0)
    pts_tgt_cam = pts_tgt_cam.reshape(4, -1)
    
    tgt_c2w = torch.linalg.inv(tgt_w2c[0]).to(torch.float32)
    pts_world = tgt_c2w @ pts_tgt_cam
    
    pts_src_cam = src_w2c[0].to(torch.float32) @ pts_world
    
    fx_s = src_k3[0, 0, 0]
    fy_s = src_k3[0, 1, 1]
    cx_s = src_k3[0, 0, 2]
    cy_s = src_k3[0, 1, 2]
    
    x_src = pts_src_cam[0] / pts_src_cam[2].clamp(min=1e-6)
    y_src = pts_src_cam[1] / pts_src_cam[2].clamp(min=1e-6)
    
    u_src = fx_s * x_src + cx_s
    v_src = fy_s * y_src + cy_s
    
    margin = max(float(frustum_margin), 1.0)
    margin_x = 0.5 * (margin - 1.0) * float(src_w)
    margin_y = 0.5 * (margin - 1.0) * float(src_h)
    valid_depth = (torch.isfinite(z) & (z > 0)).reshape(-1)
    valid = (
        (u_src >= -margin_x) & (u_src < float(src_w) + margin_x) &
        (v_src >= -margin_y) & (v_src < float(src_h) + margin_y) &
        valid_depth &
        (pts_src_cam[2] > 0)
    )

    if torch.is_tensor(source_depth):
        if source_depth.ndim == 3:
            source_depth = source_depth.unsqueeze(1)
        if source_depth.ndim != 4 or int(source_depth.shape[0]) != 1 or int(source_depth.shape[1]) != 1:
            raise ValueError(f"Expected source_depth shape (1,1,H,W), got {tuple(source_depth.shape)}")

        src_depth = source_depth.to(device=device, dtype=torch.float32)
        if tuple(src_depth.shape[-2:]) != (src_h, src_w):
            src_depth = F.interpolate(src_depth, size=(src_h, src_w), mode="nearest")

        src_depth_valid = torch.isfinite(src_depth) & (src_depth > 0.0)
        invalid_depth_fill = 1.0e9
        src_depth_for_min = torch.where(
            src_depth_valid,
            src_depth,
            torch.full_like(src_depth, invalid_depth_fill),
        )
        radius = max(int(source_visibility_radius_px), 0)
        if radius > 0:
            kernel = 2 * radius + 1
            padded_depth = F.pad(src_depth_for_min, (radius, radius, radius, radius), value=invalid_depth_fill)
            src_depth_for_min = -F.max_pool2d(-padded_depth, kernel_size=kernel, stride=1)
            src_depth_valid = (
                F.max_pool2d(src_depth_valid.to(dtype=torch.float32), kernel_size=kernel, stride=1, padding=radius)
                > 0.0
            )

        u_grid = (u_src.reshape(img_h, img_w) / max(float(src_w - 1), 1.0)) * 2.0 - 1.0
        v_grid = (v_src.reshape(img_h, img_w) / max(float(src_h - 1), 1.0)) * 2.0 - 1.0
        sample_grid = torch.stack([u_grid, v_grid], dim=-1)[None]
        sampled_src_z = F.grid_sample(
            src_depth_for_min,
            sample_grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        )[0, 0].reshape(-1)
        sampled_src_valid = (
            F.grid_sample(
                src_depth_valid.to(dtype=torch.float32),
                sample_grid,
                mode="nearest",
                padding_mode="zeros",
                align_corners=True,
            )[0, 0].reshape(-1)
            > 0.5
        )
        z_src_projected = pts_src_cam[2].reshape(-1)
        tolerance = float(source_occlusion_tolerance_m) + float(source_occlusion_tolerance_ratio) * sampled_src_z.abs()
        source_visible = sampled_src_valid & torch.isfinite(sampled_src_z) & (
            z_src_projected <= sampled_src_z + tolerance
        )
        valid = valid & source_visible

    mask = valid.reshape(img_h, img_w).float()[None, None, :, :]
    return mask


def resize_batch(
    batch: dict[str, torch.Tensor],
    target_h: int,
    target_w: int,
    keys_to_resize: list[str] = ["image", "image_u8", "depth"],
) -> dict[str, torch.Tensor]:
    for key in keys_to_resize:
        if key not in batch:
            continue
        
        tensor = batch[key]
        if tensor.shape[-2:] == (target_h, target_w):
            continue
        
        if key.endswith("_u8"):
            tensor = F.interpolate(
                tensor.float(),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            ).round().clamp(0, 255).to(torch.uint8)
        elif "depth" in key:
            tensor = F.interpolate(
                tensor,
                size=(target_h, target_w),
                mode="nearest",
            )
        else:
            tensor = F.interpolate(
                tensor,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )
        
        batch[key] = tensor
    
    return batch

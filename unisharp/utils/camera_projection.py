from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from unisharp.utils.pixel_convention import integer_pixel_center_grid

from .pano import get_cubemap_extrinsics_4x4, get_pinhole_intrinsics_4x4


PoseConvention = Literal["c2w", "w2c", "c2w_t_w2c", "w2c_t_camcenter"]


def build_extrinsics_w2c(
    R: torch.Tensor, t: torch.Tensor, convention: PoseConvention = "c2w"
) -> torch.Tensor:
    R = R.to(torch.float32)
    t = t.to(torch.float32)
    ext = torch.eye(4, dtype=torch.float32, device=R.device)
    if convention == "w2c":
        ext[:3, :3] = R
        ext[:3, 3] = t
        return ext
    if convention == "c2w":
        ext[:3, :3] = R.T
        ext[:3, 3] = -(R.T @ t)
        return ext
    if convention == "c2w_t_w2c":
        ext[:3, :3] = R.T
        ext[:3, 3] = t
        return ext
    if convention == "w2c_t_camcenter":
        ext[:3, :3] = R
        ext[:3, 3] = -(R @ t)
        return ext
    raise ValueError(f"Unsupported convention: {convention}")


def cubemap_face_cameras(base_extr_w2c: torch.Tensor, device: torch.device) -> torch.Tensor:
    face_only = get_cubemap_extrinsics_4x4(device=device, yaw_degrees=0.0)
    return face_only @ base_extr_w2c[None]


def view_frustum_mask_cubemap_union(
    depth_novel: torch.Tensor,
    extr_novel_w2c: torch.Tensor,
    extr_source_w2c: torch.Tensor,
    face_w: int,
    margin: float = 1.05,
    source_depth: torch.Tensor | None = None,
    source_occlusion_tolerance_m: float = 0.0,
    source_occlusion_tolerance_ratio: float = 0.10,
    source_visibility_radius_px: int = 0,
) -> torch.Tensor:
    with torch.autocast(device_type=depth_novel.device.type, enabled=False):
        depth_novel = depth_novel.to(dtype=torch.float32)
        extr_novel_w2c = extr_novel_w2c.to(dtype=torch.float32)
        extr_source_w2c = extr_source_w2c.to(dtype=torch.float32)

        device = depth_novel.device
        H = face_w
        W = face_w
        if tuple(depth_novel.shape[:3]) != (6, H, W):
            raise ValueError("depth_novel must be (6,face_w,face_w,1)")

        intr = get_pinhole_intrinsics_4x4(face_w).to(device=device)
        fx, fy = intr[0, 0], intr[1, 1]
        cx, cy = intr[0, 2], intr[1, 2]

        uu, vv = integer_pixel_center_grid(H, W, device=device, dtype=torch.float32)
        x = (uu - cx) / fx
        y = (vv - cy) / fy
        z = torch.ones_like(x)
        rays_cam = torch.stack([x, y, z], dim=-1)
        rays_cam = rays_cam / torch.norm(rays_cam, dim=-1, keepdim=True).clamp(min=1e-6)

        extr_faces_novel = cubemap_face_cameras(extr_novel_w2c, device=device)
        extr_faces_src = cubemap_face_cameras(extr_source_w2c, device=device)
        cam2world_novel = torch.linalg.inv(extr_faces_novel)

        depth = depth_novel[..., 0].to(torch.float32)
        depth_valid = torch.isfinite(depth) & (depth > 0.0)
        depth = torch.where(depth_valid, depth, torch.zeros_like(depth))
        xyz_cam = rays_cam[None].repeat(6, 1, 1, 1) * depth[..., None]
        xyz_cam_h = torch.cat(
            [xyz_cam, torch.ones_like(xyz_cam[..., :1])], dim=-1
        )
        xyz_world_h = torch.bmm(
            xyz_cam_h.reshape(6, -1, 4),
            cam2world_novel.transpose(-1, -2),
        ).reshape(6, H, W, 4)

        src_depth_for_min = None
        src_depth_valid = None
        if torch.is_tensor(source_depth):
            src_depth = source_depth.to(device=device, dtype=torch.float32)
            if src_depth.ndim != 4:
                raise ValueError(f"Expected source_depth shape (6,H,W,1) or (6,1,H,W), got {tuple(src_depth.shape)}")
            if int(src_depth.shape[0]) != 6:
                raise ValueError(f"Expected source_depth first dimension to be 6, got {tuple(src_depth.shape)}")
            if int(src_depth.shape[-1]) == 1:
                src_depth_bchw = src_depth.permute(0, 3, 1, 2).contiguous()
            elif int(src_depth.shape[1]) == 1:
                src_depth_bchw = src_depth.contiguous()
            else:
                raise ValueError(f"Expected source_depth shape (6,H,W,1) or (6,1,H,W), got {tuple(src_depth.shape)}")
            if tuple(src_depth_bchw.shape[-2:]) != (H, W):
                src_depth_bchw = F.interpolate(src_depth_bchw, size=(H, W), mode="nearest")
            src_depth_valid = torch.isfinite(src_depth_bchw) & (src_depth_bchw > 0.0)
            invalid_depth_fill = 1.0e9
            src_depth_for_min = torch.where(
                src_depth_valid,
                src_depth_bchw,
                torch.full_like(src_depth_bchw, invalid_depth_fill),
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

        mask_any = torch.zeros((6, H, W), dtype=torch.bool, device=device)
        for j in range(6):
            ext = extr_faces_src[j]
            xyz_src = xyz_world_h @ ext.T
            X, Y, Z = xyz_src[..., 0], xyz_src[..., 1], xyz_src[..., 2].clamp(min=1e-6)
            x_ndc = X / Z
            y_ndc = Y / Z
            inside = (
                (x_ndc >= -margin)
                & (x_ndc <= margin)
                & (y_ndc >= -margin)
                & (y_ndc <= margin)
                & (Z > 0)
            )
            inside = inside & depth_valid
            if src_depth_for_min is not None and src_depth_valid is not None:
                u = (x_ndc * fx) + cx
                v = (y_ndc * fy) + cy
                sample_grid = torch.stack(
                    [
                        (u / max(float(W - 1), 1.0)) * 2.0 - 1.0,
                        (v / max(float(H - 1), 1.0)) * 2.0 - 1.0,
                    ],
                    dim=-1,
                )
                src_depth_face = src_depth_for_min[j : j + 1].expand(6, -1, -1, -1)
                src_valid_face = src_depth_valid[j : j + 1].to(dtype=torch.float32).expand(6, -1, -1, -1)
                sampled_src_dist = F.grid_sample(
                    src_depth_face,
                    sample_grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True,
                )[:, 0]
                sampled_src_valid = (
                    F.grid_sample(
                        src_valid_face,
                        sample_grid,
                        mode="nearest",
                        padding_mode="zeros",
                        align_corners=True,
                    )[:, 0]
                    > 0.5
                )
                projected_src_dist = torch.linalg.vector_norm(xyz_src[..., :3], dim=-1)
                tolerance = float(source_occlusion_tolerance_m) + float(source_occlusion_tolerance_ratio) * sampled_src_dist.abs()
                source_visible = sampled_src_valid & torch.isfinite(sampled_src_dist) & (
                    projected_src_dist <= sampled_src_dist + tolerance
                )
                inside = inside & source_visible
            mask_any |= inside
        return mask_any


from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from .unisharp_params import PanoInitializerParams


class PanoGaussianBaseValues(NamedTuple):

    rays: torch.Tensor
    inv_distance: torch.Tensor
    angular_cell: torch.Tensor
    scales: torch.Tensor
    quaternions: torch.Tensor
    colors: torch.Tensor
    opacities: torch.Tensor


class PanoInitializerOutput(NamedTuple):

    gaussian_base_values: PanoGaussianBaseValues
    feature_input: torch.Tensor
    global_scale: torch.Tensor | None = None
    grid_cell_size: torch.Tensor | None = None


def _rescale_distance(
    distance: torch.Tensor,
    dist_min: float = 1.0,
    dist_max: float = 1e2,
    scale_quantile: float = 0.02,
    scale_floor: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    sample = distance
    if int(sample.shape[-2]) > 256 or int(sample.shape[-1]) > 256:
        step_h = max(1, int(sample.shape[-2]) // 256)
        step_w = max(1, int(sample.shape[-1]) // 256)
        sample = sample[..., ::step_h, ::step_w]

    flat = sample.flatten(sample.ndim - 3)
    finite_positive = torch.isfinite(flat) & (flat > 0.0)
    safe_flat = torch.where(finite_positive, flat, torch.full_like(flat, float(dist_max)))
    k = max(1, min(int(safe_flat.shape[-1]), int(round(float(scale_quantile) * float(safe_flat.shape[-1])))))
    robust_min = safe_flat.kthvalue(k, dim=-1).values
    robust_min = robust_min.clamp(min=float(scale_floor), max=float(dist_max))
    factor = dist_min / (robust_min + 1e-6)
    distance = (distance * factor[..., None, None, None]).clamp(max=dist_max)
    return distance, factor


def _downsample_avg(x: torch.Tensor, stride: int, circular_horizontal: bool = True) -> torch.Tensor:
    if stride == 1:
        return x
    del circular_horizontal
    return F.avg_pool2d(x, kernel_size=stride, stride=stride)


def _resize_to_grid(
    x: torch.Tensor,
    target_hw: tuple[int, int] | None,
    *,
    mode: str,
) -> torch.Tensor:
    if target_hw is None:
        return x
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"target_hw must be positive, got {target_hw}")
    if tuple(x.shape[-2:]) == (target_h, target_w):
        return x
    if mode in {"bilinear", "bicubic"}:
        return F.interpolate(x, size=(target_h, target_w), mode=mode, align_corners=False)
    return F.interpolate(x, size=(target_h, target_w), mode=mode)


def _grid_cell_size_uv(
    *,
    batch_size: int,
    image_h: int,
    image_w: int,
    stride: float,
    circular_horizontal: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if bool(circular_horizontal):
        cell_w = (2.0 * torch.pi * float(stride)) / float(max(int(image_w), 1))
        cell_h = (torch.pi * float(stride)) / float(max(int(image_h), 1))
    else:
        cell_w = float(stride) / float(max(int(image_w), 1))
        cell_h = float(stride) / float(max(int(image_h), 1))
    return torch.tensor(
        [float(cell_w), float(cell_h)],
        device=device,
        dtype=dtype,
    ).view(1, 2, 1, 1, 1).expand(int(batch_size), -1, -1, -1, -1)


def _format_cell_size_override_uv(
    value: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    cell = value.to(device=device, dtype=dtype)
    if cell.ndim == 0:
        cell = cell.view(1, 1).expand(int(batch_size), 2)
    elif cell.ndim > 1:
        cell = cell.reshape(int(cell.shape[0]), -1)
        if int(cell.shape[1]) == 1:
            cell = cell.expand(-1, 2)
        else:
            cell = cell[:, :2]
    else:
        cell = cell.reshape(-1)
        if int(cell.numel()) == 1:
            cell = cell.view(1, 1).expand(int(batch_size), 2)
        elif int(cell.numel()) == 2 and int(batch_size) > 1:
            cell = cell.view(1, 2).expand(int(batch_size), 2)
        else:
            cell = cell.view(int(batch_size), -1)
            if int(cell.shape[1]) == 1:
                cell = cell.expand(-1, 2)
    if int(cell.shape[0]) != int(batch_size):
        raise ValueError(f"grid_cell_size_override must have batch size {int(batch_size)}, got {int(cell.shape[0])}")
    return cell.clamp(min=1e-6).view(int(batch_size), 2, 1, 1, 1)


def _safe_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp(min=eps)


def _angle_between_unit(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    dot = (a * b).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
    cross = torch.linalg.vector_norm(torch.cross(a, b, dim=1), dim=1, keepdim=True)
    return torch.atan2(cross, dot).clamp(min=1e-6)


def _ray_angular_cell(rays: torch.Tensor, *, circular_horizontal: bool) -> torch.Tensor:
    r = _safe_normalize(rays)
    b, _, h, w = r.shape
    if w > 1:
        if bool(circular_horizontal):
            angle_right = _angle_between_unit(r, torch.roll(r, shifts=-1, dims=-1))
            angle_left = _angle_between_unit(r, torch.roll(r, shifts=1, dims=-1))
            cell_u = 0.5 * (angle_right + angle_left)
        else:
            pair_u = _angle_between_unit(r[..., :-1], r[..., 1:])
            right = F.pad(pair_u, (0, 1, 0, 0), mode="replicate")
            left = F.pad(pair_u, (1, 0, 0, 0), mode="replicate")
            cell_u = 0.5 * (left + right)
    else:
        cell_u = torch.full((b, 1, h, w), 1e-3, device=r.device, dtype=r.dtype)
    if h > 1:
        pair_v = _angle_between_unit(r[..., :-1, :], r[..., 1:, :])
        down = F.pad(pair_v, (0, 0, 0, 1), mode="replicate")
        up = F.pad(pair_v, (0, 0, 1, 0), mode="replicate")
        cell_v = 0.5 * (up + down)
    else:
        cell_v = torch.full((b, 1, h, w), 1e-3, device=r.device, dtype=r.dtype)
    return torch.cat([cell_u, cell_v], dim=1).clamp(min=1e-6, max=0.25).unsqueeze(2)


def _smooth_angular_cell(
    cell: torch.Tensor,
    *,
    kernel_size: int = 9,
    circular_horizontal: bool,
) -> torch.Tensor:
    k = int(kernel_size)
    if k <= 1:
        return cell
    if k % 2 == 0:
        k += 1
    pad = k // 2
    b, c, one, h, w = cell.shape
    if one != 1:
        raise ValueError(f"Expected angular cell shape [B,2,1,H,W], got {tuple(cell.shape)}")
    x = cell.reshape(b * c, 1, h, w)
    if pad > 0:
        if bool(circular_horizontal):
            x = F.pad(x, (pad, pad, 0, 0), mode="circular")
        else:
            x = F.pad(x, (pad, pad, 0, 0), mode="replicate")
        x = F.pad(x, (0, 0, pad, pad), mode="replicate")
    x = F.avg_pool2d(x, kernel_size=k, stride=1)
    return x.reshape(b, c, one, h, w).clamp(min=1e-6, max=0.25)


def _fallback_tangent_basis(rays: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    b, _, h, w = rays.shape
    r = rays.reshape(b, 3, -1)
    up1 = torch.tensor([0.0, 1.0, 0.0], device=r.device, dtype=r.dtype)[:, None]
    up2 = torch.tensor([1.0, 0.0, 0.0], device=r.device, dtype=r.dtype)[:, None]
    dot = (r * up1).sum(dim=1, keepdim=True).abs()
    up = torch.where(dot > 0.9, up2.expand_as(up1), up1.expand_as(up1))
    e_u = _safe_normalize(torch.cross(up.expand_as(r), r, dim=1))
    e_v = _safe_normalize(torch.cross(r, e_u, dim=1))
    return e_u.reshape(b, 3, h, w), e_v.reshape(b, 3, h, w)


def _central_ray_difference(rays: torch.Tensor, *, dim: int, circular: bool) -> torch.Tensor:
    if int(rays.shape[dim]) <= 1:
        return torch.zeros_like(rays)
    if bool(circular):
        return 0.5 * (
            torch.roll(rays, shifts=-1, dims=dim) - torch.roll(rays, shifts=1, dims=dim)
        )
    forward = torch.roll(rays, shifts=-1, dims=dim)
    backward = torch.roll(rays, shifts=1, dims=dim)
    diff = 0.5 * (forward - backward)
    sl_first = [slice(None)] * rays.ndim
    sl_first[dim] = 0
    sl_second = [slice(None)] * rays.ndim
    sl_second[dim] = 1
    diff[tuple(sl_first)] = rays[tuple(sl_second)] - rays[tuple(sl_first)]
    sl_last = [slice(None)] * rays.ndim
    sl_last[dim] = -1
    sl_prev = [slice(None)] * rays.ndim
    sl_prev[dim] = -2
    diff[tuple(sl_last)] = rays[tuple(sl_last)] - rays[tuple(sl_prev)]
    return diff


def _build_tangent_basis(rays: torch.Tensor, *, circular_horizontal: bool) -> tuple[torch.Tensor, torch.Tensor]:
    r = _safe_normalize(rays)
    fallback_u, fallback_v = _fallback_tangent_basis(r)

    du = _central_ray_difference(r, dim=-1, circular=bool(circular_horizontal))
    du = du - (du * r).sum(dim=1, keepdim=True) * r
    du_norm = du.norm(dim=1, keepdim=True)
    e_u = torch.where(du_norm > 1e-7, du / du_norm.clamp(min=1e-7), fallback_u)
    e_u = _safe_normalize(e_u)

    dv = _central_ray_difference(r, dim=-2, circular=False)
    dv = dv - (dv * r).sum(dim=1, keepdim=True) * r
    dv = dv - (dv * e_u).sum(dim=1, keepdim=True) * e_u
    dv_norm = dv.norm(dim=1, keepdim=True)
    e_v_fallback = _safe_normalize(torch.cross(r, e_u, dim=1))
    e_v = torch.where(dv_norm > 1e-7, dv / dv_norm.clamp(min=1e-7), e_v_fallback)
    handed = (torch.cross(e_u, e_v, dim=1) * r).sum(dim=1, keepdim=True)
    e_v = torch.where(handed < 0.0, -e_v, e_v)
    e_v = torch.where(torch.isfinite(e_v).all(dim=1, keepdim=True), e_v, fallback_v)
    return e_u, _safe_normalize(e_v)


def _rotmat_to_quat_wxyz(rot: torch.Tensor) -> torch.Tensor:
    m00 = rot[:, 0, 0]
    m01 = rot[:, 0, 1]
    m02 = rot[:, 0, 2]
    m10 = rot[:, 1, 0]
    m11 = rot[:, 1, 1]
    m12 = rot[:, 1, 2]
    m20 = rot[:, 2, 0]
    m21 = rot[:, 2, 1]
    m22 = rot[:, 2, 2]
    qw = 0.5 * torch.sqrt((1.0 + m00 + m11 + m22).clamp(min=1e-8))
    qx = torch.copysign(0.5 * torch.sqrt((1.0 + m00 - m11 - m22).clamp(min=1e-8)), m21 - m12)
    qy = torch.copysign(0.5 * torch.sqrt((1.0 - m00 + m11 - m22).clamp(min=1e-8)), m02 - m20)
    qz = torch.copysign(0.5 * torch.sqrt((1.0 - m00 - m11 + m22).clamp(min=1e-8)), m10 - m01)
    quat = torch.stack([qw, qx, qy, qz], dim=1)
    return quat / quat.norm(dim=1, keepdim=True).clamp(min=1e-8)


class PanoInitializer(nn.Module):

    def __init__(self, params: PanoInitializerParams) -> None:
        super().__init__()
        self.params = params

    def forward(
        self,
        image: torch.Tensor,
        rays: torch.Tensor,
        distance: torch.Tensor,
        angular_cell_rays: torch.Tensor | None = None,
        grid_cell_size_override: torch.Tensor | None = None,
        force_grid_cell_size_override: bool = False,
        target_hw: tuple[int, int] | None = None,
    ) -> PanoInitializerOutput:
        p = self.params
        b, _, h, w = image.shape
        device = image.device

        global_scale: torch.Tensor | None = None
        if p.normalize_distance:
            distance, factor = _rescale_distance(distance)
            global_scale = 1.0 / factor

        stride = int(p.stride)
        if target_hw is None:
            img_ds = _downsample_avg(image, stride, circular_horizontal=bool(p.circular_horizontal))
            rays_ds = _downsample_avg(rays, stride, circular_horizontal=bool(p.circular_horizontal))
            angular_rays_ds = (
                _downsample_avg(angular_cell_rays, stride, circular_horizontal=bool(p.circular_horizontal))
                if torch.is_tensor(angular_cell_rays)
                else rays_ds
            )
            dist_ds = _downsample_avg(
                distance, stride, circular_horizontal=bool(p.circular_horizontal)
            ).clamp(min=1e-4)
            pool_hw: tuple[int, int] | None = None
            fallback_stride_u = fallback_stride_v = float(stride)
        else:
            target_h, target_w = int(target_hw[0]), int(target_hw[1])
            img_ds = _resize_to_grid(image, (target_h, target_w), mode="area")
            rays_ds = _resize_to_grid(rays, (target_h, target_w), mode="bilinear")
            angular_rays_ds = (
                _resize_to_grid(angular_cell_rays, (target_h, target_w), mode="bilinear")
                if torch.is_tensor(angular_cell_rays)
                else rays_ds
            )
            dist_ds = _resize_to_grid(distance, (target_h, target_w), mode="area").clamp(min=1e-4)
            pool_hw = (target_h, target_w)
            fallback_stride_u = float(w) / float(max(target_w, 1))
            fallback_stride_v = float(h) / float(max(target_h, 1))

        rays_ds_n = _safe_normalize(rays_ds)
        angular_rays_ds_n = _safe_normalize(angular_rays_ds)
        angular_cell = _ray_angular_cell(
            angular_rays_ds_n,
            circular_horizontal=bool(p.circular_horizontal),
        ).to(dtype=dist_ds.dtype)
        if torch.is_tensor(grid_cell_size_override):
            cell_size = _format_cell_size_override_uv(
                grid_cell_size_override,
                batch_size=b,
                device=device,
                dtype=dist_ds.dtype,
            )
        else:
            cell_size = _grid_cell_size_uv(
                batch_size=b,
                image_h=h,
                image_w=w,
                stride=0.5 * (fallback_stride_u + fallback_stride_v),
                circular_horizontal=bool(p.circular_horizontal),
                device=device,
                dtype=dist_ds.dtype,
            )
        cell_size_full = cell_size.expand(-1, -1, 1, int(rays_ds_n.shape[-2]), int(rays_ds_n.shape[-1]))
        if bool(force_grid_cell_size_override) and torch.is_tensor(grid_cell_size_override):
            angular_cell = cell_size_full
        else:
            valid_angular = torch.isfinite(angular_cell) & (angular_cell > 1e-6)
            angular_cell = torch.where(valid_angular, angular_cell, cell_size_full)
            angular_cell = _smooth_angular_cell(
                angular_cell,
                kernel_size=9,
                circular_horizontal=bool(p.circular_horizontal),
            )
        angular_cell = angular_cell.clamp(min=1e-6, max=0.25)

        inv = 1.0 / dist_ds.clamp(min=1e-4)

        inv_full = 1.0 / distance.clamp(min=1e-4)
        if p.first_layer_depth_option == "surface_min":
            inv1_src = inv_full[:, 0:1]
            inv1 = (
                F.adaptive_max_pool2d(inv1_src, output_size=pool_hw)
                if pool_hw is not None
                else F.max_pool2d(inv1_src, kernel_size=stride, stride=stride)
            )
        else:
            inv1_src = -inv_full[:, 0:1]
            inv1 = -(
                F.adaptive_max_pool2d(inv1_src, output_size=pool_hw)
                if pool_hw is not None
                else F.max_pool2d(inv1_src, kernel_size=stride, stride=stride)
            )

        if int(p.num_layers) == 1:
            inv_L = inv1[:, :, None]
        else:
            following = inv_full if inv_full.shape[1] == 1 else inv_full[:, 1:2]
            if p.rest_layer_depth_option == "surface_min":
                inv2_src = following
                inv2 = (
                    F.adaptive_max_pool2d(inv2_src, output_size=pool_hw)
                    if pool_hw is not None
                    else F.max_pool2d(inv2_src, kernel_size=stride, stride=stride)
                )
            else:
                inv2_src = -following
                inv2 = -(
                    F.adaptive_max_pool2d(inv2_src, output_size=pool_hw)
                    if pool_hw is not None
                    else F.max_pool2d(inv2_src, kernel_size=stride, stride=stride)
                )
            inv_L = torch.cat([inv1[:, :, None], inv2[:, :, None]], dim=2)

        L = inv_L.shape[2]
        rays_L = rays_ds_n[:, :, None].repeat(1, 1, L, 1, 1)
        inv_dist_L = inv_L.clamp(min=1e-6)
        dist_L = 1.0 / inv_dist_L.clamp(min=1e-6)

        cell_u = angular_cell[:, 0:1].to(dtype=dist_L.dtype, device=dist_L.device)
        cell_v = angular_cell[:, 1:2].to(dtype=dist_L.dtype, device=dist_L.device)
        cell_r = 0.5 * (cell_u + cell_v)
        scales = torch.cat(
            [
                dist_L * cell_u,
                dist_L * cell_v,
                dist_L * cell_r,
            ],
            dim=1,
        ) * float(p.scale_factor)

        e_u, e_v = _build_tangent_basis(
            rays_ds_n,
            circular_horizontal=bool(p.circular_horizontal),
        )
        rot = torch.stack([e_u, e_v, rays_ds_n], dim=-1)
        rot_flat = rot.permute(0, 2, 3, 1, 4).reshape(-1, 3, 3)
        quat_flat = _rotmat_to_quat_wxyz(rot_flat)
        quaternions = quat_flat.reshape(b, int(rays_ds_n.shape[-2]), int(rays_ds_n.shape[-1]), 4)
        quaternions = quaternions.permute(0, 3, 1, 2)[:, :, None].repeat(1, 1, L, 1, 1)

        colors = img_ds[:, :, None].repeat(1, 1, L, 1, 1).clamp(0.0, 1.0)

        opacities = torch.full(
            (b, 1, L, 1, 1),
            float(p.opacity_init),
            device=device,
            dtype=torch.float32,
        )

        base = PanoGaussianBaseValues(
            rays=rays_L,
            inv_distance=inv_dist_L,
            angular_cell=angular_cell,
            scales=scales,
            quaternions=quaternions,
            colors=colors,
            opacities=opacities,
        )

        inv_dist_1 = inv_dist_L[:, :, 0]
        feat = torch.cat([2.0 * img_ds - 1.0, 2.0 * inv_dist_1 - 1.0, rays_ds_n], dim=1)

        return PanoInitializerOutput(
            gaussian_base_values=base,
            feature_input=feat,
            global_scale=global_scale,
            grid_cell_size=0.5 * (angular_cell[:, 0:1] + angular_cell[:, 1:2]),
        )


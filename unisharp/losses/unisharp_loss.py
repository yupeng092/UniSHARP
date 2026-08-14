from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass
import math
import torch.nn.functional as F

from unisharp import DEFAULT_MAX_DEPTH_M
from unisharp.utils import linalg


def _masked_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    if m.dtype != x.dtype:
        m = m.to(dtype=x.dtype)
    while m.ndim < x.ndim:
        m = m.unsqueeze(1)
    m_expanded = m.expand_as(x)
    return (x * m_expanded).sum() / m_expanded.sum().clamp(min=1.0)


def _finite_masked_mean_flat(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    mask = valid.to(device=x.device, dtype=torch.bool) & torch.isfinite(x)
    x_safe = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    safe = torch.where(mask, x_safe, torch.zeros_like(x_safe))
    return safe.sum() / mask.to(dtype=x.dtype).sum().clamp(min=1.0)


def _finite_abs_mean(x: torch.Tensor) -> torch.Tensor:
    mask = torch.isfinite(x)
    x_safe = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    safe_abs = torch.where(mask, x_safe.abs(), torch.zeros_like(x_safe))
    return safe_abs.sum() / mask.to(dtype=x.dtype).sum().clamp(min=1.0)


_ERP_PROJECTION_MODELS = {"erp", "spherical", "equirect", "equirectangular"}
_FISHEYE_PROJECTION_MODELS = {"fisheye624", "opencv_fisheye"}


def _tv_l1(img: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros((), device=img.device, dtype=img.dtype)
    dx = (img[..., :, 1:] - img[..., :, :-1]).abs().mean() if int(img.shape[-1]) > 1 else zero
    dy = (img[..., 1:, :] - img[..., :-1, :]).abs().mean() if int(img.shape[-2]) > 1 else zero
    return dx + dy


def _tv_l1_circular_h(img: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros((), device=img.device, dtype=img.dtype)
    dx = (torch.roll(img, shifts=-1, dims=-1) - img).abs().mean() if int(img.shape[-1]) > 1 else zero
    dy = (img[..., 1:, :] - img[..., :-1, :]).abs().mean() if int(img.shape[-2]) > 1 else zero
    return dx + dy


def _checkerboard_l1_5d(x: torch.Tensor, *, circular_h: bool) -> torch.Tensor:
    if x.ndim != 5:
        raise ValueError(f"Expected [B,C,L,H,W], got {tuple(x.shape)}")
    if int(x.shape[-2]) < 2 or int(x.shape[-1]) < 2:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    x = x.to(dtype=torch.float32)
    if bool(circular_h):
        top = x[..., :-1, :]
        bottom = x[..., 1:, :]
        response = top - torch.roll(top, shifts=-1, dims=-1) - bottom + torch.roll(bottom, shifts=-1, dims=-1)
    else:
        response = x[..., :-1, :-1] - x[..., :-1, 1:] - x[..., 1:, :-1] + x[..., 1:, 1:]
    return _finite_abs_mean(response)


def _delta_grid_checkerboard_loss(delta_grid: torch.Tensor, *, circular_h: bool) -> torch.Tensor:
    if delta_grid.ndim != 5 or int(delta_grid.shape[1]) < 14:
        raise ValueError(f"Expected delta grid [B,14,L,H,W], got {tuple(delta_grid.shape)}")
    delta = delta_grid.to(dtype=torch.float32)
    parts = [
        delta[:, 3:6],
        0.1 * delta[:, 10:13],
        delta[:, 13:14],
    ]
    return torch.stack([_checkerboard_l1_5d(part, circular_h=circular_h) for part in parts]).mean()


def _avg_pool2d_circular_h(x: torch.Tensor, kernel_size: int, stride: int) -> torch.Tensor:
    if kernel_size <= 1 and stride <= 1:
        return x
    x = F.pad(x, (kernel_size - 1, 0, 0, 0), mode="circular")
    return F.avg_pool2d(x, kernel_size=kernel_size, stride=stride)


def _resize_max_side(img: torch.Tensor, max_side: int, *, mode: str = "bilinear") -> torch.Tensor:
    if max_side <= 0:
        return img
    h, w = int(img.shape[-2]), int(img.shape[-1])
    ms = max(h, w)
    if ms <= max_side:
        return img
    scale = float(max_side) / float(ms)
    nh = max(1, int(math.floor(h * scale)))
    nw = max(1, int(math.floor(w * scale)))
    if mode in ("bilinear", "bicubic"):
        return F.interpolate(img, size=(nh, nw), mode=mode, align_corners=False)
    return F.interpolate(img, size=(nh, nw), mode=mode)


def _gram_matrix(fmap: torch.Tensor) -> torch.Tensor:
    b, c, h, w = fmap.shape
    x = fmap.reshape(b, c, h * w)
    return x @ x.transpose(1, 2)


class _ResNet50Perceptual(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        try:
            from torchvision.models import resnet50, ResNet50_Weights

            net = resnet50(weights=ResNet50_Weights.DEFAULT)
        except Exception:
            from torchvision.models import resnet50

            net = resnet50(pretrained=True)

        net.eval()
        net.requires_grad_(False)

        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("_mean", mean, persistent=False)
        self.register_buffer("_std", std, persistent=False)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = x.clamp(0.0, 1.0)
        x = (x - self._mean) / self._std
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return [f1, f2, f3, f4]


def _to_linear_rgb(img_srgb: torch.Tensor) -> torch.Tensor:
    from unisharp.utils.color_space import sRGB2linearRGB

    return sRGB2linearRGB(img_srgb.clamp(0.0, 1.0))


@dataclass
class UnisharpLossWeights:

    lambda_color: float = 1.0
    lambda_alpha: float = 1.5
    lambda_percep: float = 3.0
    lambda_depth: float = 0.5
    lambda_tv: float = 1.0
    lambda_grad: float = 1.0
    lambda_delta: float = 0.0
    lambda_delta_rho: float = 0.0
    lambda_splat: float = 0.0
    lambda_edge_splat: float = 0.0
    lambda_grid: float = 0.0
    lambda_grad_img: float = 0.2
    lambda_edge_rgb: float = 0.0


class UnisharpLoss(nn.Module):

    SUPERVISION_MAX_DEPTH_M: float = DEFAULT_MAX_DEPTH_M

    def __init__(
        self,
        weights: UnisharpLossWeights | None = None,
        *,
        grad_sigma: float = 1e-2,
        grad_eps: float = 1e-2,
        delta_clip: float = 10.0,
        raw_delta_clip: float = 400.0,
        raw_delta_rho_clip: float = 5.0,
        alpha_tail_min: float = 0.99,
        alpha_tail_weight: float = 0.0,
        splat_sigma_min: float = 1e-1,
        splat_sigma_max: float = 1e2,
        edge_splat_sigma_max: float = 2.0,
        depth_edge_log_threshold: float = 0.05,
        depth_edge_dilate_px: int = 2,
        percep_max_side: int = 384,
        grad_img_scales: int = 4,
        grad_img_circular_h: bool = True,
    ) -> None:
        super().__init__()
        self.w = weights or UnisharpLossWeights()
        self.grad_sigma = float(grad_sigma)
        self.grad_eps = float(grad_eps)
        self.delta_clip = float(delta_clip)
        self.raw_delta_clip = float(raw_delta_clip)
        self.raw_delta_rho_clip = float(raw_delta_rho_clip)
        self.alpha_tail_min = float(alpha_tail_min)
        self.alpha_tail_weight = float(alpha_tail_weight)
        self.splat_sigma_min = float(splat_sigma_min)
        self.splat_sigma_max = float(splat_sigma_max)
        self.edge_splat_sigma_max = float(edge_splat_sigma_max)
        self.depth_edge_log_threshold = float(depth_edge_log_threshold)
        self.depth_edge_dilate_px = int(depth_edge_dilate_px)
        self.percep_max_side = int(percep_max_side)
        self.grad_img_scales = int(grad_img_scales)
        self.grad_img_circular_h = bool(grad_img_circular_h)

        sobel_kx = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3)
        sobel_ky = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        ).view(1, 1, 3, 3)
        self.register_buffer("_sobel_kx", sobel_kx, persistent=False)
        self.register_buffer("_sobel_ky", sobel_ky, persistent=False)

        self._percep_net: nn.Module | None = None
        if self.w.lambda_percep > 0:
            self._percep_net = _ResNet50Perceptual()

    @staticmethod
    def _flatten_gaussian_xyz(x: torch.Tensor | None, gauss_grid_shape: tuple[int, int, int] | None = None) -> torch.Tensor | None:
        if not torch.is_tensor(x):
            return None
        if x.ndim == 5:
            return x.permute(0, 2, 3, 4, 1).flatten(1, 3)
        if x.ndim == 3 and int(x.shape[-1]) == 3:
            return x
        if x.ndim == 2 and gauss_grid_shape is not None:
            return x.unsqueeze(-1)
        return None

    @staticmethod
    def _flatten_gaussian_quat(
        x: torch.Tensor | None,
        gauss_grid_shape: tuple[int, int, int] | None = None,
    ) -> torch.Tensor | None:
        if not torch.is_tensor(x):
            return None
        if x.ndim == 5 and int(x.shape[1]) == 4:
            return x.permute(0, 2, 3, 4, 1).flatten(1, 3)
        if x.ndim == 3 and int(x.shape[-1]) == 4:
            return x
        if x.ndim == 2 and gauss_grid_shape is not None:
            return x.unsqueeze(-1)
        return None

    @staticmethod
    def _flatten_gaussian_scalar(
        x: torch.Tensor | None,
        gauss_grid_shape: tuple[int, int, int] | None = None,
    ) -> torch.Tensor | None:
        if not torch.is_tensor(x):
            return None
        if x.ndim == 5:
            return x[:, 0].flatten(1)
        if x.ndim == 4:
            return x.flatten(1)
        if x.ndim == 3 and int(x.shape[-1]) == 1:
            return x[..., 0]
        if x.ndim == 2:
            return x
        return None

    @staticmethod
    def _central_disparity_gradient(inv_depth: torch.Tensor, *, circular_h: bool) -> torch.Tensor:
        if circular_h:
            gx = 0.5 * (torch.roll(inv_depth, shifts=-1, dims=-1) - torch.roll(inv_depth, shifts=1, dims=-1)).abs()
        else:
            padded_x = F.pad(inv_depth, (1, 1, 0, 0), mode="replicate")
            gx = 0.5 * (padded_x[..., 2:] - padded_x[..., :-2]).abs()
        padded_y = F.pad(inv_depth, (0, 0, 1, 1), mode="replicate")
        gy = 0.5 * (padded_y[..., 2:, :] - padded_y[..., :-2, :]).abs()
        return torch.sqrt(gx * gx + gy * gy + 1e-12)

    @staticmethod
    def _sample_map_at_uv(feat: torch.Tensor, u: torch.Tensor, v: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        b, _, h, w = feat.shape
        valid_bool = valid.to(dtype=torch.bool) & torch.isfinite(u) & torch.isfinite(v)
        u_safe = torch.where(valid_bool, u, torch.zeros_like(u)).clamp(0.0, float(max(w - 1, 0)))
        v_safe = torch.where(valid_bool, v, torch.zeros_like(v)).clamp(0.0, float(max(h - 1, 0)))
        grid_x = (u_safe / max(float(w - 1), 1.0)) * 2.0 - 1.0
        grid_y = (v_safe / max(float(h - 1), 1.0)) * 2.0 - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1).view(b, -1, 1, 2)
        sampled = F.grid_sample(feat, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        return sampled[:, 0, :, 0] * valid_bool.to(dtype=feat.dtype)

    @staticmethod
    def _expand_camera_params(camera_params: torch.Tensor, *, batch_size: int, device: torch.device) -> torch.Tensor:
        params = camera_params.to(device=device, dtype=torch.float32)
        if params.ndim == 1:
            params = params.unsqueeze(0)
        if int(params.shape[0]) == 1 and int(batch_size) > 1:
            params = params.expand(int(batch_size), -1)
        return params

    @staticmethod
    def _project_fisheye624_points_px_stable(
        pts: torch.Tensor,
        camera_params: torch.Tensor,
        *,
        image_h: int,
        image_w: int,
        finite: torch.Tensor,
        require_in_bounds: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b, n, _ = pts.shape
        params = UnisharpLoss._expand_camera_params(camera_params, batch_size=b, device=pts.device)
        x, y, z = pts.unbind(dim=-1)
        radius = torch.linalg.vector_norm(pts, dim=-1).clamp(min=1e-6)
        front = z > (radius * 1e-4).clamp(min=1e-4)
        projectable = finite & front

        safe_pts = torch.zeros_like(pts)
        safe_pts[..., 2] = 1.0
        pts_proj = torch.where(projectable.unsqueeze(-1), pts, safe_pts)
        x, y, z = pts_proj.unbind(dim=-1)
        z_safe = z.clamp(min=1e-4)

        ab = torch.stack([x / z_safe, y / z_safe], dim=-1)
        r = torch.sqrt((ab * ab).sum(dim=-1, keepdim=True) + 1e-12)
        theta = torch.atan(r)
        unit_ab = ab / r

        coeffs = params[:, 4:10].reshape(b, 1, 6)
        theta_powers = torch.cat([theta.pow(3 + i * 2) for i in range(6)], dim=-1)
        theta_distorted = theta + (theta_powers * coeffs).sum(dim=-1, keepdim=True)
        uv_dist = theta_distorted * unit_ab

        p0 = params[..., -6].reshape(b, 1)
        p1 = params[..., -5].reshape(b, 1)
        xr = uv_dist[..., 0]
        yr = uv_dist[..., 1]
        xr_sq = xr.square()
        yr_sq = yr.square()
        rd_sq = xr_sq + yr_sq
        uv_x = uv_dist[..., 0] + (2.0 * xr_sq + rd_sq) * p0 + 2.0 * xr * yr * p1
        uv_y = uv_dist[..., 1] + (2.0 * yr_sq + rd_sq) * p1 + 2.0 * xr * yr * p0

        s0 = params[..., -4].reshape(b, 1)
        s1 = params[..., -3].reshape(b, 1)
        s2 = params[..., -2].reshape(b, 1)
        s3 = params[..., -1].reshape(b, 1)
        rd_4 = rd_sq.square()
        uv_x = uv_x + s0 * rd_sq + s1 * rd_4
        uv_y = uv_y + s2 * rd_sq + s3 * rd_4

        if int(params.shape[-1]) == 15:
            fx = fy = params[..., 0:1]
            cx = params[..., 1:2]
            cy = params[..., 2:3]
        else:
            fx = params[..., 0:1]
            fy = params[..., 1:2]
            cx = params[..., 2:3]
            cy = params[..., 3:4]
        u = uv_x * fx + cx
        v = uv_y * fy + cy
        valid = projectable & torch.isfinite(u) & torch.isfinite(v)
        if require_in_bounds:
            valid = valid & (u >= 0.0) & (u <= float(image_w - 1)) & (v >= 0.0) & (v <= float(image_h - 1))
        return u, v, valid, radius

    @staticmethod
    def _project_points_px(
        points: torch.Tensor,
        *,
        projection_model: str | None,
        image_h: int,
        image_w: int,
        intrinsics: torch.Tensor | None = None,
        camera_params: torch.Tensor | None = None,
        require_in_bounds: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pts_raw = points.to(dtype=torch.float32)
        finite = torch.isfinite(pts_raw).all(dim=-1)
        pts = torch.nan_to_num(pts_raw, nan=0.0, posinf=0.0, neginf=0.0)
        b, n, _ = pts.shape
        x, y, z = pts.unbind(dim=-1)
        model = (projection_model or "pinhole").lower()

        if model in _ERP_PROJECTION_MODELS:
            radius_sq_raw = (pts * pts).sum(dim=-1)
            direction_valid = finite & (radius_sq_raw > 1e-12)
            safe_pts = torch.zeros_like(pts)
            safe_pts[..., 2] = 1.0
            pts_erp = torch.where(direction_valid.unsqueeze(-1), pts, safe_pts)
            x, y, z = pts_erp.unbind(dim=-1)
            radius_sq = (pts_erp * pts_erp).sum(dim=-1)
            radius = torch.sqrt(radius_sq + 1e-12)
            horizontal_sq = x.square() + z.square()
            horizontal = torch.sqrt(horizontal_sq + 1e-12)
            pole_angle_eps = max(1e-4, 0.5 * math.pi / float(max(image_h, image_w, 1)))
            lon_valid = horizontal > radius * pole_angle_eps
            lon_x = torch.where(lon_valid, x, torch.zeros_like(x))
            lon_z = torch.where(lon_valid, z, torch.ones_like(z))
            lon = torch.atan2(lon_x, lon_z)
            pitch_down = torch.atan2(y, horizontal)
            u = (lon / (2.0 * math.pi) + 0.5) * float(max(image_w, 1)) - 0.5
            v = (0.5 + pitch_down / math.pi) * float(max(image_h, 1)) - 0.5
            valid = direction_valid & lon_valid
            valid = (
                valid
                & torch.isfinite(u)
                & torch.isfinite(v)
                & (u >= 0.0)
                & (u <= float(image_w - 1))
                & (v >= 0.0)
                & (v <= float(image_h - 1))
            )
            return u, v, valid, radius.clamp(min=1e-6)

        if model in _FISHEYE_PROJECTION_MODELS and torch.is_tensor(camera_params):
            return UnisharpLoss._project_fisheye624_points_px_stable(
                pts,
                camera_params,
                image_h=image_h,
                image_w=image_w,
                finite=finite,
                require_in_bounds=require_in_bounds,
            )

        valid = finite & (z > 1e-4)
        if not torch.is_tensor(intrinsics):
            fx = torch.full((b, 1), float(max(image_w, image_h)), device=pts.device, dtype=torch.float32)
            fy = fx.clone()
            cx = torch.full((b, 1), 0.5 * float(max(image_w - 1, 1)), device=pts.device, dtype=torch.float32)
            cy = torch.full((b, 1), 0.5 * float(max(image_h - 1, 1)), device=pts.device, dtype=torch.float32)
        else:
            k = intrinsics.to(device=pts.device, dtype=torch.float32)
            if k.ndim == 2:
                k = k.unsqueeze(0)
            if int(k.shape[0]) == 1 and b > 1:
                k = k.expand(b, -1, -1)
            fx = k[:, 0, 0:1]
            fy = k[:, 1, 1:2]
            cx = k[:, 0, 2:3]
            cy = k[:, 1, 2:3]
        z_safe = z.clamp(min=1e-4)
        u = fx * (x / z_safe) + cx
        v = fy * (y / z_safe) + cy
        valid = valid & torch.isfinite(u) & torch.isfinite(v)
        if require_in_bounds:
            valid = valid & (u >= 0.0) & (u <= float(image_w - 1)) & (v >= 0.0) & (v <= float(image_h - 1))
        return u, v, valid, z_safe

    def _projected_sigma_px(
        self,
        *,
        gaussian_scales: torch.Tensor,
        gaussian_quaternions: torch.Tensor | None,
        gaussian_mean_vectors: torch.Tensor,
        valid: torch.Tensor,
        projection_model: str | None,
        image_h: int,
        image_w: int,
        intrinsics: torch.Tensor | None = None,
        camera_params: torch.Tensor | None = None,
        projected_scale_factor: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        scales = self._flatten_gaussian_xyz(gaussian_scales)
        quats = self._flatten_gaussian_quat(gaussian_quaternions)
        means = self._flatten_gaussian_xyz(gaussian_mean_vectors)
        if scales is None or means is None:
            return torch.zeros_like(valid, dtype=torch.float32)
        valid = valid.to(dtype=torch.bool) & torch.isfinite(scales).all(dim=-1) & torch.isfinite(means).all(dim=-1)
        scales = torch.nan_to_num(scales.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0).abs()
        means = torch.nan_to_num(means.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
        model = (projection_model or "pinhole").lower()
        if model in _ERP_PROJECTION_MODELS:
            radius = torch.norm(means, dim=-1).clamp(min=1e-4)
            sigma_u = scales[..., 0] / radius * (float(max(image_w, 1)) / (2.0 * math.pi))
            sigma_v = scales[..., 1] / radius * (float(max(image_h, 1)) / math.pi)
            sigma_px = torch.maximum(sigma_u.square(), sigma_v.square())
            valid = valid & torch.isfinite(sigma_px)
            sigma_px = torch.nan_to_num(sigma_px, nan=0.0, posinf=0.0, neginf=0.0)
            return torch.where(valid, sigma_px, torch.zeros_like(sigma_px))

        if quats is not None and tuple(quats.shape[:2]) == tuple(means.shape[:2]):
            quats = torch.nan_to_num(quats.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
            quat_norm = quats.norm(dim=-1, keepdim=True)
            valid = valid & torch.isfinite(quats).all(dim=-1) & (quat_norm.squeeze(-1) > 1e-8)
            quats = quats / quat_norm.clamp(min=1e-8)
            rotations = linalg.rotation_matrices_from_quaternions(quats)
            tangent_scales = scales[..., :2]
            tangent_rotations = rotations[..., :, :2]
            axis_offsets = (tangent_rotations * tangent_scales[..., None, :]).transpose(-1, -2)
            axis_points = means[:, :, None, :] + axis_offsets
            u0, v0, valid0, _ = self._project_points_px(
                means,
                projection_model=projection_model,
                image_h=image_h,
                image_w=image_w,
                intrinsics=intrinsics,
                camera_params=camera_params,
                require_in_bounds=False,
            )
            b, n, axis_count, _ = axis_points.shape
            u1, v1, valid1, _ = self._project_points_px(
                axis_points.reshape(b, n * axis_count, 3),
                projection_model=projection_model,
                image_h=image_h,
                image_w=image_w,
                intrinsics=intrinsics,
                camera_params=camera_params,
                require_in_bounds=False,
            )
            u1 = u1.reshape(b, n, axis_count)
            v1 = v1.reshape(b, n, axis_count)
            valid1 = valid1.reshape(b, n, axis_count)
            du = u1 - u0[..., None]
            dv = v1 - v0[..., None]
            if (projection_model or "pinhole").lower() in _ERP_PROJECTION_MODELS:
                width = float(max(image_w, 1))
                du = torch.remainder(du + 0.5 * width, width) - 0.5 * width
            cov_xx = (du * du).sum(dim=-1)
            cov_xy = (du * dv).sum(dim=-1)
            cov_yy = (dv * dv).sum(dim=-1)
            trace = cov_xx + cov_yy
            disc = (cov_xx - cov_yy).square() + 4.0 * cov_xy.square()
            sigma_px = 0.5 * (trace + (disc.clamp(min=0.0) + 1e-12).sqrt())
            valid = valid & valid0 & valid1.all(dim=-1) & torch.isfinite(sigma_px)
            sigma_px = torch.nan_to_num(sigma_px, nan=0.0, posinf=0.0, neginf=0.0)
            return torch.where(valid, sigma_px, torch.zeros_like(sigma_px))

        sigma_screen_3d = scales[..., :2].to(dtype=torch.float32).abs().amax(dim=-1).clamp(min=1e-8)
        if model in {"fisheye624", "opencv_fisheye"} and torch.is_tensor(camera_params):
            params = camera_params.to(device=means.device, dtype=torch.float32)
            if params.ndim == 1:
                params = params.unsqueeze(0)
            if int(params.shape[0]) == 1 and int(means.shape[0]) > 1:
                params = params.expand(int(means.shape[0]), -1)
            if int(params.shape[-1]) == 15:
                focal = params[:, 0:1].clamp(min=1.0)
            else:
                focal = 0.5 * (params[:, 0:1] + params[:, 1:2]).clamp(min=1.0)
            radius = torch.norm(means, dim=-1).clamp(min=1e-4)
            sigma_px = (sigma_screen_3d / radius * focal).square()
        elif torch.is_tensor(intrinsics):
            k = intrinsics.to(device=means.device, dtype=torch.float32)
            if k.ndim == 2:
                k = k.unsqueeze(0)
            if int(k.shape[0]) == 1 and int(means.shape[0]) > 1:
                k = k.expand(int(means.shape[0]), -1, -1)
            focal = 0.5 * (k[:, 0, 0:1] + k[:, 1, 1:2]).clamp(min=1.0)
            depth = means[..., 2].clamp(min=1e-4)
            sigma_px = (sigma_screen_3d / depth * focal).square()
        else:
            depth = torch.norm(means, dim=-1).clamp(min=1e-4)
            sigma_px = sigma_screen_3d / depth
            if torch.is_tensor(projected_scale_factor):
                sigma_px = sigma_px * projected_scale_factor.to(device=sigma_px.device, dtype=sigma_px.dtype)
            elif projected_scale_factor is not None:
                sigma_px = sigma_px * float(projected_scale_factor)
            sigma_px = sigma_px.square()
        valid = valid & torch.isfinite(sigma_px)
        sigma_px = torch.nan_to_num(sigma_px, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.where(valid, sigma_px, torch.zeros_like(sigma_px))

    def _depth_edge_band(
        self,
        depth_m: torch.Tensor,
        valid_weight: torch.Tensor,
        *,
        circular_h: bool,
    ) -> torch.Tensor:
        depth = depth_m.to(dtype=torch.float32)
        if depth.ndim == 3:
            depth = depth.unsqueeze(1)
        valid = torch.isfinite(depth) & (depth > 0.0) & (valid_weight[:, :1].to(dtype=torch.float32) > 0.5)
        log_depth = torch.where(valid, depth.clamp(min=1e-4).log(), torch.zeros_like(depth))

        if bool(circular_h):
            right = torch.roll(log_depth, shifts=-1, dims=-1)
            valid_right = valid & torch.roll(valid, shifts=-1, dims=-1)
            edge_x = (right - log_depth).abs() > float(self.depth_edge_log_threshold)
            edge_x = edge_x & valid_right
        else:
            edge_x = torch.zeros_like(valid)
            edge_x[..., :, :-1] = (
                (log_depth[..., :, 1:] - log_depth[..., :, :-1]).abs() > float(self.depth_edge_log_threshold)
            ) & valid[..., :, 1:] & valid[..., :, :-1]

        edge_y = torch.zeros_like(valid)
        edge_y[..., :-1, :] = (
            (log_depth[..., 1:, :] - log_depth[..., :-1, :]).abs() > float(self.depth_edge_log_threshold)
        ) & valid[..., 1:, :] & valid[..., :-1, :]
        edge = (edge_x | edge_y).to(dtype=torch.float32)

        radius = max(int(self.depth_edge_dilate_px), 0)
        if radius <= 0:
            return edge
        kernel = 2 * radius + 1
        if bool(circular_h):
            edge = F.pad(edge, (radius, radius, 0, 0), mode="circular")
            edge = F.pad(edge, (0, 0, radius, radius), mode="constant", value=0.0)
            return F.max_pool2d(edge, kernel_size=kernel, stride=1)
        return F.max_pool2d(edge, kernel_size=kernel, stride=1, padding=radius)

    def _ray_cell_sigma(
        self,
        *,
        gaussian_scales: torch.Tensor,
        gaussian_mean_vectors: torch.Tensor,
        gaussian_angular_cell: torch.Tensor,
        gauss_grid_shape: tuple[int, int, int] | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        scales = self._flatten_gaussian_xyz(gaussian_scales, gauss_grid_shape)
        means = self._flatten_gaussian_xyz(gaussian_mean_vectors, gauss_grid_shape)
        if scales is None or means is None:
            return None, None
        if not torch.is_tensor(gaussian_angular_cell):
            return None, None
        cell = gaussian_angular_cell.to(device=scales.device, dtype=torch.float32)
        if cell.ndim != 5 or int(cell.shape[1]) != 2:
            return None, None
        if gauss_grid_shape is None:
            return None, None
        l, h, w = (int(gauss_grid_shape[0]), int(gauss_grid_shape[1]), int(gauss_grid_shape[2]))
        if tuple(cell.shape[-2:]) != (h, w):
            return None, None
        if int(cell.shape[2]) == 1 and l > 1:
            cell = cell.expand(-1, -1, l, -1, -1)
        elif int(cell.shape[2]) != l:
            return None, None
        cell_flat = cell.permute(0, 2, 3, 4, 1).flatten(1, 3)
        if int(cell_flat.shape[1]) != int(scales.shape[1]):
            return None, None
        radius = torch.linalg.norm(means.to(dtype=torch.float32), dim=-1, keepdim=True).clamp(min=1e-4)
        tangent = scales[..., :2].to(dtype=torch.float32).abs()
        sigma_cells = (tangent / radius / cell_flat.clamp(min=1e-6)).square()
        valid = torch.isfinite(sigma_cells).all(dim=-1) & torch.isfinite(radius.squeeze(-1))
        sigma_cells = torch.nan_to_num(sigma_cells, nan=0.0, posinf=0.0, neginf=0.0)
        return sigma_cells, valid

    def _dynamic_splat_sigma_limits(
        self,
        *,
        sigma_proj: torch.Tensor,
        projection_model: str | None,
        image_h: int,
        image_w: int,
        intrinsics: torch.Tensor | None = None,
        camera_params: torch.Tensor | None = None,
        projected_scale_factor: float | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del projection_model, image_h, image_w, intrinsics, camera_params, projected_scale_factor
        return (
            torch.as_tensor(self.splat_sigma_min, device=sigma_proj.device, dtype=sigma_proj.dtype),
            torch.as_tensor(self.splat_sigma_max, device=sigma_proj.device, dtype=sigma_proj.dtype),
        )

    def _sanitize_supervision_depth(self, depth_m: torch.Tensor, *, clamp_max: bool = True) -> torch.Tensor:
        depth = depth_m.to(torch.float32)
        valid = torch.isfinite(depth) & (depth > 0.0)
        depth = torch.where(valid, depth, torch.zeros_like(depth))
        if bool(valid.any().item()):
            depth = depth.clone()
            if bool(clamp_max):
                depth[valid] = depth[valid].clamp(min=1e-4, max=float(self.SUPERVISION_MAX_DEPTH_M))
            else:
                depth[valid] = depth[valid].clamp(min=1e-4)
        return depth

    def _sobel_gradient_loss_erp(
        self,
        pred_depth_m: torch.Tensor,
        gt_depth_m: torch.Tensor,
        depth_weight: torch.Tensor,
        circular_h: bool | None = None,
    ) -> torch.Tensor:
        dtype = pred_depth_m.dtype
        device = pred_depth_m.device

        kx = self._sobel_kx.to(dtype=dtype, device=device)  # type: ignore[attr-defined]
        ky = self._sobel_ky.to(dtype=dtype, device=device)  # type: ignore[attr-defined]

        log_pred = torch.log(pred_depth_m.clamp(min=1e-4))
        log_gt = torch.log(gt_depth_m.clamp(min=1e-4))
        log_diff = log_pred - log_gt

        mask = depth_weight.to(dtype=dtype).clamp(min=0.0, max=1.0)
        valid_mask = (mask > 0.5).to(dtype=dtype)
        log_diff = torch.where(valid_mask > 0.5, log_diff, torch.zeros_like(log_diff))

        total = torch.zeros((), device=device, dtype=dtype)
        n_computed = 0

        use_circular_h = self.grad_img_circular_h if circular_h is None else bool(circular_h)
        ones_kernel = torch.ones((1, 1, 3, 3), device=device, dtype=dtype)
        for _s in range(self.grad_img_scales):
            if min(log_diff.shape[-2:]) < 4:
                break

            if use_circular_h:
                padded = F.pad(log_diff, (1, 1, 0, 0), mode="circular")
                padded = F.pad(padded, (0, 0, 1, 1), mode="reflect")
                padded_mask = F.pad(valid_mask, (1, 1, 0, 0), mode="circular")
                padded_mask = F.pad(padded_mask, (0, 0, 1, 1), mode="replicate")
            else:
                padded = F.pad(log_diff, (1, 1, 1, 1), mode="reflect")
                padded_mask = F.pad(valid_mask, (1, 1, 1, 1), mode="replicate")

            gx = F.conv2d(padded, kx)
            gy = F.conv2d(padded, ky)
            grad_mag = torch.sqrt(gx * gx + gy * gy + 1e-8)

            stencil_valid = (F.conv2d(padded_mask, ones_kernel) >= 8.999).to(dtype=dtype)
            n_valid = stencil_valid.sum().clamp(min=1.0)
            total = total + (grad_mag * stencil_valid).sum() / n_valid
            n_computed += 1

            if _s < self.grad_img_scales - 1:
                if use_circular_h:
                    pooled_mask = _avg_pool2d_circular_h(valid_mask, kernel_size=2, stride=2)
                    pooled_diff = _avg_pool2d_circular_h(log_diff * valid_mask, kernel_size=2, stride=2)
                else:
                    pooled_mask = F.avg_pool2d(valid_mask, kernel_size=2, stride=2)
                    pooled_diff = F.avg_pool2d(log_diff * valid_mask, kernel_size=2, stride=2)
                log_diff = pooled_diff / pooled_mask.clamp(min=1e-6)
                valid_mask = (pooled_mask > 0.999).to(dtype=dtype)
                log_diff = torch.where(valid_mask > 0.5, log_diff, torch.zeros_like(log_diff))

        if n_computed == 0:
            return torch.zeros((), device=device, dtype=dtype)
        return total / float(n_computed)

    def _sobel_xy_rgb(self, img: torch.Tensor, *, circular_h: bool) -> tuple[torch.Tensor, torch.Tensor]:
        channels = int(img.shape[1])
        kx = self._sobel_kx.to(dtype=img.dtype, device=img.device).expand(channels, 1, 3, 3)  # type: ignore[attr-defined]
        ky = self._sobel_ky.to(dtype=img.dtype, device=img.device).expand(channels, 1, 3, 3)  # type: ignore[attr-defined]
        if bool(circular_h):
            padded = F.pad(img, (1, 1, 0, 0), mode="circular")
            padded = F.pad(padded, (0, 0, 1, 1), mode="reflect")
        else:
            padded = F.pad(img, (1, 1, 1, 1), mode="reflect")
        return (
            F.conv2d(padded, kx, groups=channels),
            F.conv2d(padded, ky, groups=channels),
        )

    def _edge_rgb_gradient_loss(
        self,
        pred_rgb_linear: torch.Tensor,
        gt_rgb_linear: torch.Tensor,
        valid_weight: torch.Tensor,
        depth_edge_band: torch.Tensor | None,
        *,
        circular_h: bool,
    ) -> torch.Tensor:
        dtype = pred_rgb_linear.dtype
        device = pred_rgb_linear.device
        pred = pred_rgb_linear.to(dtype=torch.float32)
        gt = gt_rgb_linear.to(device=device, dtype=torch.float32)
        weight = valid_weight.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)[:, :1]

        pred_gx, pred_gy = self._sobel_xy_rgb(pred, circular_h=circular_h)
        gt_gx, gt_gy = self._sobel_xy_rgb(gt, circular_h=circular_h)
        gt_mag = torch.sqrt(gt_gx.square() + gt_gy.square() + 1e-8).mean(dim=1, keepdim=True)

        flat = gt_mag.detach().flatten(2)
        mean = flat.mean(dim=-1, keepdim=True)[..., None]
        std = flat.std(dim=-1, keepdim=True, unbiased=False)[..., None]
        rgb_edge = (gt_mag.detach() > (mean + 0.5 * std).clamp(min=0.02)).to(dtype=torch.float32)

        if torch.is_tensor(depth_edge_band):
            edge_boost = depth_edge_band.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
            if tuple(edge_boost.shape[-2:]) != tuple(gt_mag.shape[-2:]):
                edge_boost = F.interpolate(edge_boost, size=gt_mag.shape[-2:], mode="nearest")
            edge_weight = rgb_edge * (1.0 + edge_boost[:, :1])
        else:
            edge_weight = rgb_edge

        ones_kernel = torch.ones((1, 1, 3, 3), device=device, dtype=torch.float32)
        if bool(circular_h):
            padded_weight = F.pad(weight, (1, 1, 0, 0), mode="circular")
            padded_weight = F.pad(padded_weight, (0, 0, 1, 1), mode="replicate")
        else:
            padded_weight = F.pad(weight, (1, 1, 1, 1), mode="replicate")
        stencil_valid = (F.conv2d(padded_weight, ones_kernel) >= 8.999).to(dtype=torch.float32)

        diff = (pred_gx - gt_gx).abs() + (pred_gy - gt_gy).abs()
        diff = diff.mean(dim=1, keepdim=True)
        final_weight = edge_weight * stencil_valid
        return (diff * final_weight).sum().to(dtype=dtype) / final_weight.sum().clamp(min=1.0).to(dtype=dtype)

    def forward(
        self,
        pred_rgb_linear: torch.Tensor,
        pred_alpha: torch.Tensor,
        pred_depth_m: torch.Tensor,
        gt_rgb_u8: torch.Tensor,
        gt_depth_m: torch.Tensor,
        pred_depth2_m: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        depth_mask: torch.Tensor | None = None,
        delta_xy: torch.Tensor | None = None,
        delta_rho: torch.Tensor | None = None,
        delta_grid: torch.Tensor | None = None,
        gaussian_scales: torch.Tensor | None = None,
        gaussian_quaternions: torch.Tensor | None = None,
        gaussian_mean_vectors: torch.Tensor | None = None,
        gaussian_base_mean_vectors: torch.Tensor | None = None,
        gaussian_angular_cell: torch.Tensor | None = None,
        gaussian_opacities: torch.Tensor | None = None,
        gauss_grid_shape: tuple[int, int, int] | None = None,
        projected_scale_factor: float | torch.Tensor | None = None,
        projection_model: str | None = None,
        projection_intrinsics: torch.Tensor | None = None,
        projection_camera_params: torch.Tensor | None = None,
        apply_color: bool = True,
        apply_alpha: bool = True,
        apply_depth: bool = True,
        apply_percep: bool = False,
        apply_tv: bool = True,
        apply_grad: bool = True,
        apply_delta: bool = True,
        apply_splat: bool = True,
        apply_grad_img: bool = True,
        grad_img_circular_h: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}
        circular_h = bool(grad_img_circular_h) if grad_img_circular_h is not None else False

        gt_rgb = gt_rgb_u8.to(pred_rgb_linear.device).float() / 255.0
        gt_rgb_linear = _to_linear_rgb(gt_rgb)
        pred_depth_m = self._sanitize_supervision_depth(pred_depth_m.to(pred_rgb_linear.device), clamp_max=False)
        if pred_depth2_m is not None:
            pred_depth2_m = self._sanitize_supervision_depth(pred_depth2_m.to(pred_rgb_linear.device), clamp_max=False)
        gt_depth_raw = self._sanitize_supervision_depth(gt_depth_m.to(pred_rgb_linear.device))
        depth_valid = torch.isfinite(gt_depth_raw) & (gt_depth_raw > 0.0)
        gt_depth = gt_depth_raw.clamp(min=1e-4)

        if mask is None:
            m = torch.ones_like(pred_alpha)
        else:
            m = mask.to(pred_rgb_linear.device).to(pred_rgb_linear.dtype)
        depth_weight = depth_valid.to(dtype=pred_depth_m.dtype) * m[:, :1].to(dtype=pred_depth_m.dtype)
        if depth_mask is not None:
            depth_weight = depth_weight * depth_mask.to(pred_rgb_linear.device).to(dtype=pred_depth_m.dtype)[:, :1]
        pred_rgb_rendered = pred_rgb_linear.clamp(0.0, 1.0)

        if apply_color and self.w.lambda_color > 0:
            color_l1 = (pred_rgb_rendered - gt_rgb_linear).abs()
            losses["color"] = _masked_mean(color_l1, m)
        else:
            losses["color"] = torch.zeros((), device=pred_rgb_linear.device)

        if apply_alpha and self.w.lambda_alpha > 0:
            a = pred_alpha.clamp(1e-6, 1.0 - 1e-6)
            with torch.autocast(device_type=a.device.type, enabled=False):
                alpha_bce = F.binary_cross_entropy(
                    a.to(dtype=torch.float32),
                    torch.ones_like(a, dtype=torch.float32),
                    reduction="none",
                )
                alpha_loss = _masked_mean(alpha_bce, m)
                alpha_tail_min = torch.as_tensor(
                    self.alpha_tail_min,
                    device=a.device,
                    dtype=torch.float32,
                ).clamp(min=0.0, max=1.0)
                alpha_tail_weight = torch.as_tensor(
                    max(0.0, self.alpha_tail_weight),
                    device=a.device,
                    dtype=torch.float32,
                )
                if self.alpha_tail_min > 0.0 and self.alpha_tail_weight > 0.0:
                    tail = F.relu(alpha_tail_min - a.to(dtype=torch.float32))
                    tail = tail / alpha_tail_min.clamp(min=1e-6)
                    tail_mask = (m[:, :1].to(dtype=torch.bool)) & (tail > 0.0)
                    alpha_loss = alpha_loss + alpha_tail_weight * _finite_masked_mean_flat(tail, tail_mask)
            losses["alpha"] = alpha_loss.to(dtype=pred_rgb_linear.dtype)
        else:
            losses["alpha"] = torch.zeros((), device=pred_rgb_linear.device)

        if apply_depth and self.w.lambda_depth > 0:
            w_depth = depth_weight
            inv_pred1 = 1.0 / pred_depth_m.clamp(min=1e-4)
            inv_gt = torch.zeros_like(inv_pred1)
            inv_gt[depth_valid] = 1.0 / gt_depth[depth_valid]
            depth_abs = (inv_pred1 - inv_gt).abs()
            losses["depth"] = _masked_mean(depth_abs, w_depth)
        else:
            losses["depth"] = torch.zeros((), device=pred_rgb_linear.device)

        if apply_tv and self.w.lambda_tv > 0 and (pred_depth2_m is not None):
            inv2 = 1.0 / pred_depth2_m.clamp(min=1e-4)
            losses["tv"] = _tv_l1_circular_h(inv2) if circular_h else _tv_l1(inv2)
        else:
            losses["tv"] = torch.zeros((), device=pred_rgb_linear.device)

        image_h, image_w = int(pred_depth_m.shape[-2]), int(pred_depth_m.shape[-1])
        projection_points = self._flatten_gaussian_xyz(gaussian_mean_vectors, gauss_grid_shape)
        projected_u = projected_v = None
        projected_valid = None
        if projection_points is not None:
            projected_u, projected_v, projected_valid, _projected_depth = self._project_points_px(
                projection_points,
                projection_model=projection_model,
                image_h=image_h,
                image_w=image_w,
                intrinsics=projection_intrinsics,
                camera_params=projection_camera_params,
            )

        if apply_grad and self.w.lambda_grad > 0:
            inv1 = 1.0 / pred_depth_m.clamp(min=1e-4)
            op_flat = self._flatten_gaussian_scalar(gaussian_opacities, gauss_grid_shape)
            if projected_u is not None and projected_v is not None and projected_valid is not None and op_flat is not None:
                grad_map = self._central_disparity_gradient(inv1, circular_h=circular_h)
                grad_at_gauss = self._sample_map_at_uv(grad_map, projected_u, projected_v, projected_valid)
                penalty = 1.0 - torch.exp(
                    -(1.0 / max(self.grad_sigma, 1e-8)) * F.relu(grad_at_gauss - self.grad_eps)
                )
                weight = projected_valid & torch.isfinite(grad_at_gauss) & torch.isfinite(op_flat)
                mask_at_gauss = self._sample_map_at_uv(m[:, :1], projected_u, projected_v, projected_valid)
                weight = weight & (mask_at_gauss > 0.5)
                grad_value = op_flat.to(dtype=penalty.dtype).clamp(0, 1) * penalty
                losses["grad"] = _finite_masked_mean_flat(grad_value, weight)
            else:
                raise RuntimeError(
                    "L_grad requires gaussian_mean_vectors, gaussian_opacities, "
                    "gauss_grid_shape, and projection metadata. The old "
                    "pred_alpha image-space fallback is disabled for ray-local training."
                )
        else:
            losses["grad"] = torch.zeros((), device=pred_rgb_linear.device)

        if apply_grad_img and self.w.lambda_grad_img > 0:
            losses["grad_img"] = self._sobel_gradient_loss_erp(
                pred_depth_m=pred_depth_m,
                gt_depth_m=gt_depth,
                depth_weight=depth_weight,
                circular_h=grad_img_circular_h,
            )
        else:
            losses["grad_img"] = torch.zeros((), device=pred_rgb_linear.device)

        if apply_color and self.w.lambda_edge_rgb > 0:
            depth_edge_for_rgb = self._depth_edge_band(gt_depth, depth_weight, circular_h=circular_h)
            losses["edge_rgb"] = self._edge_rgb_gradient_loss(
                pred_rgb_linear=pred_rgb_rendered,
                gt_rgb_linear=gt_rgb_linear,
                valid_weight=m,
                depth_edge_band=depth_edge_for_rgb,
                circular_h=circular_h,
            )
        else:
            losses["edge_rgb"] = torch.zeros((), device=pred_rgb_linear.device)

        if apply_delta and self.w.lambda_delta > 0:
            if delta_xy is not None:
                dx = F.relu(delta_xy[:, 0:1].abs() - self.raw_delta_clip)
                dy = F.relu(delta_xy[:, 1:2].abs() - self.raw_delta_clip)
                losses["delta"] = (dx + dy).mean()
            else:
                del gaussian_base_mean_vectors
                raise RuntimeError(
                    "L_delta requires raw delta_xy in ray-local training. The old "
                    "screen-space pixel displacement fallback is disabled."
                )
        else:
            losses["delta"] = torch.zeros((), device=pred_rgb_linear.device)

        if apply_delta and self.w.lambda_delta_rho > 0 and delta_rho is not None:
            dz = delta_rho.to(device=pred_rgb_linear.device, dtype=pred_rgb_linear.dtype)
            finite = torch.isfinite(dz)
            dz_safe = torch.nan_to_num(dz, nan=0.0, posinf=0.0, neginf=0.0)
            penalty = F.relu(dz_safe.abs() - self.raw_delta_rho_clip)
            penalty = torch.where(finite, penalty, torch.zeros_like(penalty))
            losses["delta_rho"] = penalty.sum() / finite.to(dtype=penalty.dtype).sum().clamp(min=1.0)
        else:
            losses["delta_rho"] = torch.zeros((), device=pred_rgb_linear.device)

        if self.w.lambda_grid > 0 and torch.is_tensor(delta_grid):
            losses["grid"] = _delta_grid_checkerboard_loss(
                delta_grid.to(device=pred_rgb_linear.device),
                circular_h=circular_h,
            ).to(dtype=pred_rgb_linear.dtype)
        else:
            losses["grid"] = torch.zeros((), device=pred_rgb_linear.device)

        if apply_splat and self.w.lambda_splat > 0:
            if gaussian_scales is None:
                raise RuntimeError("L_splat requires gaussian_scales for projected screen-space variance.")
            if gaussian_mean_vectors is None or projected_valid is None:
                raise RuntimeError(
                    "L_splat requires gaussian_mean_vectors and projection metadata "
                    "to compute projected screen-space variance."
                )
            sigma_proj = self._projected_sigma_px(
                gaussian_scales=gaussian_scales,
                gaussian_quaternions=gaussian_quaternions,
                gaussian_mean_vectors=gaussian_mean_vectors,
                valid=projected_valid,
                projection_model=projection_model,
                image_h=image_h,
                image_w=image_w,
                intrinsics=projection_intrinsics,
                camera_params=projection_camera_params,
                projected_scale_factor=projected_scale_factor,
            )
            valid_splat = projected_valid & torch.isfinite(sigma_proj)
            splat_sigma_min = torch.as_tensor(
                self.splat_sigma_min,
                device=sigma_proj.device,
                dtype=sigma_proj.dtype,
            )
            splat_sigma_max = torch.as_tensor(
                self.splat_sigma_max,
                device=sigma_proj.device,
                dtype=sigma_proj.dtype,
            )
            lower_penalty = F.relu(splat_sigma_min - sigma_proj)
            upper_penalty = F.relu(sigma_proj - splat_sigma_max)
            splat_penalty = lower_penalty + upper_penalty
            losses["splat"] = _finite_masked_mean_flat(splat_penalty, valid_splat)
        else:
            sigma_proj = None
            valid_splat = None
            losses["splat"] = torch.zeros((), device=pred_rgb_linear.device)

        if apply_splat and self.w.lambda_edge_splat > 0:
            if gaussian_scales is None:
                raise RuntimeError("L_edge_splat requires gaussian_scales for projected screen-space variance.")
            if gaussian_mean_vectors is None or projected_valid is None:
                raise RuntimeError(
                    "L_edge_splat requires gaussian_mean_vectors and projection metadata "
                    "to sample source depth-edge bands."
                )
            if sigma_proj is None or valid_splat is None:
                sigma_proj = self._projected_sigma_px(
                    gaussian_scales=gaussian_scales,
                    gaussian_quaternions=gaussian_quaternions,
                    gaussian_mean_vectors=gaussian_mean_vectors,
                    valid=projected_valid,
                    projection_model=projection_model,
                    image_h=image_h,
                    image_w=image_w,
                    intrinsics=projection_intrinsics,
                    camera_params=projection_camera_params,
                    projected_scale_factor=projected_scale_factor,
                )
                valid_splat = projected_valid & torch.isfinite(sigma_proj)
            edge_band = self._depth_edge_band(gt_depth, depth_weight, circular_h=circular_h)
            edge_at_gauss = self._sample_map_at_uv(edge_band, projected_u, projected_v, projected_valid)
            edge_valid = valid_splat & torch.isfinite(edge_at_gauss) & (edge_at_gauss > 0.5)
            edge_sigma_max = torch.as_tensor(
                self.edge_splat_sigma_max,
                device=sigma_proj.device,
                dtype=sigma_proj.dtype,
            )
            losses["edge_splat"] = _finite_masked_mean_flat(F.relu(sigma_proj - edge_sigma_max), edge_valid)
        else:
            losses["edge_splat"] = torch.zeros((), device=pred_rgb_linear.device)

        zero = torch.zeros((), device=pred_rgb_linear.device)
        losses["percep_feat"] = zero
        losses["percep_gram"] = zero
        if apply_percep and self.w.lambda_percep > 0 and (self._percep_net is not None):
            from unisharp.utils.color_space import linearRGB2sRGB

            pred_srgb = linearRGB2sRGB(pred_rgb_rendered.to(torch.float32)).clamp(0, 1)
            gt_srgb = gt_rgb.clamp(0, 1)

            pred_srgb = _resize_max_side(pred_srgb, self.percep_max_side, mode="bilinear")
            gt_srgb = _resize_max_side(gt_srgb, self.percep_max_side, mode="bilinear")

            feats_p = self._percep_net(pred_srgb)
            feats_g = self._percep_net(gt_srgb)
            loss_feat_total = torch.zeros((), device=pred_rgb_linear.device)
            loss_gram_total = torch.zeros((), device=pred_rgb_linear.device)
            for fp, fg in zip(feats_p, feats_g):
                d, h, w = fp.shape[1], fp.shape[2], fp.shape[3]
                lam_gram = 10.0 / float(max(1, d * d))
                lam_feat = 1.0 / float(max(1, d * h * w))
                diff = (fp - fg).pow(2)
                loss_feat = (diff.sum(dim=[1, 2, 3]) * lam_feat).mean()
                gram_norm = float(max(1, h * w))
                gp = _gram_matrix(fp) / gram_norm
                gg = _gram_matrix(fg) / gram_norm
                loss_gram = ((gp - gg).pow(2).sum(dim=[1, 2]) * lam_gram).mean()
                loss_feat_total = loss_feat_total + loss_feat
                loss_gram_total = loss_gram_total + loss_gram
            layer_count = float(max(1, len(feats_p)))
            losses["percep_feat"] = loss_feat_total / layer_count
            losses["percep_gram"] = loss_gram_total / layer_count
            losses["percep"] = losses["percep_feat"] + losses["percep_gram"]
        else:
            losses["percep"] = torch.zeros((), device=pred_rgb_linear.device)

        losses["total"] = (
            self.w.lambda_color * losses["color"]
            + self.w.lambda_alpha * losses["alpha"]
            + self.w.lambda_percep * losses["percep"]
            + self.w.lambda_depth * losses["depth"]
            + self.w.lambda_tv * losses["tv"]
            + self.w.lambda_grad * losses["grad"]
            + self.w.lambda_grad_img * losses["grad_img"]
            + self.w.lambda_edge_rgb * losses["edge_rgb"]
            + self.w.lambda_delta * losses["delta"]
            + self.w.lambda_delta_rho * losses["delta_rho"]
            + self.w.lambda_splat * losses["splat"]
            + self.w.lambda_edge_splat * losses["edge_splat"]
            + self.w.lambda_grid * losses["grid"]
        )
        return losses


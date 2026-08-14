from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from unisharp.utils import math as math_utils
from unisharp.utils.color_space import ColorSpace, sRGB2linearRGB
from unisharp.utils.gaussians import Gaussians3D

from .gaussian_initializer import PanoGaussianBaseValues, _build_tangent_basis
from .unisharp_params import DeltaFactor


def _safe_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp(min=eps)


def _infer_circular_horizontal(rays_2d: torch.Tensor) -> bool:
    if rays_2d.ndim != 4 or int(rays_2d.shape[-1]) < 4:
        return False
    r = _safe_normalize(rays_2d.detach().to(dtype=torch.float32))
    seam_dot = (r[..., 0] * r[..., -1]).sum(dim=1).clamp(-1.0, 1.0)
    seam_cross = torch.linalg.vector_norm(torch.cross(r[..., 0], r[..., -1], dim=1), dim=1)
    seam_angle = torch.atan2(seam_cross, seam_dot)
    inner_dot = (r[..., :-1] * r[..., 1:]).sum(dim=1).clamp(-1.0, 1.0)
    inner_cross = torch.linalg.vector_norm(torch.cross(r[..., :-1], r[..., 1:], dim=1), dim=1)
    inner_angle = torch.atan2(inner_cross, inner_dot)
    typical = torch.nanmedian(inner_angle).clamp(min=1e-6)
    return bool(torch.isfinite(seam_angle).all() and float(torch.nanmedian(seam_angle).item()) < 4.0 * float(typical.item()))


class PanoGaussianComposer(nn.Module):

    def __init__(
        self,
        delta_factor: DeltaFactor,
        min_scale: float,
        max_scale: float,
        color_activation_type: math_utils.ActivationType,
        opacity_activation_type: math_utils.ActivationType,
        color_space: ColorSpace,
        base_scale_on_predicted_mean: bool = True,
        scale_factor: int = 1,
        delta_rho_limit: float = 2.0,
    ) -> None:
        super().__init__()
        self.delta_factor = delta_factor
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.color_activation_type = color_activation_type
        self.opacity_activation_type = opacity_activation_type
        self.color_space = color_space
        self.base_scale_on_predicted_mean = bool(base_scale_on_predicted_mean)
        self.scale_factor = int(max(1, scale_factor))
        self.delta_rho_limit = float(delta_rho_limit)

        (
            self.ray_scale_min_factor,
            self.ray_scale_max_factor,
        ) = self._ray_cell_coverage_scale_bounds()
        (
            self.ray_radial_min_factor,
            self.ray_radial_max_factor,
        ) = self._ray_radial_scale_bounds()
        (
            self.ray_anisotropy_min_factor,
            self.ray_anisotropy_max_factor,
        ) = self._ray_tangent_anisotropy_bounds()
        (
            self._ray_scale_const_a,
            self._ray_scale_const_b,
        ) = self._get_scale_activation_constant(
            self.ray_scale_max_factor,
            self.ray_scale_min_factor,
        )
        (
            self._ray_radial_const_a,
            self._ray_radial_const_b,
        ) = self._get_scale_activation_constant(
            self.ray_radial_max_factor,
            self.ray_radial_min_factor,
        )
        (
            self._ray_anisotropy_const_a,
            self._ray_anisotropy_const_b,
        ) = self._get_scale_activation_constant(
            self.ray_anisotropy_max_factor,
            self.ray_anisotropy_min_factor,
        )

    @staticmethod
    def _get_scale_activation_constant(max_scale: float, min_scale: float) -> tuple[float, float]:
        constant_a = (max_scale - min_scale) / (1 - min_scale) / (max_scale - 1)
        constant_b = math_utils.inverse_sigmoid(
            torch.tensor((1.0 - min_scale) / (max_scale - min_scale))
        ).item()
        return constant_a, constant_b

    @staticmethod
    def _ray_cell_coverage_scale_bounds() -> tuple[float, float]:
        min_factor = 0.1
        max_factor = 5.0
        return float(min_factor), float(max_factor)

    @staticmethod
    def _ray_radial_scale_bounds() -> tuple[float, float]:
        return 0.1, 5.0

    @staticmethod
    def _ray_tangent_anisotropy_bounds() -> tuple[float, float]:
        max_factor = 4.0
        min_factor = 1.0 / max_factor
        return float(min_factor), float(max_factor)

    @staticmethod
    def _smooth_scale_delta(
        learned_delta_scale: torch.Tensor,
        *,
        circular_horizontal: bool,
        kernel_size: int = 9,
    ) -> torch.Tensor:
        k = int(kernel_size)
        if k <= 1:
            return learned_delta_scale
        pad = k // 2
        b, c, l, h, w = learned_delta_scale.shape
        x = learned_delta_scale.permute(0, 2, 1, 3, 4).reshape(b * l, c, h, w)
        if bool(circular_horizontal):
            x = F.pad(x, (pad, pad, 0, 0), mode="circular")
            x = F.pad(x, (0, 0, pad, pad), mode="replicate")
        else:
            x = F.pad(x, (pad, pad, pad, pad), mode="replicate")
        x = F.avg_pool2d(x, kernel_size=k, stride=1)
        return x.reshape(b, l, c, h, w).permute(0, 2, 1, 3, 4).contiguous()

    def apply_delta_rho(self, learned_delta_rho: torch.Tensor) -> torch.Tensor:
        raw_d_rho = self.delta_factor.z * learned_delta_rho
        limit = float(self.delta_rho_limit)
        if limit <= 0.0:
            return raw_d_rho
        return limit * torch.tanh(raw_d_rho / limit)

    def apply_scale_factor(self, learned_delta_scale: torch.Tensor) -> torch.Tensor:
        tangent = (self.ray_scale_max_factor - self.ray_scale_min_factor) * torch.sigmoid(
            self._ray_scale_const_a * self.delta_factor.scale * learned_delta_scale[:, 0:1]
            + self._ray_scale_const_b
        ) + self.ray_scale_min_factor
        anisotropy = (
            (self.ray_anisotropy_max_factor - self.ray_anisotropy_min_factor)
            * torch.sigmoid(
                self._ray_anisotropy_const_a
                * self.delta_factor.scale
                * learned_delta_scale[:, 1:2]
                + self._ray_anisotropy_const_b
            )
            + self.ray_anisotropy_min_factor
        )
        radial = (self.ray_radial_max_factor - self.ray_radial_min_factor) * torch.sigmoid(
            self._ray_radial_const_a * self.delta_factor.scale * learned_delta_scale[:, 2:3]
            + self._ray_radial_const_b
        ) + self.ray_radial_min_factor

        sqrt_anisotropy = anisotropy.sqrt()
        tangent_u = (tangent * sqrt_anisotropy).clamp(
            min=self.ray_scale_min_factor,
            max=self.ray_scale_max_factor,
        )
        tangent_v = (tangent / sqrt_anisotropy.clamp(min=1e-6)).clamp(
            min=self.ray_scale_min_factor,
            max=self.ray_scale_max_factor,
        )
        return torch.cat([tangent_u, tangent_v, radial], dim=1)

    def forward(
        self,
        delta: torch.Tensor,
        base_values: PanoGaussianBaseValues,
        global_scale: torch.Tensor | None = None,
        flatten_output: bool = True,
    ) -> Gaussians3D:
        if delta.ndim != 5 or int(delta.shape[1]) != 14:
            raise ValueError(f"Expected delta shape [B,14,L,H,W], got {tuple(delta.shape)}")
        if int(delta.shape[2]) != int(base_values.rays.shape[2]):
            raise ValueError(
                "Delta layer count must match base Gaussian layers, "
                f"got delta L={int(delta.shape[2])} base L={int(base_values.rays.shape[2])}"
            )
        base_h, base_w = int(base_values.rays.shape[-2]), int(base_values.rays.shape[-1])
        delta_h, delta_w = int(delta.shape[-2]), int(delta.shape[-1])
        if (delta_h, delta_w) != (base_h, base_w):
            raise ValueError(
                "Delta grid must match base Gaussian grid before composition, "
                f"got delta={(delta_h, delta_w)} base={(base_h, base_w)}"
            )

        mean = self._forward_mean(base_values, delta[:, 0:3])

        base_scales = base_values.scales
        if self.base_scale_on_predicted_mean:
            radius_pred = mean.norm(dim=1, keepdim=True).clamp(min=1e-4)
            radius_base_inv = base_values.inv_distance.clamp(min=1e-6)
            scale_ratio = (radius_pred * radius_base_inv).clamp(min=1e-6)
            base_scales = base_scales * scale_ratio
        scale_delta = self._smooth_scale_delta(
            delta[:, 3:6],
            circular_horizontal=_infer_circular_horizontal(base_values.rays[:, :, 0]),
        )
        scales = self._scale_activation(base_scales, scale_delta)
        quat_raw = base_values.quaternions + self.delta_factor.quaternion * delta[:, 6:10]
        quat_norm = quat_raw.norm(dim=1, keepdim=True)
        base_quats = base_values.quaternions / base_values.quaternions.norm(dim=1, keepdim=True).clamp(min=1e-8)
        quats = torch.where(quat_norm > 1e-8, quat_raw / quat_norm.clamp(min=1e-8), base_quats)
        colors = self._color_activation(base_values.colors, delta[:, 10:13])
        opacities = self._opacity_activation(base_values.opacities, delta[:, 13:14])

        if flatten_output:
            mean = mean.permute(0, 2, 3, 4, 1).flatten(1, 3)
            scales = scales.permute(0, 2, 3, 4, 1).flatten(1, 3)
            quats = quats.permute(0, 2, 3, 4, 1).flatten(1, 3)
            colors = colors.permute(0, 2, 3, 4, 1).flatten(1, 3)
            opacities = opacities.squeeze(1).flatten(1)

        if global_scale is not None:
            mean = mean * global_scale[:, None, None]
            scales = scales * global_scale[:, None, None]

        return Gaussians3D(
            mean_vectors=mean,
            singular_values=scales,
            quaternions=quats,
            colors=colors,
            opacities=opacities,
        )

    def _forward_mean(self, base: PanoGaussianBaseValues, learned_delta: torch.Tensor) -> torch.Tensor:
        rays = _safe_normalize(base.rays)
        b, _, l, h, w = rays.shape
        rays_2d = rays[:, :, 0]
        e1_2d, e2_2d = _build_tangent_basis(
            rays_2d,
            circular_horizontal=_infer_circular_horizontal(rays_2d),
        )
        e1 = e1_2d[:, :, None].expand(b, -1, l, h, w)
        e2 = e2_2d[:, :, None].expand(b, -1, l, h, w)
        angular_cell = base.angular_cell.to(device=learned_delta.device, dtype=learned_delta.dtype)
        if angular_cell.ndim != 5 or int(angular_cell.shape[1]) != 2:
            raise ValueError(f"Expected angular_cell shape [B,2,1,H,W], got {tuple(angular_cell.shape)}")
        cell_u = angular_cell[:, 0:1]
        cell_v = angular_cell[:, 1:2]

        du = self.delta_factor.xy * learned_delta[:, 0:1] * cell_u
        dv = self.delta_factor.xy * learned_delta[:, 1:2] * cell_v
        d_rho = self.apply_delta_rho(learned_delta[:, 2:3])

        rho0 = base.inv_distance
        rho = F.softplus(math_utils.inverse_softplus(rho0.clamp(min=1e-6)) + d_rho)
        r = 1.0 / (rho + 1e-4)

        ray_new = rays + du * e1 + dv * e2
        ray_new = _safe_normalize(ray_new)

        return r * ray_new

    def _scale_activation(self, base: torch.Tensor, learned_delta: torch.Tensor) -> torch.Tensor:
        scale_factor = self.apply_scale_factor(learned_delta)
        return base * scale_factor

    def _color_activation(self, base: torch.Tensor, learned_delta: torch.Tensor) -> torch.Tensor:
        if self.color_activation_type == "sigmoid":
            base = torch.clamp(base, min=0.01, max=0.99)
        elif self.color_activation_type in ("exp", "softplus"):
            base = torch.clamp(base, min=0.01)

        activation = math_utils.create_activation_pair(self.color_activation_type)
        colors: torch.Tensor = activation.forward(
            activation.inverse(base) + self.delta_factor.color * learned_delta
        )
        if self.color_space == "linearRGB":
            colors = sRGB2linearRGB(colors)
        return colors

    def _opacity_activation(self, base: torch.Tensor, learned_delta: torch.Tensor) -> torch.Tensor:
        activation = math_utils.create_activation_pair(self.opacity_activation_type)
        return activation.forward(
            activation.inverse(base) + self.delta_factor.opacity * learned_delta
        )


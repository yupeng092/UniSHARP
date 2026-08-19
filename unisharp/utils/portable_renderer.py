"""Differentiable pinhole Gaussian renderer for CPU and Ascend NPU training.

This implementation deliberately uses ordinary PyTorch operations.  It is a
small, portable training renderer rather than a replacement for the CUDA
``gsplat`` renderer: it supports calibrated pinhole cameras only and uses a
bounded number of splats per image/tile to keep CPU and NPU memory practical.
The discrete visibility, tile assignment and depth sorting steps are expected
to be non-differentiable; gradients flow through every selected splat's
geometry, opacity and colour.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from unisharp.utils.gaussians import BackgroundColor, Gaussians3D


class RenderingOutputs(NamedTuple):
    color: torch.Tensor
    depth: torch.Tensor
    alpha: torch.Tensor


def _quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    q = F.normalize(quaternion.to(dtype=torch.float32), dim=-1, eps=1e-8)
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack(
        (
            1 - 2 * (y.square() + z.square()),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x.square() + z.square()),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x.square() + y.square()),
        ),
        dim=-1,
    ).reshape(-1, 3, 3)


class PortableGaussianRenderer(nn.Module):
    """Tile-based differentiable renderer usable on ``cpu`` and ``npu``.

    It has the same call signature as :class:`GSplatRenderer`, allowing the
    pinhole UniSHARP training path to select it without changing the trainer.
    Fisheye and panoramic samples still require their CUDA renderers.
    """

    def __init__(
        self,
        *,
        background_color: BackgroundColor = "black",
        tile_size: int = 16,
        tile_span: int = 5,
        max_gaussians: int = 8192,
        max_gaussians_per_tile: int = 128,
        min_variance: float = 0.30,
        sigma_cutoff: float = 3.0,
        max_radius: float = 32.0,
        near: float = 0.01,
        far: float = 100.0,
    ) -> None:
        super().__init__()
        if background_color not in {"black", "white"}:
            raise ValueError("Portable renderer supports only black or white backgrounds.")
        if int(tile_size) < 1 or int(tile_span) < 1 or int(tile_span) % 2 == 0:
            raise ValueError("tile_size must be positive and tile_span must be a positive odd number.")
        if int(max_gaussians) < 0 or int(max_gaussians_per_tile) < 0:
            raise ValueError("Gaussian limits must be non-negative.")
        self.background_color = background_color
        self.tile_size = int(tile_size)
        self.tile_span = int(tile_span)
        self.max_gaussians = int(max_gaussians)
        self.max_gaussians_per_tile = int(max_gaussians_per_tile)
        self.min_variance = float(min_variance)
        self.sigma_cutoff = float(sigma_cutoff)
        self.max_radius = float(max_radius)
        self.near = float(near)
        self.far = float(far)

    def _project(
        self,
        gaussians: Gaussians3D,
        extrinsic: torch.Tensor,
        intrinsics: torch.Tensor,
        width: int,
        height: int,
    ) -> dict[str, torch.Tensor]:
        means = gaussians.mean_vectors.to(dtype=torch.float32)
        scales = gaussians.singular_values.to(dtype=torch.float32).clamp_min(1e-7)
        quaternions = gaussians.quaternions.to(dtype=torch.float32)
        colours = gaussians.colors.to(dtype=torch.float32)
        opacity = gaussians.opacities.to(dtype=torch.float32).reshape(-1).clamp(0.0, 0.999)
        if means.ndim != 2 or means.shape[-1] != 3:
            raise ValueError(f"Expected flattened Gaussian means [N,3], got {tuple(means.shape)}")

        rotation = extrinsic[:3, :3].to(dtype=torch.float32)
        translation = extrinsic[:3, 3].to(dtype=torch.float32)
        camera_xyz = means @ rotation.transpose(0, 1) + translation
        x, y, z = camera_xyz.unbind(dim=-1)
        fx, fy = intrinsics[0, 0].to(torch.float32), intrinsics[1, 1].to(torch.float32)
        cx, cy = intrinsics[0, 2].to(torch.float32), intrinsics[1, 2].to(torch.float32)
        inv_z = z.clamp_min(self.near).reciprocal()
        u = fx * x * inv_z + cx
        v = fy * y * inv_z + cy

        basis = _quaternion_to_matrix(quaternions) * scales[:, None, :]
        cov_world = basis @ basis.transpose(-1, -2)
        cov_camera = rotation[None] @ cov_world @ rotation.transpose(0, 1)[None]
        jacobian = torch.zeros((means.shape[0], 2, 3), device=means.device, dtype=torch.float32)
        jacobian[:, 0, 0] = fx * inv_z
        jacobian[:, 0, 2] = -fx * x * inv_z.square()
        jacobian[:, 1, 1] = fy * inv_z
        jacobian[:, 1, 2] = -fy * y * inv_z.square()
        cov_2d = jacobian @ cov_camera @ jacobian.transpose(-1, -2)
        cov_2d[:, 0, 0] = cov_2d[:, 0, 0] + self.min_variance
        cov_2d[:, 1, 1] = cov_2d[:, 1, 1] + self.min_variance
        a, b, c = cov_2d[:, 0, 0], cov_2d[:, 0, 1], cov_2d[:, 1, 1]
        determinant = (a * c - b.square()).clamp_min(1e-10)
        inverse = torch.stack((c / determinant, -b / determinant, a / determinant), dim=-1)
        largest_eigenvalue = 0.5 * (a + c + torch.sqrt(((a - c).square() + 4.0 * b.square()).clamp_min(0.0)))
        radius = (self.sigma_cutoff * torch.sqrt(largest_eigenvalue.clamp_min(0.0))).clamp(max=self.max_radius)
        visible = (
            (z > self.near)
            & (z < self.far)
            & torch.isfinite(u)
            & torch.isfinite(v)
            & torch.isfinite(inverse).all(dim=-1)
            & (u + radius >= 0.0)
            & (u - radius < float(width))
            & (v + radius >= 0.0)
            & (v - radius < float(height))
        )
        indices = torch.where(visible)[0]
        if self.max_gaussians > 0 and int(indices.numel()) > self.max_gaussians:
            keep = torch.topk(opacity[indices], k=self.max_gaussians, sorted=False).indices
            indices = indices[keep]
        return {
            "u": u[indices],
            "v": v[indices],
            "z": z[indices],
            "radius": radius[indices],
            "inverse": inverse[indices],
            "opacity": opacity[indices],
            "colour": colours[indices],
        }

    def _render_projected(
        self, projected: dict[str, torch.Tensor], *, width: int, height: int
    ) -> RenderingOutputs:
        device = projected["u"].device
        background_value = 0.0 if self.background_color == "black" else 1.0
        background = torch.full((3,), background_value, device=device, dtype=torch.float32)
        color = background[:, None, None].expand(3, height, width).clone()
        alpha_image = torch.zeros((height, width), device=device, dtype=torch.float32)
        depth_image = torch.zeros((height, width), device=device, dtype=torch.float32)
        cutoff2 = self.sigma_cutoff * self.sigma_cutoff
        u, v, radius = projected["u"], projected["v"], projected["radius"]
        half_span = self.tile_span // 2
        for y0 in range(0, height, self.tile_size):
            y1 = min(y0 + self.tile_size, height)
            for x0 in range(0, width, self.tile_size):
                x1 = min(x0 + self.tile_size, width)
                tile_x = x0 // self.tile_size
                tile_y = y0 // self.tile_size
                centre_x = torch.div(u.floor().to(torch.long), self.tile_size, rounding_mode="floor")
                centre_y = torch.div(v.floor().to(torch.long), self.tile_size, rounding_mode="floor")
                overlap = (
                    (u + radius >= float(x0))
                    & (u - radius < float(x1))
                    & (v + radius >= float(y0))
                    & (v - radius < float(y1))
                    & ((centre_x - tile_x).abs() <= half_span)
                    & ((centre_y - tile_y).abs() <= half_span)
                )
                selected = torch.where(overlap)[0]
                if self.max_gaussians_per_tile > 0 and int(selected.numel()) > self.max_gaussians_per_tile:
                    keep = torch.topk(
                        projected["opacity"][selected], k=self.max_gaussians_per_tile, sorted=False
                    ).indices
                    selected = selected[keep]
                if int(selected.numel()) == 0:
                    continue
                selected = selected[torch.argsort(projected["z"][selected])]
                yy, xx = torch.meshgrid(
                    torch.arange(y0, y1, device=device, dtype=torch.float32),
                    torch.arange(x0, x1, device=device, dtype=torch.float32),
                    indexing="ij",
                )
                dx = xx.reshape(1, -1) - projected["u"][selected, None]
                dy = yy.reshape(1, -1) - projected["v"][selected, None]
                inverse = projected["inverse"][selected]
                mahalanobis = (
                    inverse[:, 0, None] * dx.square()
                    + 2.0 * inverse[:, 1, None] * dx * dy
                    + inverse[:, 2, None] * dy.square()
                )
                alpha = projected["opacity"][selected, None] * torch.exp(-0.5 * mahalanobis)
                alpha = torch.where(mahalanobis <= cutoff2, alpha, torch.zeros_like(alpha)).clamp(0.0, 0.999)
                one_minus_alpha = 1.0 - alpha
                exclusive = torch.cumprod(
                    torch.cat((torch.ones_like(one_minus_alpha[:1]), one_minus_alpha[:-1]), dim=0), dim=0
                )
                weights = alpha * exclusive
                transmittance = one_minus_alpha.prod(dim=0)
                tile_color = weights.transpose(0, 1) @ projected["colour"][selected]
                tile_color = tile_color + transmittance[:, None] * background[None]
                tile_alpha = 1.0 - transmittance
                tile_depth = (weights * projected["z"][selected, None]).sum(dim=0)
                tile_depth = torch.where(
                    tile_alpha > 1e-6,
                    tile_depth / tile_alpha.clamp_min(1e-6),
                    torch.zeros_like(tile_depth),
                )
                color[:, y0:y1, x0:x1] = tile_color.transpose(0, 1).reshape(3, y1 - y0, x1 - x0)
                alpha_image[y0:y1, x0:x1] = tile_alpha.reshape(y1 - y0, x1 - x0)
                depth_image[y0:y1, x0:x1] = tile_depth.reshape(y1 - y0, x1 - x0)
        return RenderingOutputs(
            color=color.unsqueeze(0),
            depth=depth_image.unsqueeze(0).unsqueeze(0),
            alpha=alpha_image.unsqueeze(0).unsqueeze(0),
        )

    def forward(
        self,
        gaussians: Gaussians3D,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        image_width: int,
        image_height: int,
    ) -> RenderingOutputs:
        gaussian_batch_size = int(gaussians.mean_vectors.shape[0])
        camera_batch_size = int(extrinsics.shape[0])
        if int(intrinsics.shape[0]) != camera_batch_size:
            raise ValueError("Intrinsics and extrinsics must have the same batch size.")
        if gaussian_batch_size not in (1, camera_batch_size):
            raise ValueError("Gaussians must have batch size one or match the camera batch size.")
        outputs: list[RenderingOutputs] = []
        for index in range(camera_batch_size):
            gaussian_index = 0 if gaussian_batch_size == 1 else index
            single = Gaussians3D(*(value[gaussian_index] for value in gaussians))
            projected = self._project(
                single,
                extrinsics[index].to(dtype=torch.float32),
                intrinsics[index, :3, :3].to(dtype=torch.float32),
                int(image_width),
                int(image_height),
            )
            outputs.append(self._render_projected(projected, width=int(image_width), height=int(image_height)))
        return RenderingOutputs(
            color=torch.cat([output.color for output in outputs], dim=0),
            depth=torch.cat([output.depth for output in outputs], dim=0),
            alpha=torch.cat([output.alpha for output in outputs], dim=0),
        )

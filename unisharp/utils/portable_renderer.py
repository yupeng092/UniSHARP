"""Portable reference implementation of gsplat's classic 3D Gaussian path.

The renderer deliberately uses only ordinary PyTorch operations, so it can run
on CPU and Ascend NPU. Its pinhole path follows ``gsplat.rendering.
rasterization(..., rasterize_mode="classic")`` mathematically: covariance
projection through the perspective Jacobian, ``eps2d`` filtering, 3.33-sigma
tile bounds, front-to-back compositing, gsplat's alpha threshold/cap, and
early-transmittance termination. Tile assignment and depth sorting are
discrete, as in gsplat; selected splats remain fully differentiable.

This is intentionally a *reference* renderer, not an Ascend fused kernel.
It therefore cannot be bitwise-identical to CUDA gsplat and can be much slower
when no Gaussian limits are configured. It supports calibrated pinhole cameras
only; fisheye and panorama still require a generic-camera renderer.
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
    """Tile-based gsplat-classic reference renderer usable on ``cpu`` and ``npu``.

    It has the same call signature as :class:`GSplatRenderer`, allowing the
    pinhole UniSHARP training path to select it without changing the trainer.
    Fisheye and panoramic samples still require their CUDA renderers.
    """

    def __init__(
        self,
        *,
        background_color: BackgroundColor = "black",
        tile_size: int = 16,
        tile_span: int = 0,
        max_gaussians: int = 0,
        max_gaussians_per_tile: int = 0,
        low_pass_filter_eps: float = 1e-2,
        near: float = 0.01,
        far: float = 1e10,
    ) -> None:
        super().__init__()
        if background_color not in {"black", "white"}:
            raise ValueError("Portable renderer supports only black or white backgrounds.")
        if int(tile_size) < 1 or int(tile_span) < 0:
            raise ValueError("tile_size must be positive and tile_span must be non-negative.")
        if int(max_gaussians) < 0 or int(max_gaussians_per_tile) < 0:
            raise ValueError("Gaussian limits must be non-negative.")
        if float(low_pass_filter_eps) < 0.0:
            raise ValueError("low_pass_filter_eps must be non-negative.")
        self.background_color = background_color
        self.tile_size = int(tile_size)
        self.tile_span = int(tile_span)
        self.max_gaussians = int(max_gaussians)
        self.max_gaussians_per_tile = int(max_gaussians_per_tile)
        self.low_pass_filter_eps = float(low_pass_filter_eps)
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
        # Keep the projected mean exact, while clamping only the coordinates
        # used by the perspective Jacobian. This mirrors gsplat._persp_proj's
        # 30%-of-FoV guard against extreme off-screen covariances.
        inv_z = z.reciprocal()
        u = fx * x * inv_z + cx
        v = fy * y * inv_z + cy
        tan_fov_x = 0.5 * float(width) / fx
        tan_fov_y = 0.5 * float(height) / fy
        x_limit_positive = (float(width) - cx) / fx + 0.3 * tan_fov_x
        x_limit_negative = cx / fx + 0.3 * tan_fov_x
        y_limit_positive = (float(height) - cy) / fy + 0.3 * tan_fov_y
        y_limit_negative = cy / fy + 0.3 * tan_fov_y
        jacobian_x = z * torch.clamp(x * inv_z, min=-x_limit_negative, max=x_limit_positive)
        jacobian_y = z * torch.clamp(y * inv_z, min=-y_limit_negative, max=y_limit_positive)

        basis = _quaternion_to_matrix(quaternions) * scales[:, None, :]
        cov_world = basis @ basis.transpose(-1, -2)
        cov_camera = rotation[None] @ cov_world @ rotation.transpose(0, 1)[None]
        jacobian = torch.zeros((means.shape[0], 2, 3), device=means.device, dtype=torch.float32)
        jacobian[:, 0, 0] = fx * inv_z
        jacobian[:, 0, 2] = -fx * jacobian_x * inv_z.square()
        jacobian[:, 1, 1] = fy * inv_z
        jacobian[:, 1, 2] = -fy * jacobian_y * inv_z.square()
        # Same filter as gsplat's fully_fused_projection: eps2d is added to
        # both projected covariance eigenvalues (equivalently its diagonal).
        cov_2d = jacobian @ cov_camera @ jacobian.transpose(-1, -2)
        cov_2d[:, 0, 0] = cov_2d[:, 0, 0] + self.low_pass_filter_eps
        cov_2d[:, 1, 1] = cov_2d[:, 1, 1] + self.low_pass_filter_eps
        a, b, c = cov_2d[:, 0, 0], cov_2d[:, 0, 1], cov_2d[:, 1, 1]
        determinant = (a * c - b.square()).clamp_min(1e-10)
        inverse = torch.stack((c / determinant, -b / determinant, a / determinant), dim=-1)
        # gsplat's classic 3DGS projection uses per-axis 3.33-sigma bounds,
        # rather than the looser largest-eigenvalue circle used previously.
        radius_x = torch.ceil(3.33 * torch.sqrt(a.clamp_min(0.0)))
        radius_y = torch.ceil(3.33 * torch.sqrt(c.clamp_min(0.0)))
        visible = (
            (z > self.near)
            & (z < self.far)
            & torch.isfinite(u)
            & torch.isfinite(v)
            & torch.isfinite(inverse).all(dim=-1)
            & (u + radius_x > 0.0)
            & (u - radius_x < float(width))
            & (v + radius_y > 0.0)
            & (v - radius_y < float(height))
        )
        indices = torch.where(visible)[0]
        if self.max_gaussians > 0 and int(indices.numel()) > self.max_gaussians:
            keep = torch.topk(opacity[indices], k=self.max_gaussians, sorted=False).indices
            indices = indices[keep]
        return {
            "u": u[indices],
            "v": v[indices],
            "z": z[indices],
            "radius_x": radius_x[indices],
            "radius_y": radius_y[indices],
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
        u, v = projected["u"], projected["v"]
        radius_x, radius_y = projected["radius_x"], projected["radius_y"]
        for y0 in range(0, height, self.tile_size):
            y1 = min(y0 + self.tile_size, height)
            for x0 in range(0, width, self.tile_size):
                x1 = min(x0 + self.tile_size, width)
                # This is gsplat's tile intersection predicate: every
                # Gaussian whose 3.33-sigma axis-aligned bounds intersects a
                # tile is included. tile_span is kept as an accepted legacy
                # argument, but deliberately does not cull coverage.
                overlap = (
                    (u + radius_x > float(x0))
                    & (u - radius_x < float(x1))
                    & (v + radius_y > float(y0))
                    & (v - radius_y < float(y1))
                )
                selected = torch.where(overlap)[0]
                if self.max_gaussians_per_tile > 0 and int(selected.numel()) > self.max_gaussians_per_tile:
                    keep = torch.topk(
                        projected["opacity"][selected], k=self.max_gaussians_per_tile, sorted=False
                    ).indices
                    selected = selected[keep]
                if int(selected.numel()) == 0:
                    continue
                # gsplat groups intersecting splats by tile and sorts them by
                # increasing centre depth before classic alpha compositing.
                selected = selected[torch.argsort(projected["z"][selected])]
                yy, xx = torch.meshgrid(
                    torch.arange(y0, y1, device=device, dtype=torch.float32) + 0.5,
                    torch.arange(x0, x1, device=device, dtype=torch.float32) + 0.5,
                    indexing="ij",
                )
                pixel_count = int((y1 - y0) * (x1 - x0))
                transmittance = torch.ones((pixel_count,), device=device, dtype=torch.float32)
                done = torch.zeros((pixel_count,), device=device, dtype=torch.bool)
                tile_color = torch.zeros((pixel_count, 3), device=device, dtype=torch.float32)
                tile_depth_unnormalized = torch.zeros((pixel_count,), device=device, dtype=torch.float32)

                # Match gsplat's classic kernel exactly at the per-pixel
                # level: exp(-sigma), alpha cap/threshold, then early stop if
                # the next transmittance is <= 1e-4. The loop is slower than a
                # fused kernel but preserves the same mathematical operation.
                for gaussian_id in selected.unbind():
                    dx = xx.reshape(-1) - projected["u"][gaussian_id]
                    dy = yy.reshape(-1) - projected["v"][gaussian_id]
                    conic = projected["inverse"][gaussian_id]
                    sigma = (
                        0.5 * (conic[0] * dx.square() + conic[2] * dy.square())
                        + conic[1] * dx * dy
                    )
                    alpha = (projected["opacity"][gaussian_id] * torch.exp(-sigma)).clamp(max=0.999)
                    next_transmittance = transmittance * (1.0 - alpha)
                    eligible = (~done) & (sigma >= 0.0) & (alpha >= (1.0 / 255.0))
                    active = eligible & (next_transmittance > 1e-4)
                    weight = torch.where(active, transmittance * alpha, torch.zeros_like(alpha))
                    tile_color = tile_color + weight[:, None] * projected["colour"][gaussian_id]
                    tile_depth_unnormalized = tile_depth_unnormalized + weight * projected["z"][gaussian_id]
                    transmittance = torch.where(active, next_transmittance, transmittance)
                    done = done | (eligible & ~active)

                tile_alpha = 1.0 - transmittance
                tile_color = tile_color + transmittance[:, None] * background[None]
                tile_depth = tile_depth_unnormalized / tile_alpha.clamp_min(1e-8)
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

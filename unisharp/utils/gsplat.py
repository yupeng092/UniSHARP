
from __future__ import annotations

import os
from typing import NamedTuple

import torch
from torch import nn

from unisharp.utils import color_space as cs_utils
from unisharp.utils.gaussians import BackgroundColor, Gaussians3D


if "TORCH_CUDA_ARCH_LIST" not in os.environ and torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability(0)
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"

import gsplat  # noqa: E402  (must happen after env var setup)


class RenderingOutputs(NamedTuple):

    color: torch.Tensor
    depth: torch.Tensor
    alpha: torch.Tensor


class GSplatRenderer(nn.Module):

    color_space: cs_utils.ColorSpace
    background_color: BackgroundColor

    def __init__(
        self,
        color_space: cs_utils.ColorSpace = "sRGB",
        background_color: BackgroundColor = "black",
        low_pass_filter_eps: float = 1e-2,
    ) -> None:
        super().__init__()
        self.color_space = color_space
        self.background_color = background_color
        self.low_pass_filter_eps = low_pass_filter_eps

    def forward(
        self,
        gaussians: Gaussians3D,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        image_width: int,
        image_height: int,
    ) -> RenderingOutputs:
        gaussian_batch_size = len(gaussians.mean_vectors)
        camera_batch_size = int(extrinsics.shape[0])
        if int(intrinsics.shape[0]) != camera_batch_size:
            raise ValueError(
                f"Expected intrinsics batch to match extrinsics batch, got "
                f"{tuple(intrinsics.shape)} vs {tuple(extrinsics.shape)}"
            )
        if gaussian_batch_size not in (1, camera_batch_size):
            raise ValueError(
                f"Unsupported batch combination: gaussians={gaussian_batch_size}, cameras={camera_batch_size}. "
                "Expected either one Gaussian batch for many cameras or one-to-one batches."
            )
        outputs_list: list[RenderingOutputs] = []

        for ib in range(camera_batch_size):
            g_idx = 0 if gaussian_batch_size == 1 else ib
            means = gaussians.mean_vectors[g_idx].to(dtype=torch.float32)
            quats = gaussians.quaternions[g_idx].to(dtype=torch.float32)
            scales = gaussians.singular_values[g_idx].to(dtype=torch.float32)
            opacities = gaussians.opacities[g_idx].to(dtype=torch.float32)
            colors_in = gaussians.colors[g_idx].to(dtype=torch.float32)
            viewmats = extrinsics[ib : ib + 1].to(dtype=torch.float32)
            Ks = intrinsics[ib : ib + 1, :3, :3].to(dtype=torch.float32)

            colors, alphas, meta = gsplat.rendering.rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors_in,
                viewmats=viewmats,
                Ks=Ks,
                width=image_width,
                height=image_height,
                render_mode="RGB+D",
                rasterize_mode="classic",
                absgrad=False,
                packed=False,
                eps2d=self.low_pass_filter_eps,
            )

            rendered_color = colors[..., 0:3].permute([0, 3, 1, 2])
            rendered_depth_unnormalized = colors[..., 3:4].permute([0, 3, 1, 2])
            rendered_alpha = alphas.permute([0, 3, 1, 2])

            rendered_color = self.compose_with_background(
                rendered_color, rendered_alpha, self.background_color
            )

            if self.color_space == "sRGB":
                pass
            elif self.color_space == "linearRGB":
                rendered_color = cs_utils.linearRGB2sRGB(rendered_color)
            else:
                raise ValueError(f"Unsupported ColorSpace type: {self.color_space!r}")

            cov2d = self._conics_to_covars2d(meta["conics"])
            splats_visible_mask = meta["depths"] > 1e-2
            cov2d[~splats_visible_mask][..., 0, 0] = 1
            cov2d[~splats_visible_mask][..., 1, 1] = 1
            cov2d[~splats_visible_mask][..., 0, 1] = 0

            rendered_depth = rendered_depth_unnormalized / torch.clip(rendered_alpha, min=1e-8)

            outputs = RenderingOutputs(
                color=rendered_color,
                depth=rendered_depth,
                alpha=rendered_alpha,
            )
            outputs_list.append(outputs)

        return RenderingOutputs(
            color=torch.cat([item.color for item in outputs_list], dim=0).contiguous(),
            depth=torch.cat([item.depth for item in outputs_list], dim=0).contiguous(),
            alpha=torch.cat([item.alpha for item in outputs_list], dim=0).contiguous(),
        )

    @staticmethod
    def compose_with_background(
        rendered_rgb: torch.Tensor,
        rendered_alpha: torch.Tensor,
        background_color: BackgroundColor,
    ) -> torch.Tensor:
        if background_color == "black":
            return rendered_rgb
        elif background_color == "white":
            return rendered_rgb + (1.0 - rendered_alpha)
        elif background_color == "random_color":
            return (
                rendered_rgb
                + (1.0 - rendered_alpha)
                * torch.rand(3, dtype=rendered_rgb.dtype, device=rendered_rgb.device)[
                    None, :, None, None
                ]
            )
        elif background_color == "random_pixel":
            return rendered_rgb + (1.0 - rendered_alpha) * torch.rand_like(rendered_rgb)
        else:
            raise ValueError("Unsupported BackgroundColor type.")

    @staticmethod
    def _conics_to_covars2d(conics: torch.Tensor, eps=1e-8) -> torch.Tensor:
        a = conics[..., 0]
        b = conics[..., 1]
        c = conics[..., 2]
        det = 1 / (a * c - b**2 + eps)
        det = det.clamp(min=eps)
        covars2d = torch.zeros(*conics.shape[:-1], 2, 2, device=conics.device)
        covars2d[..., 1, 1] = a * det
        covars2d[..., 0, 0] = c * det
        covars2d[..., 0, 1] = -b * det
        covars2d[..., 1, 0] = -b * det
        covars2d = torch.nan_to_num(covars2d, nan=0.0, posinf=0.0, neginf=0.0)
        return covars2d

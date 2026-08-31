"""Ascend CANN fused 3D Gaussian renderer for UniSHARP.

This adapter targets the ``meta_gauss_render`` custom-op wheel from CANN's
``cann-recipes-embodied-ai/3d_vision/gaussian_splatting`` recipe.  It keeps
the UniSHARP renderer contract while moving projection, frustum culling,
tile/intersection construction, depth sorting and alpha compositing to Ascend
custom operators.  Only the small schedule construction is host-side, as in
the upstream recipe.

The wheel is intentionally imported lazily: CPU/CUDA users do not need CANN,
``acl`` or ``meta_gauss_render`` installed.  This backend supports calibrated
pinhole cameras only.  It is a high-performance NPU backend, not a bitwise
replacement for CUDA gsplat: the CANN recipe has its own clipping thresholds
and tile scheduling implementation.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from unisharp.utils.gaussians import BackgroundColor, Gaussians3D
from unisharp.utils.portable_renderer import RenderingOutputs


class AscendFusedRendererUnavailable(RuntimeError):
    """Raised when the CANN 3DGS custom-op wheel is unavailable or unusable."""


@dataclass(frozen=True)
class _AscendOps:
    acl: Any
    projection: Any
    build_mask: Any
    sort: Any
    render: Any
    schedule: Any


def _load_ascend_ops() -> _AscendOps:
    """Import the CANN custom-op API only when this renderer is selected."""
    try:
        acl = importlib.import_module("acl")
        meta = importlib.import_module("meta_gauss_render")
    except Exception as exc:  # pragma: no cover - depends on the NPU host.
        raise AscendFusedRendererUnavailable(
            "The Ascend fused renderer requires the CANN `meta_gauss_render` custom-op wheel. "
            "Install it with `bash scripts/install_meta_gauss_render.sh --source /path/to/"
            "cann-recipes-embodied-ai` and retry, or set NPU_RENDERER_BACKEND=portable."
        ) from exc
    required = {
        "projection_three_dims_gaussian_fused": "projection",
        "flash_gaussian_build_mask": "build_mask",
        "gaussian_sort": "sort",
        "calc_render": "render",
        "get_render_schedule": "schedule",
    }
    missing = [name for name in required if not hasattr(meta, name)]
    if missing:
        raise AscendFusedRendererUnavailable(
            "Installed meta_gauss_render is missing required CANN 3DGS APIs: " + ", ".join(missing)
        )
    return _AscendOps(
        acl=acl,
        projection=getattr(meta, "projection_three_dims_gaussian_fused"),
        build_mask=getattr(meta, "flash_gaussian_build_mask"),
        sort=getattr(meta, "gaussian_sort"),
        render=getattr(meta, "calc_render"),
        schedule=getattr(meta, "get_render_schedule"),
    )


def ascend_fused_renderer_available() -> bool:
    """Return whether the CANN fused renderer can be imported on this host."""
    try:
        _load_ascend_ops()
    except AscendFusedRendererUnavailable:
        return False
    return True


class AscendFusedGaussianRenderer(nn.Module):
    """Differentiable NPU renderer backed by CANN ``meta_gauss_render`` ops.

    UniSHARP predicts activated linear RGB and opacity values, whereas the
    CANN recipe's high-level trainer stores logit/SH parameters.  This adapter
    calls its low-level ops directly with UniSHARP's already-activated values.
    A second fused compositing pass using unit RGB derives alpha without a
    slow PyTorch raster loop; the first pass's projection/intersection/sort
    result is reused.
    """

    def __init__(
        self,
        *,
        background_color: BackgroundColor = "black",
        tile_size: int = 32,
        low_pass_filter_eps: float = 1e-2,
        near: float = 0.01,
        far: float = 1e10,
    ) -> None:
        super().__init__()
        if background_color not in {"black", "white"}:
            raise ValueError("Ascend fused renderer supports black or white backgrounds only.")
        if int(tile_size) != 32:
            raise ValueError(
                "meta_gauss_render's tested fused path uses tile_size=32; "
                f"got {tile_size}."
            )
        if float(low_pass_filter_eps) < 0.0:
            raise ValueError("low_pass_filter_eps must be non-negative.")
        self.background_color = background_color
        self.tile_size = int(tile_size)
        self.low_pass_filter_eps = float(low_pass_filter_eps)
        self.near = float(near)
        self.far = float(far)
        self._ops: _AscendOps | None = None
        self._geometry_cache: dict[tuple[str, int, int], tuple[torch.Tensor, torch.Tensor, int, int]] = {}

    @property
    def ops(self) -> _AscendOps:
        if self._ops is None:
            self._ops = _load_ascend_ops()
        return self._ops

    def _geometry(
        self, device: torch.device, width: int, height: int
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        """Cache fixed tile origins and pixel positions on the target NPU."""
        key = (str(device), int(width), int(height))
        cached = self._geometry_cache.get(key)
        if cached is not None:
            return cached
        tile_h = (int(height) + self.tile_size - 1) // self.tile_size
        tile_w = (int(width) + self.tile_size - 1) // self.tile_size
        padded_h, padded_w = tile_h * self.tile_size, tile_w * self.tile_size
        tile_grid = torch.stack(
            torch.meshgrid(
                torch.arange(0, padded_h, self.tile_size, device=device),
                torch.arange(0, padded_w, self.tile_size, device=device),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(-1, 2).contiguous()
        pixel_grid = torch.stack(
            torch.meshgrid(
                torch.arange(padded_w, device=device),
                torch.arange(padded_h, device=device),
                indexing="xy",
            ),
            dim=-1,
        )
        pixel_tiles = (
            pixel_grid.reshape(tile_h, self.tile_size, tile_w, self.tile_size, 2)
            .permute(0, 2, 1, 3, 4)
            .reshape(tile_h * tile_w, self.tile_size * self.tile_size, 2)
            .permute(0, 2, 1)
            .to(dtype=torch.float32)
            .contiguous()
        )
        value = (tile_grid, pixel_tiles, padded_h, padded_w)
        self._geometry_cache[key] = value
        return value

    def _tiles_to_image(
        self,
        values: torch.Tensor,
        *,
        padded_height: int,
        padded_width: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Convert CANN tile-major [C, tiles, pixels] output into [C,H,W]."""
        tile_h, tile_w = padded_height // self.tile_size, padded_width // self.tile_size
        image = (
            values.permute(1, 2, 0)
            .reshape(tile_h, tile_w, self.tile_size, self.tile_size, -1)
            .transpose(1, 2)
            .reshape(padded_height, padded_width, -1)
            .permute(2, 0, 1)
        )
        return image[:, :height, :width].contiguous()

    def _render_one(
        self,
        gaussians: Gaussians3D,
        *,
        extrinsic: torch.Tensor,
        intrinsics: torch.Tensor,
        width: int,
        height: int,
    ) -> RenderingOutputs:
        means = gaussians.mean_vectors.to(dtype=torch.float32).contiguous()
        quats = gaussians.quaternions.to(dtype=torch.float32).contiguous()
        scales = gaussians.singular_values.to(dtype=torch.float32).clamp_min(1e-7).contiguous()
        colors = gaussians.colors.to(dtype=torch.float32).contiguous()
        opacities = gaussians.opacities.to(dtype=torch.float32).reshape(-1).contiguous()
        if means.ndim != 2 or means.shape[-1] != 3:
            raise ValueError(f"Expected flattened Gaussian means [N,3], got {tuple(means.shape)}")
        if means.device.type != "npu":
            raise ValueError("Ascend fused renderer requires Gaussians on an `npu` device.")

        tile_grid, pixel_tiles, padded_h, padded_w = self._geometry(means.device, int(width), int(height))
        # The custom projection has explicit backward kernels for all UniSHARP
        # Gaussian attributes.  B=C=1 avoids changing the trainer's existing
        # Gaussian/camera batch semantics and reuses its per-camera loop.
        means2d, depths, conics, opacity_out, _radii, covars2d, colors_out, count = self.ops.projection(
            means.unsqueeze(0),
            colors.unsqueeze(0),
            None,
            quats.unsqueeze(0),
            scales.unsqueeze(0),
            opacities.unsqueeze(0),
            extrinsic.to(dtype=torch.float32).reshape(1, 1, 4, 4).contiguous(),
            intrinsics[:3, :3].to(dtype=torch.float32).reshape(1, 1, 3, 3).contiguous(),
            int(width),
            int(height),
            float(self.low_pass_filter_eps),
            float(self.near),
            float(self.far),
            False,
            "pinhole",
        )

        tile_sums, tile_offsets, tile_depths, tile_gaussian_ids = self.ops.build_mask(
            means2d,
            opacity_out[None, :],
            conics,
            covars2d,
            depths,
            count[None, :],
            tile_grid.to(dtype=torch.float32),
            int(width),
            int(height),
            self.tile_size,
        )
        sorted_counts = tile_offsets.squeeze(-1)[:, :, -1].reshape(-1)
        total_intersections = int(sorted_counts[0].detach().cpu())
        if total_intersections == 0:
            zeros = torch.zeros((1, 1, int(height), int(width)), device=means.device, dtype=torch.float32)
            color = zeros.expand(1, 3, -1, -1).clone()
            if self.background_color == "white":
                color = color + 1.0
            return RenderingOutputs(color=color, depth=zeros, alpha=zeros)

        sorted_offset = torch.cumsum(sorted_counts, dim=0)
        # CANN's load-balancing schedule is deliberately built on the host in
        # the official recipe, then consumed by its fused sort/render kernels.
        tile_sums_cpu = tile_sums.squeeze(-1).detach().cpu().to(torch.int64)
        vector_cores = int(self.ops.acl.get_device_capability(0, 1)[0])
        schedule = self.ops.schedule(tile_sums_cpu, vector_cores).to(device=means.device)
        max_tile_gaussians = int(torch.amax(tile_sums).detach().cpu())
        sorted_ids = self.ops.sort(
            schedule,
            tile_sums,
            tile_depths,
            tile_gaussian_ids,
            sorted_offset,
            max_tile_gaussians,
        )
        selected_ids = sorted_ids[: sorted_offset[0]]
        render_schedule = schedule[0, 0, :]
        means_2d = means2d[0, 0]
        conic = conics[0, 0]
        opacity = opacity_out[0, 0]
        depth = depths[0, 0]
        rendered_color, rendered_depth_sum = self.ops.render(
            means_2d,
            conic[0],
            conic[1],
            conic[2],
            opacity,
            colors_out[0, 0],
            depth.unsqueeze(0),
            pixel_tiles,
            render_schedule,
            selected_ids,
        )
        # The CANN kernel emits pre-multiplied RGB and depth.  Render a unit
        # RGB field with the identical projection/sort state to obtain alpha;
        # it remains differentiable and avoids restoring the Python raster loop.
        alpha_color, _ = self.ops.render(
            means_2d,
            conic[0],
            conic[1],
            conic[2],
            opacity,
            torch.ones_like(colors_out[0, 0]),
            depth.unsqueeze(0),
            pixel_tiles,
            render_schedule,
            selected_ids,
        )
        color = self._tiles_to_image(
            rendered_color,
            padded_height=padded_h,
            padded_width=padded_w,
            height=int(height),
            width=int(width),
        )
        depth_sum = self._tiles_to_image(
            rendered_depth_sum,
            padded_height=padded_h,
            padded_width=padded_w,
            height=int(height),
            width=int(width),
        )
        alpha = self._tiles_to_image(
            alpha_color,
            padded_height=padded_h,
            padded_width=padded_w,
            height=int(height),
            width=int(width),
        )[0:1].clamp(0.0, 1.0)
        if self.background_color == "white":
            color = color + (1.0 - alpha)
        return RenderingOutputs(
            color=color.unsqueeze(0),
            depth=(depth_sum / alpha.clamp_min(1e-8)).unsqueeze(0),
            alpha=alpha.unsqueeze(0),
        )

    def forward(
        self,
        gaussians: Gaussians3D,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        image_width: int,
        image_height: int,
    ) -> RenderingOutputs:
        gaussian_batch = int(gaussians.mean_vectors.shape[0])
        camera_batch = int(extrinsics.shape[0])
        if int(intrinsics.shape[0]) != camera_batch:
            raise ValueError("Intrinsics and extrinsics must have the same batch size.")
        if gaussian_batch not in (1, camera_batch):
            raise ValueError("Gaussians must have batch size one or match the camera batch size.")
        outputs: list[RenderingOutputs] = []
        for index in range(camera_batch):
            gaussian_index = 0 if gaussian_batch == 1 else index
            single = Gaussians3D(*(value[gaussian_index] for value in gaussians))
            outputs.append(
                self._render_one(
                    single,
                    extrinsic=extrinsics[index],
                    intrinsics=intrinsics[index],
                    width=int(image_width),
                    height=int(image_height),
                )
            )
        return RenderingOutputs(
            color=torch.cat([item.color for item in outputs], dim=0),
            depth=torch.cat([item.depth for item in outputs], dim=0),
            alpha=torch.cat([item.alpha for item in outputs], dim=0),
        )

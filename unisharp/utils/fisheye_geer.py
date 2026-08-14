
from __future__ import annotations

from functools import lru_cache
import importlib
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F

from unisharp.utils.pixel_convention import integer_pixel_center_grid



def _geer_rasterizer_root() -> Path:
    return Path(__file__).resolve().parents[2] / "3dgeer" / "submodules" / "geer-rasterizer"


def _load_fisheye624_class() -> Any:
    repo_root = Path(__file__).resolve().parents[2]
    unik3d_root = repo_root / "UniK3D"
    if unik3d_root.exists() and str(unik3d_root) not in sys.path:
        sys.path.insert(0, str(unik3d_root))
    from unik3d.utils.camera import Fisheye624  # type: ignore

    return Fisheye624


@lru_cache(maxsize=1)
def _load_geer_rasterizer() -> Any:
    root = _geer_rasterizer_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    try:
        return importlib.import_module("diff_gaussian_rasterization")
    except Exception as exc:
        raise ImportError(
            "Failed to import 3DGEER rasterizer. Build it first with: "
            f"cd '{root_str}' && python setup.py build_ext --inplace"
        ) from exc


def build_fisheye624_raymap(
    camera_params: torch.Tensor,
    *,
    image_h: int,
    image_w: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if camera_params.ndim == 1:
        camera_params = camera_params.unsqueeze(0)
    params = camera_params.to(device=device, dtype=torch.float32)
    Fisheye624 = _load_fisheye624_class()
    cam = Fisheye624(params=params)
    uu, vv = integer_pixel_center_grid(int(image_h), int(image_w), device=device, dtype=torch.float32)
    uv = torch.stack([uu, vv], dim=0).unsqueeze(0).expand(int(params.shape[0]), -1, -1, -1)
    rays = cam.unproject(uv).to(dtype=dtype)
    return rays / torch.norm(rays, dim=1, keepdim=True).clamp(min=1e-6)


def build_fisheye624_tangent_arrays(
    camera_params: torch.Tensor,
    *,
    image_h: int,
    image_w: int,
    valid_mask: torch.Tensor | None = None,
    extent_quantile: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dev = camera_params.device
    rays = build_fisheye624_raymap(
        camera_params,
        image_h=image_h,
        image_w=image_w,
        device=dev,
        dtype=torch.float32,
    )
    z = rays[:, 2].clamp(min=1e-4)
    valid = torch.isfinite(z) & (z > 1e-4)
    if torch.is_tensor(valid_mask):
        vm = valid_mask[:, 0] if valid_mask.ndim == 4 else valid_mask
        valid = valid & (vm.to(device=dev) > 0.5)
    tan_x = (rays[:, 0] / z).abs()
    tan_y = (rays[:, 1] / z).abs()
    valid_x = tan_x[valid & torch.isfinite(tan_x)]
    valid_y = tan_y[valid & torch.isfinite(tan_y)]
    q = float(extent_quantile)
    if int(valid_x.numel()) > 0:
        tx_max = float(valid_x.max().item()) if q >= 1.0 else float(torch.quantile(valid_x, q).item())
    else:
        tx_max = 1.0
    if int(valid_y.numel()) > 0:
        ty_max = float(valid_y.max().item()) if q >= 1.0 else float(torch.quantile(valid_y, q).item())
    else:
        ty_max = 1.0
    tan_theta = torch.linspace(-max(tx_max, 1e-3), max(tx_max, 1e-3), steps=int(image_w), device=dev)
    tan_phi = torch.linspace(-max(ty_max, 1e-3), max(ty_max, 1e-3), steps=int(image_h), device=dev)
    return rays, tan_theta.contiguous(), tan_phi.contiguous()


def _depth_from_accumulated_invdepth(
    invdepth_accum: torch.Tensor,
    alpha: torch.Tensor,
    *,
    eps: float = 1e-8,
    alpha_min: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    if invdepth_accum.ndim == 2:
        invdepth_accum = invdepth_accum.unsqueeze(0).unsqueeze(0)
    elif invdepth_accum.ndim == 3:
        invdepth_accum = invdepth_accum.unsqueeze(1)
    alpha_1 = alpha[:, :1].to(device=invdepth_accum.device, dtype=invdepth_accum.dtype)
    valid = torch.isfinite(invdepth_accum) & torch.isfinite(alpha_1) & (alpha_1 > float(alpha_min))
    invdepth = torch.where(valid, invdepth_accum / alpha_1.clamp(min=eps), torch.zeros_like(invdepth_accum))
    depth = torch.where(invdepth > eps, 1.0 / invdepth.clamp(min=eps), torch.zeros_like(invdepth))
    return depth, invdepth


def compute_fisheye624_frustum_mask(
    *,
    depth_distance_m: torch.Tensor,
    tgt_w2c: torch.Tensor,
    src_w2c: torch.Tensor,
    tgt_camera_params: torch.Tensor,
    src_camera_params: torch.Tensor,
    src_valid_mask: torch.Tensor | None = None,
    source_depth_distance_m: torch.Tensor | None = None,
    source_occlusion_tolerance_m: float = 0.0,
    source_occlusion_tolerance_ratio: float = 0.10,
    source_visibility_radius_px: int = 0,
    edge_eps_px: float = 0.501,
) -> torch.Tensor:
    if depth_distance_m.ndim != 4 or int(depth_distance_m.shape[0]) != 1:
        raise ValueError(f"Expected depth shape (1,1,H,W), got {tuple(depth_distance_m.shape)}")
    dev = depth_distance_m.device
    dtype = torch.float32
    _, _, h, w = depth_distance_m.shape
    rays = build_fisheye624_raymap(
        tgt_camera_params.to(device=dev, dtype=dtype),
        image_h=int(h),
        image_w=int(w),
        device=dev,
        dtype=dtype,
    )
    depth = depth_distance_m.to(dtype=dtype)
    valid = torch.isfinite(depth[:, 0]) & (depth[:, 0] > 0.0)
    xyz_tgt = rays * depth
    xyz_tgt_h = torch.cat([xyz_tgt, torch.ones_like(depth)], dim=1)
    xyz_world = torch.einsum("bij,bjhw->bihw", torch.linalg.inv(tgt_w2c.to(dtype=dtype)), xyz_tgt_h)
    xyz_src = torch.einsum("bij,bjhw->bihw", src_w2c.to(dtype=dtype), xyz_world)[:, :3]

    Fisheye624 = _load_fisheye624_class()
    src_cam = Fisheye624(params=src_camera_params.to(device=dev, dtype=dtype))
    uv_src = src_cam.project(xyz_src).to(dtype=dtype)
    proj_mask = getattr(src_cam, "projection_mask", None)
    h_src = int(src_valid_mask.shape[-2]) if torch.is_tensor(src_valid_mask) else int(h)
    w_src = int(src_valid_mask.shape[-1]) if torch.is_tensor(src_valid_mask) else int(w)
    u = uv_src[:, 0]
    v = uv_src[:, 1]
    valid = valid & torch.isfinite(u) & torch.isfinite(v)
    eps = float(edge_eps_px)
    valid = valid & (u >= -eps) & (u <= float(w_src - 1) + eps) & (v >= -eps) & (v <= float(h_src - 1) + eps)
    if torch.is_tensor(proj_mask):
        valid = valid & proj_mask[:, 0].to(device=dev, dtype=torch.bool)
    if torch.is_tensor(src_valid_mask):
        uv_grid = torch.stack(
            [
                (u / max(float(w_src - 1), 1.0)) * 2.0 - 1.0,
                (v / max(float(h_src - 1), 1.0)) * 2.0 - 1.0,
            ],
            dim=-1,
        ).clamp(-1.0, 1.0)
        src_valid_proj = F.grid_sample(
            src_valid_mask.to(device=dev, dtype=dtype),
            uv_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        valid = valid & (src_valid_proj[:, 0] > 0.5)
    if torch.is_tensor(source_depth_distance_m):
        src_depth = source_depth_distance_m.to(device=dev, dtype=dtype)
        if src_depth.ndim == 3:
            src_depth = src_depth.unsqueeze(1)
        if src_depth.ndim != 4 or int(src_depth.shape[0]) != 1 or int(src_depth.shape[1]) != 1:
            raise ValueError(f"Expected source_depth_distance_m shape (1,1,H,W), got {tuple(src_depth.shape)}")
        if tuple(src_depth.shape[-2:]) != (h_src, w_src):
            src_depth = F.interpolate(src_depth, size=(h_src, w_src), mode="nearest")
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
                F.max_pool2d(src_depth_valid.to(dtype=dtype), kernel_size=kernel, stride=1, padding=radius)
                > 0.0
            )
        uv_grid = torch.stack(
            [
                (u / max(float(w_src - 1), 1.0)) * 2.0 - 1.0,
                (v / max(float(h_src - 1), 1.0)) * 2.0 - 1.0,
            ],
            dim=-1,
        )
        sampled_src_dist = F.grid_sample(
            src_depth_for_min,
            uv_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[:, 0]
        sampled_src_valid = (
            F.grid_sample(
                src_depth_valid.to(dtype=dtype),
                uv_grid,
                mode="nearest",
                padding_mode="zeros",
                align_corners=True,
            )[:, 0]
            > 0.5
        )
        projected_src_dist = torch.linalg.vector_norm(xyz_src, dim=1)
        tolerance = float(source_occlusion_tolerance_m) + float(source_occlusion_tolerance_ratio) * sampled_src_dist.abs()
        source_visible = sampled_src_valid & torch.isfinite(sampled_src_dist) & (
            projected_src_dist <= sampled_src_dist + tolerance
        )
        valid = valid & source_visible
    return valid.unsqueeze(1).to(dtype=dtype)


def render_gaussians_fisheye624(
    gaussians_world: Any,
    *,
    extrinsics_w2c: torch.Tensor,
    camera_params: torch.Tensor,
    image_h: int,
    image_w: int,
    valid_mask: torch.Tensor | None = None,
    near_threshold: float = 0.2,
    asso_mode: int = 0,
) -> dict[str, torch.Tensor]:
    dgr = _load_geer_rasterizer()
    dev = extrinsics_w2c.device
    dtype = torch.float32

    means = gaussians_world.mean_vectors[0].to(device=dev, dtype=dtype)
    scales = gaussians_world.singular_values[0].to(device=dev, dtype=dtype)
    rotations = gaussians_world.quaternions[0].to(device=dev, dtype=dtype)
    colors = gaussians_world.colors[0].to(device=dev, dtype=dtype)
    opacities = gaussians_world.opacities[0].to(device=dev, dtype=dtype)
    if opacities.ndim == 1:
        opacities = opacities.unsqueeze(-1)

    viewmatrix = extrinsics_w2c[0].to(device=dev, dtype=dtype).transpose(0, 1).contiguous()
    campos = torch.linalg.inv(viewmatrix)[3, :3].contiguous()
    params = camera_params[0].to(device=dev, dtype=dtype) if camera_params.ndim == 2 else camera_params.to(device=dev, dtype=dtype)
    rays, tan_theta, tan_phi = build_fisheye624_tangent_arrays(
        params.unsqueeze(0),
        image_h=int(image_h),
        image_w=int(image_w),
        valid_mask=valid_mask,
    )
    raymap = rays[0].permute(1, 2, 0).contiguous()
    empty = torch.empty(0, device=dev, dtype=dtype)
    bg = torch.zeros(3, device=dev, dtype=dtype)
    tan_theta = tan_theta.to(device=dev, dtype=dtype).contiguous()
    tan_phi = tan_phi.to(device=dev, dtype=dtype).contiguous()
    raster_settings = dgr.GaussianRasterizationSettings(
        image_height=int(image_h),
        image_width=int(image_w),
        tanfovx=float(torch.max(torch.abs(tan_theta)).item()),
        tanfovy=float(torch.max(torch.abs(tan_phi)).item()),
        bg=bg,
        scale_modifier=1.0,
        viewmatrix=viewmatrix,
        mirror_transformed_tan_theta=tan_theta,
        mirror_transformed_tan_phi=tan_phi,
        tan_theta=tan_theta,
        tan_phi=tan_phi,
        focal_x=float(params[0].item()),
        focal_y=float(params[1].item()),
        principal_x=float(params[2].item()),
        principal_y=float(params[3].item()),
        distortion_coeffs=params[4:8].contiguous(),
        raymap=raymap,
        sh_degree=0,
        campos=campos,
        prefiltered=False,
        debug=False,
        antialiasing=False,
        render_mode=1,
        near_threshold=float(near_threshold),
        asso_mode=int(asso_mode),
    )
    rasterizer = dgr.GaussianRasterizer(raster_settings=raster_settings)
    means2d = torch.zeros_like(means)
    rendered_rgb, _radii, invdepth, _kernel_times, _ranges = rasterizer(
        means3D=means,
        means2D=means2d,
        opacities=opacities,
        colors_precomp=colors,
        scales=scales,
        rotations=rotations,
    )
    alpha_rgb, _radii_a, _invdepth_a, _kernel_times_a, _ranges_a = rasterizer(
        means3D=means,
        means2D=means2d,
        opacities=opacities,
        colors_precomp=torch.ones_like(colors),
        scales=scales,
        rotations=rotations,
    )
    alpha = alpha_rgb[:1].unsqueeze(0).clamp(0.0, 1.0)
    depth_distance, invdepth = _depth_from_accumulated_invdepth(invdepth, alpha)
    z = rays[:, 2].clamp(min=1e-4)
    ray_tan_x = (rays[:, 0] / z).abs()
    ray_tan_y = (rays[:, 1] / z).abs()
    angular_valid = (
        (ray_tan_x <= torch.max(torch.abs(tan_theta)).clamp(min=1e-3))
        & (ray_tan_y <= torch.max(torch.abs(tan_phi)).clamp(min=1e-3))
    ).unsqueeze(1).to(dtype=dtype)
    render_valid = angular_valid
    if torch.is_tensor(valid_mask):
        render_valid = render_valid * valid_mask.to(device=dev, dtype=dtype)
    return {
        "color": rendered_rgb.unsqueeze(0).clamp(0.0, 1.0),
        "alpha": alpha,
        "depth_distance": depth_distance,
        "invdepth": invdepth,
        "valid_mask": render_valid,
    }

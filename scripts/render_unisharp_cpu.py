#!/usr/bin/env python3
"""CPU multi-camera rendering for a Flash3D Gaussian scene.

Unlike ``render_cpu_alpha.py``'s compact interpolation trajectory, this entry
point renders a named physical-camera rig. The default ``cross5`` rig creates
five distinct camera centres (centre, left, right, up and down), and every
camera is aimed at a target in the reconstructed scene.

For calibrated/custom rigs, pass a JSON file with this schema::

    {"cameras": [
      {"name": "front_left", "position_xyz": [-0.5, 0, 0],
       "look_at_xyz": [0, 0, 7.2], "roll_deg": 0.0}
    ]}

``position_xyz`` and ``look_at_xyz`` are in the input/source-camera coordinate
system. Keep baselines conservative for a single-image reconstruction: unseen
geometry cannot be created by Gaussian splatting alone.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import psutil
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO_ROOT))

from unisharp.utils.gaussians import Gaussians3D
from unisharp.utils.portable_renderer import PortableGaussianRenderer

from render_cpu_alpha import (
    crop_and_downsample,
    evaluate_view_dependent_color,
    linear_to_srgb,
    load_gaussians,
    normalise_depth,
    project_gaussians,
    quaternion_to_rotation,
    rasterize,
    save_gray,
    save_rgb,
    unsharp_mask,
)


def _native_renderer_dependencies() -> tuple[object, object, object]:
    """Load the exact CUDA rasterizer used by Flash3D's gauss_util.py.

    Kept lazy so that importing/running the CPU reference renderer never
    requires a CUDA build of diff-gaussian-rasterization.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "--backend native requires a CUDA-enabled PyTorch build and GPU. "
            "Use --backend cpu on this machine."
        )
    try:
        from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
        from models.decoder.gauss_util import getProjectionMatrix
    except ImportError as exc:
        raise RuntimeError(
            "The native Flash3D renderer needs the project's CUDA extension "
            "diff_gaussian_rasterization. Build/install the original Flash3D "
            "CUDA environment, then retry --backend native."
        ) from exc
    return GaussianRasterizationSettings, GaussianRasterizer, getProjectionMatrix


def _gsplat_renderer_dependency() -> object:
    """Load gsplat only for its CUDA backend, keeping CPU usage dependency-free."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "--backend gsplat requires a CUDA-enabled PyTorch build and GPU. "
            "Use --backend torch on this machine."
        )
    try:
        # Current gsplat releases export this at package level.  The fallback
        # keeps the adapter usable with older package layouts.
        from gsplat import rasterization
    except ImportError:
        try:
            from gsplat.rendering import rasterization
        except ImportError as exc:
            raise RuntimeError(
                "--backend gsplat requires the CUDA gsplat package. Install it "
                "in the CUDA environment with: python -m pip install gsplat"
            ) from exc
    return rasterization


def _native_sh_degree(gaussians: dict[str, torch.Tensor]) -> int:
    """Infer the supported real-SH degree from canonical [N, K, 3] terms."""
    rest = gaussians.get("features_rest")
    if rest is None or rest.numel() == 0:
        return 0
    coefficients = rest.shape[1] + 1  # include DC
    degree = int(round(math.sqrt(coefficients) - 1))
    if (degree + 1) ** 2 != coefficients:
        raise ValueError(f"Invalid SH coefficient count: {coefficients}")
    return degree


@torch.inference_mode()
def rasterize_gsplat(
    gaussians: dict[str, torch.Tensor],
    world_to_camera: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    background: torch.Tensor,
    near: float,
    far: float,
    scale_modifier: float,
    eps2d: float,
    radius_clip: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Render a Flash3D Gaussian scene through gsplat's CUDA rasterizer.

    gsplat uses conventional (non-transposed) world-to-camera matrices and
    scalar-first ``wxyz`` quaternions, which is the same quaternion layout
    exported by Flash3D.  In contrast, Flash3D's own CUDA extension receives
    transposed matrices; do not transpose ``world_to_camera`` here.
    """
    rasterization = _gsplat_renderer_dependency()
    device = torch.device("cuda")
    means = gaussians["xyz"].to(device, non_blocking=True).contiguous()
    quats = gaussians["rotation"].to(device, non_blocking=True).contiguous()
    scales = (gaussians["scaling"] * scale_modifier).to(device, non_blocking=True).contiguous()
    opacities = gaussians["opacity"].to(device, non_blocking=True).reshape(-1).contiguous()
    features_dc = gaussians["features_dc"].to(device, non_blocking=True).reshape(-1, 1, 3).contiguous()
    features_rest = gaussians.get("features_rest")
    if features_rest is not None and features_rest.numel() > 0:
        colors = torch.cat((features_dc, features_rest.to(device, non_blocking=True)), dim=1).contiguous()
        sh_degree = _native_sh_degree(gaussians)
    else:
        # Degree-zero SH avoids an activation/convention mismatch with RGB
        # colours while retaining exactly the Flash3D DC coefficients.
        colors = features_dc
        sh_degree = 0
    intrinsic = torch.tensor(
        ((fx, 0.0, cx), (0.0, fy, cy), (0.0, 0.0, 1.0)), dtype=torch.float32, device=device
    )
    render, alpha, metadata = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=world_to_camera.to(device, non_blocking=True).float().unsqueeze(0),
        Ks=intrinsic.unsqueeze(0),
        width=width,
        height=height,
        near_plane=near,
        far_plane=far,
        radius_clip=radius_clip,
        eps2d=eps2d,
        sh_degree=sh_degree,
        packed=False,
        tile_size=16,
        backgrounds=background.to(device, non_blocking=True).float().unsqueeze(0),
        render_mode="RGB+ED",
        rasterize_mode="classic",
    )
    # gsplat returns [C, H, W, RGB + expected-depth] and [C, H, W, 1].
    rgb = render[0, ..., :3].detach().float().cpu()
    depth = render[0, ..., 3].detach().float().cpu()
    alpha = alpha[0, ..., 0].detach().float().cpu()
    radii = metadata.get("radii")
    visible = int((radii[0] > 0).sum().item()) if radii is not None and radii.ndim == 2 else int(means.shape[0])
    return rgb, alpha, depth, visible


@torch.inference_mode()
def rasterize_torch_flash3d(
    gaussians: dict[str, torch.Tensor],
    world_to_camera: torch.Tensor,
    camera_center: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    background: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Run Flash3D's portable Torch rasterizer on CPU for one camera.

    ``models/decoder/npu_differentiable_renderer.py`` is the project's
    reimplementation of the native diff-gaussian-rasterization algorithm with
    normal PyTorch operations.  It works unchanged on CPU, and this adapter
    supplies the calibrated camera matrix needed by a saved Gaussian scene.
    """
    from models.decoder.npu_differentiable_renderer import render_predicted_torch

    # Keep all selected Gaussians and all per-tile candidates: unlike the
    # training-debug profile, offline multiview evaluation must not discard
    # splats merely to cap CPU/NPU training memory.
    max_tile_reach = math.ceil(args.max_radius / args.tile_size)
    renderer_cfg = SimpleNamespace(
        dataset=SimpleNamespace(znear=args.near, zfar=args.far),
        model=SimpleNamespace(
            npu_renderer_min_variance=args.min_variance,
            npu_renderer_sigma_cutoff=args.sigma_cutoff,
            npu_renderer_max_radius=args.max_radius,
            npu_renderer_max_gaussians=0,
            npu_renderer_tile_size=args.tile_size,
            # The original CUDA renderer assigns every overlapping tile.  The
            # portable binner uses a finite square around the centre, so size
            # it from the maximum retained radius instead of silently clipping
            # a large ellipse at neighbouring tiles.
            npu_renderer_tile_span=max(5, 2 * max_tile_reach + 3),
            npu_renderer_max_gaussians_per_tile=0,
        ),
    )
    fov_x = 2.0 * math.atan(width / (2.0 * fx))
    fov_y = 2.0 * math.atan(height / (2.0 * fy))
    # render_predicted_torch expects the transposed matrix convention used by
    # Flash3D model.py before it hands matrices to the CUDA rasterizer.
    viewmatrix = world_to_camera.T.contiguous()
    result = render_predicted_torch(
        renderer_cfg,
        gaussians,
        viewmatrix,
        viewmatrix,  # API-compatible unused full projection on Torch path
        viewmatrix,  # API-compatible unused raw projection on Torch path
        camera_center,
        (fov_x, fov_y),
        (height, width),
        background,
        _native_sh_degree(gaussians),
        args.scale_modifier,
        principal_point=(cx, cy),
    )
    rgb = result["render"].permute(1, 2, 0).detach().float().cpu()
    depth = result["depth"].detach().float().cpu()
    alpha = result["alpha"].detach().float().cpu()
    return rgb, alpha, depth, int(result["visibility_filter"].sum().item())


@torch.inference_mode()
def rasterize_gsplat_cpu_reference(
    gaussians: dict[str, torch.Tensor],
    world_to_camera: torch.Tensor,
    camera_center: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    background: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """CPU reference for gsplat's 3DGS ``classic`` rasterization path.

    This follows the public gsplat reference equations: conventional
    world-to-camera transform, clamped Jacobian for perspective covariance,
    ``eps2d`` diagonal regularization, 3.33-sigma axis-aligned tile bounds,
    pixel-centre Gaussian evaluation, and front-to-back alpha compositing.
    It intentionally trades CUDA throughput for inspectable CPU semantics.
    """
    if args.tile_size != 16:
        raise ValueError("gsplat-compatible CPU reference requires --tile-size 16")
    xyz = gaussians["xyz"].float()
    rotation = world_to_camera[:3, :3].float()
    translation = world_to_camera[:3, 3].float()
    camera_xyz = xyz @ rotation.T + translation
    x, y, z = camera_xyz.unbind(dim=-1)

    # Matches gsplat.cuda._torch_impl._persp_proj: the mean uses the ordinary
    # perspective projection while the Jacobian is bounded near the FOV edge.
    inverse_z = z.clamp_min(args.near).reciprocal()
    tan_fov_x, tan_fov_y = 0.5 * width / fx, 0.5 * height / fy
    x_for_jacobian = z * torch.clamp(
        x * inverse_z,
        min=-(cx / fx + 0.3 * tan_fov_x),
        max=((width - cx) / fx + 0.3 * tan_fov_x),
    )
    y_for_jacobian = z * torch.clamp(
        y * inverse_z,
        min=-(cy / fy + 0.3 * tan_fov_y),
        max=((height - cy) / fy + 0.3 * tan_fov_y),
    )
    means2d = torch.stack((fx * x * inverse_z + cx, fy * y * inverse_z + cy), dim=-1)

    # gsplat builds the 3D covariance directly from the supplied scales.  Do
    # not pre-clamp here: its only opacity cap is applied per pixel below.
    scales = gaussians["scaling"].float() * args.scale_modifier
    basis = quaternion_to_rotation(gaussians["rotation"].float()) * scales[:, None, :]
    covariance_world = basis @ basis.transpose(1, 2)
    covariance_camera = rotation[None] @ covariance_world @ rotation.T[None]
    jacobian = torch.zeros((len(xyz), 2, 3), dtype=torch.float32)
    jacobian[:, 0, 0] = fx * inverse_z
    jacobian[:, 0, 2] = -fx * x_for_jacobian * inverse_z.square()
    jacobian[:, 1, 1] = fy * inverse_z
    jacobian[:, 1, 2] = -fy * y_for_jacobian * inverse_z.square()
    covariance_2d = jacobian @ covariance_camera @ jacobian.transpose(1, 2)
    covariance_2d[:, 0, 0] += args.gsplat_eps2d
    covariance_2d[:, 1, 1] += args.gsplat_eps2d
    a, b, c = covariance_2d[:, 0, 0], covariance_2d[:, 0, 1], covariance_2d[:, 1, 1]
    determinant = (a * c - b * b).clamp_min(1e-10)
    conics = torch.stack((c / determinant, -b / determinant, a / determinant), dim=-1)
    radii = torch.stack((torch.ceil(3.33 * a.clamp_min(0).sqrt()), torch.ceil(3.33 * c.clamp_min(0).sqrt())), dim=-1)
    visible = (
        (z > args.near) & (z < args.far) & torch.isfinite(conics).all(dim=-1)
        & (means2d[:, 0] + radii[:, 0] > 0) & (means2d[:, 0] - radii[:, 0] < width)
        & (means2d[:, 1] + radii[:, 1] > 0) & (means2d[:, 1] - radii[:, 1] < height)
        & (radii.amax(dim=-1) > args.gsplat_radius_clip)
    )
    opacity = gaussians["opacity"].float()
    colours = evaluate_view_dependent_color(gaussians, camera_center).float()
    rgb = torch.empty((height, width, 3), dtype=torch.float32)
    alpha_image = torch.empty((height, width), dtype=torch.float32)
    depth_image = torch.empty((height, width), dtype=torch.float32)
    tile_size = args.tile_size
    for y0 in range(0, height, tile_size):
        y1 = min(y0 + tile_size, height)
        for x0 in range(0, width, tile_size):
            x1 = min(x0 + tile_size, width)
            selected = torch.where(
                visible
                & (means2d[:, 0] + radii[:, 0] > x0)
                & (means2d[:, 0] - radii[:, 0] < x1)
                & (means2d[:, 1] + radii[:, 1] > y0)
                & (means2d[:, 1] - radii[:, 1] < y1)
            )[0]
            if selected.numel() == 0:
                rgb[y0:y1, x0:x1] = background
                alpha_image[y0:y1, x0:x1] = 0
                depth_image[y0:y1, x0:x1] = 0
                continue
            # gsplat's intersection key orders each tile by float32 camera z.
            selected = selected[torch.argsort(z[selected].float(), stable=True)]
            yy, xx = torch.meshgrid(
                torch.arange(y0, y1, dtype=torch.float32) + 0.5,
                torch.arange(x0, x1, dtype=torch.float32) + 0.5,
                indexing="ij",
            )
            dx, dy = xx.reshape(1, -1) - means2d[selected, 0, None], yy.reshape(1, -1) - means2d[selected, 1, None]
            conic = conics[selected]
            sigma = 0.5 * (conic[:, 0, None] * dx.square() + conic[:, 2, None] * dy.square()) + conic[:, 1, None] * dx * dy
            alpha = (opacity[selected, None] * torch.exp(-sigma)).clamp(max=0.999)
            transmittance = torch.cumprod(
                torch.cat((torch.ones_like(alpha[:1]), 1.0 - alpha[:-1]), dim=0), dim=0
            )
            weights = alpha * transmittance
            final_transmittance = (1.0 - alpha).prod(dim=0)
            tile_alpha = 1.0 - final_transmittance
            tile_rgb = weights.T @ colours[selected] + final_transmittance[:, None] * background[None]
            accumulated_depth = (weights * z[selected, None]).sum(dim=0)
            tile_depth = torch.where(tile_alpha > 1e-6, accumulated_depth / tile_alpha, torch.zeros_like(tile_alpha))
            rgb[y0:y1, x0:x1] = tile_rgb.reshape(y1 - y0, x1 - x0, 3)
            alpha_image[y0:y1, x0:x1] = tile_alpha.reshape(y1 - y0, x1 - x0)
            depth_image[y0:y1, x0:x1] = tile_depth.reshape(y1 - y0, x1 - x0)
    return rgb, alpha_image, depth_image, int(visible.sum().item())


@torch.inference_mode()
def rasterize_native_flash3d(
    gaussians: dict[str, torch.Tensor],
    world_to_camera: torch.Tensor,
    camera_center: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    background: torch.Tensor,
    near: float,
    far: float,
    scale_modifier: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Render one physical camera with Flash3D's original CUDA path.

    Matrix construction, principal-point conversion, SH layout, scales and
    quaternions intentionally match ``models/decoder/gauss_util.py``.  This
    is not a CPU approximation: it calls its ``GaussianRasterizer`` directly.
    """
    GaussianRasterizationSettings, GaussianRasterizer, getProjectionMatrix = _native_renderer_dependencies()
    device = torch.device("cuda")
    xyz = gaussians["xyz"].to(device, non_blocking=True).contiguous()
    scaling = gaussians["scaling"].to(device, non_blocking=True).contiguous()
    rotation = gaussians["rotation"].to(device, non_blocking=True).contiguous()
    opacity = gaussians["opacity"].to(device, non_blocking=True).reshape(-1, 1).contiguous()
    features_dc = gaussians["features_dc"].to(device, non_blocking=True).reshape(-1, 1, 3).contiguous()
    features_rest = gaussians.get("features_rest")
    if features_rest is not None:
        features_rest = features_rest.to(device, non_blocking=True).contiguous()
        shs = torch.cat((features_dc, features_rest), dim=1).contiguous()
    else:
        shs = features_dc

    # Same focal-to-FOV and K_to_NDC_pp conventions used by Flash3D model.py.
    fov_x = 2.0 * math.atan(width / (2.0 * fx))
    fov_y = 2.0 * math.atan(height / (2.0 * fy))
    principal_x = 2.0 * cx / width - 1.0
    principal_y = 2.0 * cy / height - 1.0
    viewmatrix = world_to_camera.to(device, non_blocking=True).T.contiguous()
    projmatrix_raw = getProjectionMatrix(near, far, fov_x, fov_y, principal_x, principal_y).to(device).T.contiguous()
    full_proj_transform = (viewmatrix @ projmatrix_raw).contiguous()
    raster_settings = GaussianRasterizationSettings(
        image_height=height,
        image_width=width,
        tanfovx=math.tan(fov_x * 0.5),
        tanfovy=math.tan(fov_y * 0.5),
        bg=background.to(device, non_blocking=True).contiguous(),
        scale_modifier=scale_modifier,
        viewmatrix=viewmatrix,
        projmatrix=full_proj_transform,
        # Flash3D enables renderer_w_pose for RE10K, therefore preserve its
        # raw projection matrix extension argument.
        projmatrix_raw=projmatrix_raw,
        sh_degree=_native_sh_degree(gaussians),
        campos=camera_center.to(device, non_blocking=True).contiguous(),
        prefiltered=False,
        debug=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    # Gradients are irrelevant at inference, unlike render_predicted()'s
    # training-only retained screen-space points.
    means2d = torch.zeros_like(xyz)
    outputs = rasterizer(
        means3D=xyz,
        means2D=means2d,
        shs=shs,
        colors_precomp=None,
        opacities=opacity,
        scales=scaling,
        rotations=rotation,
        cov3D_precomp=None,
    )
    rendered, radii = outputs[:2]
    rgb = rendered.permute(1, 2, 0).detach().float().cpu()
    # The Flash3D fork returns depth and alpha.  Retain portable image output
    # for an older upstream extension which returns only image/radii.
    if len(outputs) >= 4:
        depth = outputs[2].detach().float().squeeze().cpu()
        alpha = outputs[3].detach().float().squeeze().cpu()
    else:
        depth = torch.zeros((height, width), dtype=torch.float32)
        alpha = torch.ones((height, width), dtype=torch.float32)
    return rgb, alpha, depth, int((radii > 0).sum().item())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--gaussians", type=Path, required=True, help="UniSHARP gaussians.pt export (or a compatible 3DGS file)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--trajectory",
        choices=("official", "rig"),
        default="official",
        help=(
            "official reproduces released UniSHARP's ten forward and ten "
            "clockwise orbit poses; rig keeps the legacy named-camera renderer"
        ),
    )
    parser.add_argument(
        "--backend", choices=("torch", "cpu", "flash3d_torch", "legacy", "native", "gsplat"), default="torch",
        help=(
            "torch/default and cpu: gsplat-classic CPU reference; "
            "flash3d_torch: prior Flash3D portable PyTorch renderer; legacy is the earlier lightweight "
            "CPU approximation; native uses Flash3D CUDA "
            "diff-gaussian-rasterization; gsplat uses gsplat CUDA"
        ),
    )
    parser.add_argument("--rig", choices=("cross5", "arc5", "grid9"), default="cross5")
    parser.add_argument("--camera-file", type=Path, default=None, help="Custom rig JSON; overrides --rig")
    parser.add_argument(
        "--camera-orientation", choices=("look_at", "parallel"), default="look_at",
        help=(
            "look_at steers every camera toward the scene target; parallel keeps every optical axis "
            "parallel to the source camera, matching a translated multi-camera rig"
        ),
    )
    parser.add_argument("--baseline", type=float, default=0.5, help="Physical horizontal camera-centre offset in scene units")
    parser.add_argument("--vertical-baseline", type=float, default=None, help="Physical vertical camera-centre offset; defaults to 0.7 * baseline")
    parser.add_argument("--position-scale", type=float, default=1.0, help="Scale every camera centre from --camera-file/preset; use this to preserve the same angular motion across scenes with different depth scales")
    parser.add_argument("--height", type=int, default=0, help="0 uses the UniSHARP inference image height")
    parser.add_argument("--width", type=int, default=0, help="0 uses the UniSHARP inference image width")
    parser.add_argument("--fx", type=float, default=390.0)
    parser.add_argument("--fy", type=float, default=390.0)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    parser.add_argument(
        "--use-source-intrinsics", action=argparse.BooleanOptionalAction, default=True,
        help="Use calibrated intrinsics embedded in a compatible Gaussian export (currently UniSHARP); falls back to CLI intrinsics when unavailable",
    )
    parser.add_argument("--supersample", type=int, default=1)
    parser.add_argument("--crop-margin", type=int, default=0)
    parser.add_argument("--sharpen", type=float, default=0.0)
    parser.add_argument(
        "--linear-to-srgb", action=argparse.BooleanOptionalAction, default=True,
        help="Encode rendered linear RGB to sRGB before PNG output; use for UniSHARP colour exports",
    )
    parser.add_argument("--keep-ratio", type=float, default=1.0)
    parser.add_argument("--crop-padding", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--min-opacity", type=float, default=0.0)
    parser.add_argument("--scale-modifier", type=float, default=1.0)
    parser.add_argument("--sigma-cutoff", type=float, default=2.5)
    parser.add_argument("--min-variance", type=float, default=0.2)
    parser.add_argument("--max-radius", type=float, default=96.0)
    parser.add_argument(
        "--gsplat-eps2d", type=float, default=0.0,
        help="gsplat only: projected-covariance regularizer in pixel-squared units",
    )
    parser.add_argument(
        "--gsplat-radius-clip", type=float, default=0.0,
        help="gsplat-compatible CPU/CUDA backends: skip splats whose projected radius is at or below this value",
    )
    parser.add_argument("--near", type=float, default=0.01)
    parser.add_argument("--far", type=float, default=1000.0)
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--background", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument(
        "--prune-source-depth-outliers", action=argparse.BooleanOptionalAction, default=False,
        help=(
            "Before novel-view rendering, remove Gaussians that project behind a nearer "
            "source-view surface. This reduces floating/occluded-layer speckles."
        ),
    )
    parser.add_argument(
        "--source-prune-grid-scale", type=float, default=2.0,
        help="Depth-pruning grid resolution relative to --width/--height",
    )
    parser.add_argument(
        "--source-prune-relative-tolerance", type=float, default=0.06,
        help="Keep source-view points within this relative depth distance of the front layer",
    )
    parser.add_argument(
        "--source-prune-absolute-tolerance", type=float, default=0.08,
        help="Keep source-view points within this absolute depth distance of the front layer",
    )
    parser.add_argument(
        "--completion-mode", choices=("none", "telea", "diffusion"), default="none",
        help=(
            "Fill target pixels not visible from the source camera: telea is a local CPU texture continuation; "
            "diffusion uses a locally installed Diffusers inpainting pipeline"
        ),
    )
    parser.add_argument(
        "--completion-alpha-threshold", type=float, default=0.90,
        help="Pixels below this target/source alpha are marked unobserved for completion",
    )
    parser.add_argument(
        "--completion-relative-depth-tolerance", type=float, default=0.04,
        help="Mark a target point unobserved if it lies this much behind the source depth",
    )
    parser.add_argument(
        "--completion-absolute-depth-tolerance", type=float, default=0.04,
        help="Absolute depth slack when testing source-camera occlusion",
    )
    parser.add_argument(
        "--completion-mask-dilate", type=int, default=3,
        help="Dilate the depth-derived completion mask by this many pixels",
    )
    parser.add_argument(
        "--completion-radius", type=float, default=5.0,
        help="Telea inpainting radius in pixels",
    )
    parser.add_argument(
        "--completion-model", type=Path, default=None,
        help="Local Diffusers inpainting model directory, required by --completion-mode diffusion",
    )
    parser.add_argument(
        "--completion-prompt", default="photorealistic continuation of the scene, preserve the existing building, pavement and bicycles",
        help="Positive prompt for the optional local diffusion inpainting pipeline",
    )
    parser.add_argument("--completion-steps", type=int, default=25, help="Diffusion inpainting denoising steps")
    parser.add_argument("--completion-guidance", type=float, default=7.5, help="Diffusion inpainting guidance scale")
    parser.add_argument("--completion-seed", type=int, default=0, help="Diffusion inpainting random seed")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--save-gaussians", action=argparse.BooleanOptionalAction, default=True, help="Save the filtered Gaussian point cloud as PT and coloured PLY")
    parser.add_argument("--contact-sheet-columns", type=int, default=5, help="Number of columns in comparison_grid.png")
    parser.add_argument("--contact-sheet-title", type=str, default=None, help="Optional title drawn above comparison_grid.png")
    parser.add_argument("--contact-sheet-title-size", type=int, default=36, help="Font size for the centred contact-sheet title")
    return parser.parse_args()


def preset_cameras(rig: str, baseline: float, vertical_baseline: float) -> list[dict]:
    """Return named physical camera centres in source-camera/world coordinates."""
    if rig == "cross5":
        return [
            {"name": "center", "position_xyz": [0.0, 0.0, 0.0], "source_camera": True},
            {"name": "left", "position_xyz": [-baseline, 0.0, 0.0]},
            {"name": "right", "position_xyz": [baseline, 0.0, 0.0]},
            {"name": "up", "position_xyz": [0.0, -vertical_baseline, 0.0]},
            {"name": "down", "position_xyz": [0.0, vertical_baseline, 0.0]},
        ]
    if rig == "arc5":
        return [
            {"name": f"arc_{index:02d}", "position_xyz": [offset, 0.0, 0.0]}
            for index, offset in enumerate(torch.linspace(-baseline, baseline, 5).tolist())
        ]
    if rig == "grid9":
        cameras = []
        for row, dy in enumerate((vertical_baseline, 0.0, -vertical_baseline)):
            for column, dx in enumerate((-baseline, 0.0, baseline)):
                cameras.append({"name": f"r{row}_c{column}", "position_xyz": [dx, dy, 0.0]})
        return cameras
    raise ValueError(f"Unsupported preset rig: {rig}")


def read_cameras(args: argparse.Namespace) -> list[dict]:
    if args.camera_file is None:
        vertical = args.vertical_baseline if args.vertical_baseline is not None else args.baseline * 0.7
        cameras = preset_cameras(args.rig, args.baseline, vertical)
    else:
        payload = json.loads(args.camera_file.read_text(encoding="utf-8"))
        cameras = payload["cameras"] if isinstance(payload, dict) else payload
    if not isinstance(cameras, list) or not cameras:
        raise ValueError("Camera JSON must contain a non-empty 'cameras' list")
    names = set()
    for index, camera in enumerate(cameras):
        name = str(camera.get("name", f"camera_{index:03d}"))
        if name in names:
            raise ValueError(f"Duplicate camera name: {name}")
        names.add(name)
        if "translation_xyz" in camera:
            raise ValueError(
                f"Camera {name}: translation_xyz is an old world-to-camera extrinsic field. "
                "Use physical position_xyz instead."
            )
        position = camera.get("position_xyz", (0.0, 0.0, 0.0))
        if len(position) != 3:
            raise ValueError(f"Camera {name}: position_xyz must have 3 values")
        look_at = camera.get("look_at_xyz")
        if look_at is not None and len(look_at) != 3:
            raise ValueError(f"Camera {name}: look_at_xyz must have 3 values")
        camera["name"] = name
        camera["position_xyz"] = [float(value) for value in position]
        camera["look_at_xyz"] = None if look_at is None else [float(value) for value in look_at]
        camera["roll_deg"] = float(camera.get("roll_deg", 0.0))
        camera["source_camera"] = bool(camera.get("source_camera", False))
    return cameras


def camera_transform(
    camera: dict, default_target: torch.Tensor, orientation: str = "look_at"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a world-to-camera transform from a physical centre and look-at point.

    The source image's camera coordinate system is used as world space: x is
    right, y is down, and z is forward.  Therefore the original camera is at
    [0, 0, 0], and its centre view has an identity world-to-camera transform.
    """
    camera_center = torch.tensor(camera["position_xyz"], dtype=torch.float32)
    if camera.get("source_camera", False):
        if torch.linalg.vector_norm(camera_center) > 1e-6:
            raise ValueError(f"Camera {camera['name']}: source_camera requires position_xyz=[0, 0, 0]")
        if camera["roll_deg"] != 0.0 or camera["look_at_xyz"] is not None:
            raise ValueError(f"Camera {camera['name']}: source_camera cannot set roll_deg or look_at_xyz")
        return torch.eye(4, dtype=torch.float32), default_target
    target = default_target if camera["look_at_xyz"] is None else torch.tensor(camera["look_at_xyz"], dtype=torch.float32)
    if orientation == "parallel":
        if camera["look_at_xyz"] is not None:
            raise ValueError(f"Camera {camera['name']}: parallel orientation cannot set look_at_xyz")
        rotation = torch.eye(3, dtype=torch.float32)
        roll = math.radians(camera["roll_deg"])
        if roll:
            cosine, sine = math.cos(roll), math.sin(roll)
            rotation = torch.tensor(
                ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)), dtype=torch.float32
            )
        transform = torch.eye(4, dtype=torch.float32)
        transform[:3, :3] = rotation
        transform[:3, 3] = -rotation @ camera_center
        return transform, target
    forward = target - camera_center
    if torch.linalg.vector_norm(forward) < 1e-6:
        raise ValueError(f"Camera {camera['name']}: position_xyz must differ from look_at_xyz")
    forward = torch.nn.functional.normalize(forward, dim=0)
    # In the source image coordinate system, physical up is negative y.
    reference_up = torch.tensor((0.0, -1.0, 0.0), dtype=torch.float32)
    if torch.abs(torch.dot(forward, reference_up)) > 0.98:
        reference_up = torch.tensor((0.0, 0.0, 1.0), dtype=torch.float32)
    right = torch.nn.functional.normalize(torch.linalg.cross(forward, reference_up), dim=0)
    down = torch.linalg.cross(forward, right)
    rotation = torch.stack((right, down, forward))
    roll = math.radians(camera["roll_deg"])
    if roll:
        cosine, sine = math.cos(roll), math.sin(roll)
        in_camera_roll = torch.tensor(((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)), dtype=torch.float32)
        rotation = in_camera_roll @ rotation
    transform = torch.eye(4, dtype=torch.float32)
    transform[:3, :3] = rotation
    transform[:3, 3] = -rotation @ camera_center
    return transform, target


def save_gaussian_point_cloud(gaussians: dict[str, torch.Tensor], output: Path, source: Path) -> dict[str, str]:
    """Save a reusable tensor dump and a viewer-friendly coloured point cloud."""
    tensor_path = output / "gaussians_filtered.pt"
    ply_path = output / "gaussians_pointcloud.ply"
    torch.save({"gaussians": gaussians, "source_gaussians": str(source.resolve())}, tensor_path)
    xyz = gaussians["xyz"].numpy()
    colour = (gaussians["color"].clamp(0, 1) * 255).round().to(torch.uint8).numpy()
    opacity = gaussians["opacity"].numpy()
    scale = gaussians["scaling"].numpy()
    with ply_path.open("w", encoding="ascii", newline="\n") as file:
        file.write("ply\nformat ascii 1.0\n")
        file.write(f"element vertex {len(xyz)}\n")
        for property_name in ("x", "y", "z", "red", "green", "blue", "opacity", "scale_x", "scale_y", "scale_z"):
            property_type = "uchar" if property_name in {"red", "green", "blue"} else "float"
            file.write(f"property {property_type} {property_name}\n")
        file.write("end_header\n")
        for point, rgb, alpha, scaling in zip(xyz, colour, opacity, scale):
            file.write(f"{point[0]:.7g} {point[1]:.7g} {point[2]:.7g} {rgb[0]} {rgb[1]} {rgb[2]} {alpha:.7g} {scaling[0]:.7g} {scaling[1]:.7g} {scaling[2]:.7g}\n")
    return {"tensor": str(tensor_path.resolve()), "ply": str(ply_path.resolve())}


def save_contact_sheet(entries: list[tuple[str, Path]], output: Path, columns: int, title: str | None = None, title_size: int = 36) -> Path:
    """Create a labelled RGB comparison grid without altering individual renders."""
    if columns < 1:
        raise ValueError("--contact-sheet-columns must be >= 1")
    images = [(name, Image.open(path).convert("RGB")) for name, path in entries]
    width, height = images[0][1].size
    label_height, padding = 24, 4
    if title_size < 1:
        raise ValueError("--contact-sheet-title-size must be >= 1")
    try:
        title_font = ImageFont.truetype("arial.ttf", title_size)
    except OSError:
        title_font = ImageFont.load_default()
    title_height = title_size + 18 if title else 0
    rows = math.ceil(len(images) / columns)
    sheet = Image.new("RGB", (columns * (width + padding) + padding, title_height + rows * (height + label_height + padding) + padding), "white")
    draw = ImageDraw.Draw(sheet)
    if title:
        box = draw.textbbox((0, 0), title, font=title_font)
        draw.text(((sheet.width - (box[2] - box[0])) * 0.5, 7), title, fill="black", font=title_font)
    for index, (name, image) in enumerate(images):
        row, column = divmod(index, columns)
        x = padding + column * (width + padding)
        y = title_height + padding + row * (height + label_height + padding)
        draw.text((x + 2, y + 4), name, fill="black")
        sheet.paste(image, (x, y + label_height))
    sheet_path = output / "comparison_grid.png"
    sheet.save(sheet_path)
    return sheet_path


def source_intrinsics_at_output_resolution(
    metadata: dict, height: int, width: int
) -> tuple[float, float, float, float] | None:
    """Read and resize the [fx, fy, cx, cy] camera record embedded by UniSHARP."""
    camera = metadata.get("camera", {}) if isinstance(metadata, dict) else {}
    intrinsics = camera.get("intrinsics") if isinstance(camera, dict) else None
    if intrinsics is None:
        return None
    # Benchmark exports JSON, so a 3x3 matrix is commonly a nested Python
    # list rather than a Tensor.  Canonicalise both forms before extracting
    # fx, fy, cx and cy.
    matrix = torch.as_tensor(intrinsics, dtype=torch.float32).detach().cpu()
    if matrix.numel() >= 9 and tuple(matrix.shape[-2:]) == (3, 3):
        matrix = matrix.reshape(-1, 3, 3)[0]
        values = [matrix[0, 0].item(), matrix[1, 1].item(), matrix[0, 2].item(), matrix[1, 2].item()]
    else:
        values = matrix.reshape(-1).tolist()
    if len(values) < 4:
        return None
    input_h, input_w = metadata.get("input_size_hw", (height, width))
    if not input_h or not input_w:
        return None
    sx, sy = width / float(input_w), height / float(input_h)
    fx, fy, cx, cy = (float(value) for value in values[:4])
    return fx * sx, fy * sy, cx * sx, cy * sy


@torch.inference_mode()
def prune_source_depth_outliers(
    gaussians: dict[str, torch.Tensor],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    near: float,
    grid_scale: float,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> tuple[dict[str, torch.Tensor], dict[str, int | float]]:
    """Cull points hidden behind the nearest source-view surface per grid cell.

    A single-image Gaussian prediction frequently contains several candidate
    depth layers along a source camera ray.  The layer behind the visible
    surface is useful neither for reproducing the source image nor for stable
    small-baseline novel views; when it leaks around an occlusion edge it
    appears as a dark floating speckle.  This conservative filter retains the
    front layer and any Gaussian close enough to it to cover a continuous
    surface, accounting for its 3D scale.
    """
    if grid_scale <= 0:
        raise ValueError("--source-prune-grid-scale must be > 0")
    if relative_tolerance < 0 or absolute_tolerance < 0:
        raise ValueError("source depth-pruning tolerances must be >= 0")

    grid_width = max(1, round(width * grid_scale))
    grid_height = max(1, round(height * grid_scale))
    xyz = gaussians["xyz"]
    z = xyz[:, 2]
    # Resample camera coordinates using pixel-centre convention.
    u = (fx * xyz[:, 0] / z.clamp_min(near) + cx + 0.5) * grid_scale - 0.5
    v = (fy * xyz[:, 1] / z.clamp_min(near) + cy + 0.5) * grid_scale - 0.5
    u_int, v_int = torch.floor(u).long(), torch.floor(v).long()
    inside = (
        (z > near)
        & torch.isfinite(u)
        & torch.isfinite(v)
        & (u_int >= 0)
        & (u_int < grid_width)
        & (v_int >= 0)
        & (v_int < grid_height)
    )
    cells = v_int[inside] * grid_width + u_int[inside]
    front_depth = torch.full((grid_width * grid_height,), float("inf"), dtype=torch.float32)
    front_depth.scatter_reduce_(0, cells, z[inside].float(), reduce="amin", include_self=True)
    reference_depth = torch.full_like(z, float("inf"))
    reference_depth[inside] = front_depth[cells]
    scale_slack = 2.0 * gaussians["scaling"].abs().amax(dim=-1)
    tolerated_depth = reference_depth * (1.0 + relative_tolerance) + absolute_tolerance + scale_slack
    occluded = inside & (z > tolerated_depth)
    keep = ~occluded
    filtered = {key: value[keep].contiguous() for key, value in gaussians.items()}
    return filtered, {
        "grid_width": grid_width,
        "grid_height": grid_height,
        "input_gaussians": int(len(z)),
        "projected_gaussians": int(inside.sum().item()),
        "removed_occluded_gaussians": int(occluded.sum().item()),
        "kept_gaussians": int(keep.sum().item()),
        "relative_tolerance": relative_tolerance,
        "absolute_tolerance": absolute_tolerance,
    }


@torch.inference_mode()
def rasterize_with_backend(
    gaussians: dict[str, torch.Tensor],
    transform: torch.Tensor,
    camera_center: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    background: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Run the selected backend with one conventional world-to-camera pose."""
    if args.backend == "native":
        return rasterize_native_flash3d(
            gaussians, transform, camera_center, fx, fy, cx, cy,
            width, height, background, args.near, args.far, args.scale_modifier,
        )
    if args.backend == "gsplat":
        return rasterize_gsplat(
            gaussians, transform, fx, fy, cx, cy, width, height, background,
            args.near, args.far, args.scale_modifier, args.gsplat_eps2d, args.gsplat_radius_clip,
        )
    if args.backend in {"torch", "cpu"}:
        return rasterize_gsplat_cpu_reference(
            gaussians, transform, camera_center, fx, fy, cx, cy,
            width, height, background, args,
        )
    if args.backend == "flash3d_torch":
        return rasterize_torch_flash3d(
            gaussians, transform, camera_center, fx, fy, cx, cy,
            width, height, background, args,
        )
    projected = project_gaussians(
        gaussians, transform, fx, fy, cx, cy, args.near, args.far,
        args.min_variance, args.sigma_cutoff, args.max_radius,
        args.scale_modifier, width, height, camera_center,
    )
    rgb, alpha, depth = rasterize(
        projected, height, width, background,
        args.tile_size, args.chunk_size, args.sigma_cutoff,
    )
    return rgb, alpha, depth, int(projected["z"].numel())


@torch.inference_mode()
def source_visibility_completion_mask(
    target_depth: torch.Tensor,
    target_alpha: torch.Tensor,
    source_depth: torch.Tensor,
    source_alpha: torch.Tensor,
    target_world_to_camera: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    args: argparse.Namespace,
) -> torch.Tensor:
    """Identify target pixels absent or occluded in the source camera.

    A low target alpha catches actual holes.  More importantly, target depth is
    unprojected and reprojected into the source view: a point outside the source
    frustum or behind its frontmost rendered depth is a disocclusion even when
    a broad Gaussian happened to give it high alpha.
    """
    height, width = target_depth.shape
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32) + 0.5,
        torch.arange(width, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    target_valid = (target_alpha >= args.completion_alpha_threshold) & (target_depth > args.near)
    depth = target_depth.clamp_min(args.near)
    target_camera = torch.stack(
        ((x - cx) * depth / fx, (y - cy) * depth / fy, depth), dim=-1
    )
    rotation = target_world_to_camera[:3, :3].float()
    translation = target_world_to_camera[:3, 3].float()
    # camera = world @ R.T + t, hence world = (camera - t) @ R.
    world = (target_camera - translation) @ rotation
    source_z = world[..., 2]
    source_x = fx * world[..., 0] / source_z.clamp_min(args.near) + cx
    source_y = fy * world[..., 1] / source_z.clamp_min(args.near) + cy
    in_source_frame = (
        (source_z > args.near)
        & (source_x >= 0.0) & (source_x < width)
        & (source_y >= 0.0) & (source_y < height)
    )
    # align_corners=False represents pixel centres at 0.5, 1.5, ... .
    source_grid = torch.stack((2.0 * source_x / width - 1.0, 2.0 * source_y / height - 1.0), dim=-1)[None]
    sampled_source_depth = F.grid_sample(
        source_depth[None, None], source_grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )[0, 0]
    sampled_source_alpha = F.grid_sample(
        source_alpha[None, None], source_grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )[0, 0]
    source_surface_valid = sampled_source_alpha >= args.completion_alpha_threshold
    source_occluded = source_z > (
        sampled_source_depth * (1.0 + args.completion_relative_depth_tolerance)
        + args.completion_absolute_depth_tolerance
    )
    unseen = (~target_valid) | (~in_source_frame) | (~source_surface_valid) | source_occluded
    if args.completion_mask_dilate:
        radius = args.completion_mask_dilate
        unseen = F.max_pool2d(
            unseen[None, None].float(), kernel_size=2 * radius + 1, stride=1, padding=radius
        )[0, 0] > 0.5
    return unseen


def complete_unseen_rgb(
    rgb: torch.Tensor, mask: torch.Tensor, args: argparse.Namespace
) -> torch.Tensor:
    """Fill a depth-derived disocclusion mask without altering observed pixels."""
    if not bool(mask.any()):
        return rgb
    image = (rgb.clamp(0, 1) * 255).round().to(torch.uint8).numpy()
    mask_u8 = (mask.to(torch.uint8).numpy() * 255)
    if args.completion_mode == "telea":
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("--completion-mode telea requires opencv-python (cv2)") from exc
        completed = cv2.inpaint(image, mask_u8, args.completion_radius, cv2.INPAINT_TELEA)
        return torch.from_numpy(completed).float() / 255.0
    if args.completion_mode == "diffusion":
        if args.completion_model is None:
            raise ValueError("--completion-mode diffusion requires --completion-model pointing to a local model directory")
        try:
            from diffusers import AutoPipelineForInpainting
        except ImportError as exc:
            raise RuntimeError(
                "Diffusers is not installed. Install diffusers, transformers and accelerate in a CUDA environment, "
                "then pass a local inpainting model with --completion-model."
            ) from exc
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline = AutoPipelineForInpainting.from_pretrained(
            str(args.completion_model), local_files_only=True
        ).to(device)
        generator = torch.Generator(device=device).manual_seed(args.completion_seed)
        result = pipeline(
            prompt=args.completion_prompt,
            image=Image.fromarray(image, mode="RGB"),
            mask_image=Image.fromarray(mask_u8, mode="L"),
            num_inference_steps=args.completion_steps,
            guidance_scale=args.completion_guidance,
            generator=generator,
        ).images[0]
        return torch.from_numpy(__import__("numpy").asarray(result.convert("RGB")).copy()).float() / 255.0
    raise ValueError(f"Unsupported completion mode: {args.completion_mode}")


def _save_animation(frames: list[torch.Tensor], path: Path, duration_ms: int = 300) -> None:
    if not frames:
        raise ValueError(f"No frames available for {path}")
    images = [Image.fromarray(frame.detach().cpu().numpy()) for frame in frames]
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=int(duration_ms),
        loop=0,
        disposal=2,
    )


def _official_render_resolution(args: argparse.Namespace, metadata: dict) -> tuple[int, int]:
    """Resolve the released script's output size from the inference export."""
    height, width = int(args.height), int(args.width)
    if (height == 0) != (width == 0):
        raise ValueError("Set both --height and --width, or leave both at 0.")
    if height == 0:
        source_hw = metadata.get("input_size_hw", (0, 0))
        if not isinstance(source_hw, (tuple, list)) or len(source_hw) != 2:
            raise ValueError("The Gaussian export has no UniSHARP input resolution; specify --height and --width.")
        height, width = int(source_hw[0]), int(source_hw[1])
    if height < 1 or width < 1:
        raise ValueError("The render resolution must be positive.")
    return height, width


def _official_motion(gaussians: dict[str, torch.Tensor]) -> tuple[float, float, dict[str, float | None]]:
    """Port of released ``_adaptive_view_motion_distances`` for a saved scene."""
    values = gaussians["xyz"][:, 2].reshape(-1).float()
    values = values[torch.isfinite(values) & (values > 1e-3) & (values < 1e4)]
    forward_default, rotate_default = 0.2, 0.1
    if not int(values.numel()):
        return forward_default, rotate_default, {"scene_depth_m": None, "median_depth_m": None, "foreground_depth_m": None, "motion_scale": 1.0}
    median = float(torch.median(values).item())
    foreground = float(torch.quantile(values, 0.20).item())
    effective = median if median >= 2.5 else min(median, foreground)
    if not math.isfinite(effective) or effective >= 2.0:
        return forward_default, rotate_default, {"scene_depth_m": effective, "median_depth_m": median, "foreground_depth_m": foreground, "motion_scale": 1.0}
    scale = max(0.08, effective / 2.0)
    forward = min(forward_default * scale, effective * 0.04)
    rotate = min(rotate_default * scale, effective * 0.02)
    return forward, rotate, {"scene_depth_m": effective, "median_depth_m": median, "foreground_depth_m": foreground, "motion_scale": scale}


def _official_w2c(eye_xyz: tuple[float, float, float]) -> torch.Tensor:
    """Released inference keeps the source orientation and translates its centre."""
    extrinsic = torch.eye(4, dtype=torch.float32)
    extrinsic[:3, 3] = -torch.tensor(eye_xyz, dtype=torch.float32)
    return extrinsic


def _official_frame(
    renderer: PortableGaussianRenderer,
    gaussians: Gaussians3D,
    extrinsic: torch.Tensor,
    intrinsics: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    output = renderer(
        gaussians,
        extrinsics=extrinsic.unsqueeze(0),
        intrinsics=intrinsics.unsqueeze(0),
        image_width=int(width),
        image_height=int(height),
    )
    alpha = output.alpha[0].clamp(0.0, 1.0)
    # This is the same un-premultiply + linearRGB-to-sRGB sequence as the
    # released GSplatRenderer inference branch (black background).
    rgb = linear_to_srgb((output.color[0] / alpha.clamp_min(1e-4)).clamp(0.0, 1.0))
    frame = (rgb.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).permute(1, 2, 0)
    crop_y, crop_x = int(round(height * 0.05)), int(round(width * 0.05))
    if crop_y > 0 and crop_x > 0 and crop_y * 2 < height and crop_x * 2 < width:
        frame = frame[crop_y : height - crop_y, crop_x : width - crop_x]
    return frame.contiguous()


def render_official_unisharp(args: argparse.Namespace) -> None:
    """CPU rendering counterpart of released ``scripts/infer_unisharp.py``.

    The camera paths and output convention are identical to the official
    perspective branch.  The only substitution is the CPU gsplat-classic
    reference rasterizer for CUDA gsplat.
    """
    if args.backend not in {"torch", "cpu"}:
        raise ValueError("--trajectory official is CPU-only; use --backend torch (the default) or cpu.")
    if args.camera_file is not None or args.rig != "cross5":
        raise ValueError("--trajectory official does not use --rig/--camera-file. Use --trajectory rig for physical cameras.")
    if args.supersample != 1 or args.crop_margin != 0 or args.sharpen != 0.0:
        raise ValueError("Official trajectory requires --supersample 1 --crop-margin 0 --sharpen 0.")
    if args.completion_mode != "none" or args.prune_source_depth_outliers:
        raise ValueError("Completion and depth pruning are legacy rig options; they are not part of official UniSHARP inference.")
    if tuple(float(value) for value in args.background) != (0.0, 0.0, 0.0):
        raise ValueError("Official UniSHARP rendering uses a black background.")
    if float(args.scale_modifier) != 1.0 or float(args.gsplat_eps2d) != 0.0:
        raise ValueError("Official UniSHARP rendering requires --scale-modifier 1 and --gsplat-eps2d 0.")

    torch.set_num_threads(int(args.threads))
    gaussians, metadata = load_gaussians(
        args.gaussians,
        float(args.keep_ratio),
        float(args.min_opacity),
        bool(args.crop_padding),
    )
    height, width = _official_render_resolution(args, metadata)
    source_intrinsics = source_intrinsics_at_output_resolution(metadata, height, width)
    if source_intrinsics is None:
        raise ValueError("Official trajectory requires a UniSHARP gaussians.pt export with embedded pinhole intrinsics.")
    camera = metadata.get("camera", {}) if isinstance(metadata, dict) else {}
    if str(camera.get("model", "pinhole")).lower() not in {"pinhole", "perspective"}:
        raise ValueError("The CPU official renderer currently supports the released pinhole branch only.")
    fx, fy, cx, cy = source_intrinsics
    intrinsics = torch.tensor(((fx, 0.0, cx), (0.0, fy, cy), (0.0, 0.0, 1.0)), dtype=torch.float32)
    scene = Gaussians3D(
        gaussians["xyz"].unsqueeze(0),
        gaussians["scaling"].unsqueeze(0),
        gaussians["rotation"].unsqueeze(0),
        gaussians["color"].unsqueeze(0),
        gaussians["opacity"].unsqueeze(0),
    )
    renderer = PortableGaussianRenderer(background_color="black", low_pass_filter_eps=0.0)
    forward_distance, rotate_radius, motion = _official_motion(gaussians)
    forward_poses = [
        _official_w2c((0.0, 0.0, forward_distance * float(index + 1) / 10.0))
        for index in range(10)
    ]
    rotate_poses = [
        _official_w2c((rotate_radius * math.sin(-2.0 * math.pi * index / 10.0), rotate_radius * math.cos(-2.0 * math.pi * index / 10.0), 0.0))
        for index in range(10)
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    forward_dir, rotate_dir = args.output / "forward", args.output / "rotate"
    forward_dir.mkdir(parents=True, exist_ok=True)
    rotate_dir.mkdir(parents=True, exist_ok=True)
    forward_frames, rotate_frames = [], []
    for index, pose in enumerate(forward_poses):
        frame = _official_frame(renderer, scene, pose, intrinsics, height, width)
        Image.fromarray(frame.numpy()).save(forward_dir / f"forward_{index:02d}.png")
        forward_frames.append(frame)
    for index, pose in enumerate(rotate_poses):
        frame = _official_frame(renderer, scene, pose, intrinsics, height, width)
        Image.fromarray(frame.numpy()).save(rotate_dir / f"rotate_{index:02d}.png")
        rotate_frames.append(frame)
    _save_animation(forward_frames, args.output / "forward.gif")
    _save_animation(rotate_frames, args.output / "rotate.gif")
    report = {
        "renderer": "unisharp_official_gsplat_classic_cpu_reference",
        "trajectory": "released_infer_unisharp_perspective",
        "gaussians_input": str(args.gaussians.resolve()),
        "num_gaussians": int(gaussians["xyz"].shape[0]),
        "height": height,
        "width": width,
        "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        "forward_distance_m": forward_distance,
        "rotate_radius_m": rotate_radius,
        **motion,
        "forward_frames": 10,
        "rotate_frames": 10,
    }
    (args.output / "metadata.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved official UniSHARP forward/rotate renders to {args.output.resolve()}")


def render(args: argparse.Namespace) -> None:
    """Render a prepared argument namespace (also used by UniSHARP in-process)."""
    if args.trajectory == "official":
        render_official_unisharp(args)
        return
    if int(args.height) < 1 or int(args.width) < 1:
        raise ValueError("--trajectory rig requires explicit positive --height and --width.")
    if not 0 < args.keep_ratio <= 1:
        raise ValueError("--keep-ratio must be in (0, 1]")
    if args.supersample < 1:
        raise ValueError("--supersample must be >= 1")
    if args.gsplat_eps2d < 0:
        raise ValueError("--gsplat-eps2d must be >= 0")
    if args.gsplat_radius_clip < 0:
        raise ValueError("--gsplat-radius-clip must be >= 0")
    if not 0.0 <= args.completion_alpha_threshold <= 1.0:
        raise ValueError("--completion-alpha-threshold must be in [0, 1]")
    if args.completion_relative_depth_tolerance < 0 or args.completion_absolute_depth_tolerance < 0:
        raise ValueError("completion depth tolerances must be >= 0")
    if args.completion_mask_dilate < 0:
        raise ValueError("--completion-mask-dilate must be >= 0")
    if args.completion_radius <= 0:
        raise ValueError("--completion-radius must be > 0")
    if args.completion_mode != "none" and (args.supersample != 1 or args.crop_margin != 0):
        raise ValueError(
            "completion currently requires --supersample 1 and --crop-margin 0 so source and target depth use the same camera grid"
        )
    if args.backend == "native":
        # Fail before creating partial output directories or loading a large
        # point cloud when this machine is intentionally CPU-only.
        _native_renderer_dependencies()
    elif args.backend == "gsplat":
        _gsplat_renderer_dependency()
    torch.set_num_threads(args.threads)
    # PyTorch allows this process-wide setting only before parallel work has
    # started. The CLI sets it here; an embedding host may already have run a
    # model, in which case the existing setting is retained.
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    gaussians, metadata = load_gaussians(args.gaussians, args.keep_ratio, args.min_opacity, args.crop_padding)
    source_intrinsics = source_intrinsics_at_output_resolution(metadata, args.height, args.width)
    if args.use_source_intrinsics and source_intrinsics is not None:
        output_fx, output_fy, output_cx, output_cy = source_intrinsics
        intrinsics_source = "embedded_source_camera"
    else:
        output_fx, output_fy = args.fx, args.fy
        output_cx = (args.width - 1) * 0.5 if args.cx is None else args.cx
        output_cy = (args.height - 1) * 0.5 if args.cy is None else args.cy
        intrinsics_source = "cli" if source_intrinsics is None else "cli_override"
    depth_pruning = None
    if args.prune_source_depth_outliers:
        gaussians, depth_pruning = prune_source_depth_outliers(
            gaussians,
            output_fx,
            output_fy,
            output_cx,
            output_cy,
            args.width,
            args.height,
            args.near,
            args.source_prune_grid_scale,
            args.source_prune_relative_tolerance,
            args.source_prune_absolute_tolerance,
        )
        print(
            "source-depth pruning: "
            f"removed {depth_pruning['removed_occluded_gaussians']:,} / "
            f"{depth_pruning['input_gaussians']:,} Gaussians"
        )
    cameras = read_cameras(args)
    if args.position_scale <= 0:
        raise ValueError("--position-scale must be > 0")
    if args.position_scale != 1.0:
        for camera in cameras:
            camera["position_xyz"] = [value * args.position_scale for value in camera["position_xyz"]]
    # The opacity-filtered median gives a stable, visible scene point to aim at.
    # It preserves the identity pose for the original camera in typical scenes.
    default_target = torch.quantile(gaussians["xyz"], 0.5, dim=0)
    args.output.mkdir(parents=True, exist_ok=True)
    rgb_dir, alpha_dir, depth_dir = (args.output / "rgb", args.output / "alpha", args.output / "depth")
    raw_rgb_dir = args.output / "rgb_raw" if args.completion_mode != "none" else None
    completion_mask_dir = args.output / "completion_mask" if args.completion_mode != "none" else None
    for directory in (rgb_dir, alpha_dir, depth_dir, raw_rgb_dir, completion_mask_dir):
        if directory is None:
            continue
        directory.mkdir(parents=True, exist_ok=True)

    render_height, render_width = args.height * args.supersample, args.width * args.supersample
    fx, fy = output_fx * args.supersample, output_fy * args.supersample
    cx = (output_cx + 0.5) * args.supersample - 0.5
    cy = (output_cy + 0.5) * args.supersample - 0.5
    background = torch.tensor(args.background, dtype=torch.float32)
    report_cameras = []
    rgb_entries = []
    source_depth = source_alpha = None
    if args.completion_mode != "none":
        # Render the source camera once as an occlusion reference.  The test
        # below is geometric, so high-alpha Gaussian extrapolation alone is
        # not mistaken for source-image evidence.
        source_rgb, source_alpha, source_depth, _ = rasterize_with_backend(
            gaussians, torch.eye(4, dtype=torch.float32), torch.zeros(3),
            fx, fy, cx, cy, render_width, render_height, background, args,
        )
        source_alpha = crop_and_downsample(
            source_alpha, args.height, args.width, args.supersample, args.crop_margin
        ).clamp(0, 1)
        source_depth = crop_and_downsample(
            source_depth, args.height, args.width, args.supersample, args.crop_margin
        )

    for index, camera in enumerate(cameras):
        if args.backend in {"native", "gsplat"}:
            torch.cuda.synchronize()
        start = time.perf_counter()
        transform, target = camera_transform(camera, default_target, args.camera_orientation)
        camera_center = torch.tensor(camera["position_xyz"], dtype=torch.float32)
        rgb, alpha, depth, visible_gaussians = rasterize_with_backend(
            gaussians, transform, camera_center, fx, fy, cx, cy,
            render_width, render_height, background, args,
        )
        if args.backend in {"native", "gsplat"}:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        rgb = unsharp_mask(crop_and_downsample(rgb, args.height, args.width, args.supersample, args.crop_margin), args.sharpen)
        if args.linear_to_srgb:
            rgb = linear_to_srgb(rgb)
        alpha = crop_and_downsample(alpha, args.height, args.width, args.supersample, args.crop_margin).clamp(0, 1)
        depth = crop_and_downsample(depth, args.height, args.width, args.supersample, args.crop_margin)
        stem = f"{index:02d}_{camera['name']}"
        completion_mask = None
        completed_rgb = rgb
        completion_seconds = 0.0
        if args.completion_mode != "none" and not camera.get("source_camera", False):
            completion_mask = source_visibility_completion_mask(
                depth, alpha, source_depth, source_alpha, transform,
                output_fx, output_fy, output_cx, output_cy, args,
            )
            completion_start = time.perf_counter()
            completed_rgb = complete_unseen_rgb(rgb, completion_mask, args)
            completion_seconds = time.perf_counter() - completion_start
            save_rgb(rgb, raw_rgb_dir / f"{stem}.png")
            save_gray(completion_mask.float(), completion_mask_dir / f"{stem}.png")
        save_rgb(completed_rgb, rgb_dir / f"{stem}.png")
        save_gray(alpha, alpha_dir / f"{stem}.png")
        save_gray(normalise_depth(depth, alpha), depth_dir / f"{stem}.png")
        rgb_path = rgb_dir / f"{stem}.png"
        rgb_entries.append((camera["name"], rgb_path))
        report_cameras.append({
            **camera,
            "look_at_xyz": target.tolist(),
            "world_to_camera": transform.tolist(),
            "visible_gaussians": visible_gaussians,
            "mean_alpha": alpha.mean().item(),
            "seconds": elapsed,
            "completion_seconds": completion_seconds,
            "completion_pixels": 0 if completion_mask is None else int(completion_mask.sum().item()),
            "completion_fraction": 0.0 if completion_mask is None else completion_mask.float().mean().item(),
            "rgb": str(rgb_path.resolve()),
            "rgb_raw": None if completion_mask is None else str((raw_rgb_dir / f"{stem}.png").resolve()),
            "completion_mask": None if completion_mask is None else str((completion_mask_dir / f"{stem}.png").resolve()),
        })
        print(
            f"{stem}: {visible_gaussians} visible Gaussians, {elapsed:.3f} s, mean alpha={alpha.mean().item():.4f}, "
            f"completion={0 if completion_mask is None else int(completion_mask.sum().item()):,} px"
        )

    comparison_grid = save_contact_sheet(rgb_entries, args.output, args.contact_sheet_columns, args.contact_sheet_title, args.contact_sheet_title_size)
    point_cloud = save_gaussian_point_cloud(gaussians, args.output, args.gaussians) if args.save_gaussians else None
    report = {
        "device": "cuda" if args.backend in {"native", "gsplat"} else "cpu",
        "renderer": (
            "Flash3D native diff-gaussian-rasterization CUDA backend"
            if args.backend == "native"
            else (
                "gsplat CUDA 3D Gaussian rasterizer"
                if args.backend == "gsplat"
                else (
                    "gsplat classic-semantics CPU reference renderer"
                    if args.backend in {"torch", "cpu"}
                    else (
                        "Flash3D portable PyTorch tile renderer (CPU)"
                        if args.backend == "flash3d_torch"
                        else "Legacy PyTorch anisotropic 3D Gaussian + front-to-back alpha blending"
                    )
                )
            )
        ),
        "backend": "torch" if args.backend == "cpu" else args.backend,
        "gaussians_input": str(args.gaussians.resolve()),
        "gaussians_after_filter": gaussians["xyz"].shape[0],
        "image_size_hw": [args.height, args.width],
        "intrinsics_at_output_resolution": {"fx": output_fx, "fy": output_fy, "cx": output_cx, "cy": output_cy, "source": intrinsics_source},
        "preset_rig": None if args.camera_file else args.rig,
        "camera_orientation": args.camera_orientation,
        "camera_file": None if args.camera_file is None else str(args.camera_file.resolve()),
        "position_scale": args.position_scale,
        "source_depth_pruning": depth_pruning,
        "completion": {
            "mode": args.completion_mode,
            "alpha_threshold": args.completion_alpha_threshold,
            "relative_depth_tolerance": args.completion_relative_depth_tolerance,
            "absolute_depth_tolerance": args.completion_absolute_depth_tolerance,
            "mask_dilate": args.completion_mask_dilate,
            "raw_rgb_directory": None if raw_rgb_dir is None else str(raw_rgb_dir.resolve()),
            "mask_directory": None if completion_mask_dir is None else str(completion_mask_dir.resolve()),
        },
        "linear_to_srgb": args.linear_to_srgb,
        "default_look_at_xyz": default_target.tolist(),
        "comparison_grid": str(comparison_grid.resolve()),
        "contact_sheet_title": args.contact_sheet_title,
        "contact_sheet_title_size": args.contact_sheet_title_size,
        "point_cloud": point_cloud,
        "cameras": report_cameras,
        "mean_seconds_per_view": sum(camera["seconds"] for camera in report_cameras) / len(report_cameras),
        "rss_final_mb": psutil.Process().memory_info().rss / 1024**2,
    }
    (args.output / "multiview_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output / "camera_rig.json").write_text(json.dumps({"cameras": cameras}, indent=2), encoding="utf-8")
    print(f"Saved {len(cameras)} camera views, rig manifest, and report to {args.output.resolve()}")


def main() -> None:
    render(parse_args())


if __name__ == "__main__":
    main()

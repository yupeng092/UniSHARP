#!/usr/bin/env python3
from __future__ import annotations

"""Render UniSHARP ``gaussians.pt`` with a pure PyTorch CPU rasterizer.

The implementation projects anisotropic 3D Gaussians to screen-space ellipses,
sorts them from near to far, and performs front-to-back alpha compositing.  It
does not import gsplat, Triton, CUDA, or the 3DGEER rasterizer.

The target renderer is pinhole.  The source image may have been perspective,
fisheye, or panoramic: UniSHARP's predicted Gaussians are already expressed in
the source-camera 3D coordinate frame, so any of them can be viewed through the
generated pinhole target cameras.
"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gaussians", type=Path, required=True, help="UniSHARP gaussians.pt")
    parser.add_argument(
        "--poses",
        type=Path,
        default=None,
        help="Optional [V,4,4] target poses. Generated small-baseline W2C poses are used when omitted.",
    )
    parser.add_argument(
        "--pose-convention",
        choices=["w2c", "c2w"],
        default="w2c",
        help="Convention of matrices loaded with --poses.",
    )
    parser.add_argument("--views", type=int, default=5)
    parser.add_argument(
        "--translation",
        type=float,
        default=0.03,
        help="Maximum generated camera translation in UniSHARP world units (normally metres).",
    )
    parser.add_argument("--yaw", type=float, default=3.0, help="Maximum generated yaw in degrees.")
    parser.add_argument("--pose-index", type=int, nargs="*", default=None)
    parser.add_argument("--output", type=Path, required=True)

    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Output height. Perspective inputs default to their inference height; others default to 256.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Output width. Perspective inputs default to their inference width; others default to 384.",
    )
    parser.add_argument("--supersample", type=int, default=1)
    parser.add_argument("--crop-margin", type=int, default=0)
    parser.add_argument("--sharpen", type=float, default=0.0)

    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    parser.add_argument(
        "--fov-deg",
        type=float,
        default=70.0,
        help="Horizontal pinhole FoV used when source pinhole intrinsics are unavailable.",
    )
    parser.add_argument("--focal-scale", type=float, default=1.0)

    parser.add_argument("--keep-ratio", type=float, default=1.0)
    parser.add_argument(
        "--max-gaussians",
        type=int,
        default=0,
        help="Keep the most opaque N Gaussians after filtering; 0 keeps all.",
    )
    parser.add_argument("--min-opacity", type=float, default=0.005)
    parser.add_argument("--scale-modifier", type=float, default=1.0)
    parser.add_argument("--sigma-cutoff", type=float, default=3.0)
    parser.add_argument("--min-variance", type=float, default=0.30)
    parser.add_argument("--max-radius", type=float, default=96.0)
    parser.add_argument("--near", type=float, default=0.01)
    parser.add_argument("--far", type=float, default=1000.0)
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument(
        "--transmittance-eps",
        type=float,
        default=1e-4,
        help="Stop a fully opaque tile early when every pixel is below this transmittance.",
    )
    parser.add_argument(
        "--background",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("R", "G", "B"),
        help="sRGB background in [0,1].",
    )
    parser.add_argument("--threads", type=int, default=0, help="0 keeps PyTorch's default.")
    parser.add_argument(
        "--gif",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save render.gif in addition to individual frames.",
    )
    parser.add_argument("--gif-duration-ms", type=int, default=250)
    return parser.parse_args()


def _load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _flatten_gaussian_tensor(
    value: torch.Tensor,
    name: str,
    last_dim: int | None,
) -> torch.Tensor:
    value = value.detach().to(device="cpu", dtype=torch.float32)
    if value.ndim >= 2 and int(value.shape[0]) == 1:
        value = value[0]
    if last_dim is None:
        value = value.reshape(-1)
    else:
        if value.ndim < 2 or int(value.shape[-1]) != int(last_dim):
            raise ValueError(f"{name} must end in {last_dim}, got {tuple(value.shape)}")
        value = value.reshape(-1, int(last_dim))
    return value.contiguous()


def load_gaussians(
    path: Path,
    keep_ratio: float,
    max_gaussians: int,
    min_opacity: float,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, int]]:
    if not path.is_file():
        raise FileNotFoundError(f"Gaussian file not found: {path}")
    payload = _load_torch(path)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a dict in {path}, got {type(payload)!r}")
    raw = payload.get("gaussians", payload)
    if not isinstance(raw, dict):
        raise TypeError("The 'gaussians' entry must be a tensor dictionary.")

    required = {
        "mean_vectors": 3,
        "singular_values": 3,
        "quaternions": 4,
        "colors": 3,
        "opacities": None,
    }
    missing = required.keys() - raw.keys()
    if missing:
        raise KeyError(f"Missing UniSHARP Gaussian fields: {sorted(missing)}")
    gaussians = {
        name: _flatten_gaussian_tensor(raw[name], name, last_dim)
        for name, last_dim in required.items()
    }
    count = int(gaussians["mean_vectors"].shape[0])
    for name, value in gaussians.items():
        if int(value.shape[0]) != count:
            raise ValueError(f"Gaussian count mismatch for {name}: {value.shape[0]} vs {count}")

    valid = (
        torch.isfinite(gaussians["mean_vectors"]).all(dim=-1)
        & torch.isfinite(gaussians["singular_values"]).all(dim=-1)
        & (gaussians["singular_values"] > 0).all(dim=-1)
        & torch.isfinite(gaussians["quaternions"]).all(dim=-1)
        & (torch.linalg.vector_norm(gaussians["quaternions"], dim=-1) > 1e-8)
        & torch.isfinite(gaussians["colors"]).all(dim=-1)
        & torch.isfinite(gaussians["opacities"])
        & (gaussians["opacities"] >= float(min_opacity))
    )
    valid_indices = torch.where(valid)[0]
    valid_count = int(valid_indices.numel())
    if valid_count == 0:
        raise ValueError("No valid Gaussians remain after finite-value and opacity filtering.")

    requested_count = valid_count
    if float(keep_ratio) < 1.0:
        requested_count = min(requested_count, max(1, math.ceil(valid_count * float(keep_ratio))))
    if int(max_gaussians) > 0:
        requested_count = min(requested_count, int(max_gaussians))
    if requested_count < valid_count:
        opacity = gaussians["opacities"][valid_indices]
        chosen = torch.topk(opacity, requested_count, sorted=False).indices
        valid_indices = valid_indices[chosen]

    filtered = {name: value[valid_indices].contiguous() for name, value in gaussians.items()}
    filtered["colors"].clamp_(0.0, 1.0)
    filtered["opacities"].clamp_(0.0, 1.0)
    counts = {"input": count, "valid": valid_count, "kept": int(valid_indices.numel())}
    return filtered, payload, counts


def quaternion_to_rotation(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert UniSHARP scalar-first quaternions (w, x, y, z) to matrices."""
    quaternion = F.normalize(quaternion.float(), dim=-1)
    w, x, y, z = quaternion.unbind(dim=-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(-1, 3, 3)


def make_small_baseline_trajectory(
    views: int,
    translation: float,
    yaw_deg: float,
) -> torch.Tensor:
    """Create source-frame to target-camera W2C transforms."""
    if int(views) < 1:
        raise ValueError("--views must be >= 1")
    if int(views) == 1:
        samples = torch.zeros(1, dtype=torch.float32)
    else:
        samples = torch.linspace(-1.0, 1.0, int(views), dtype=torch.float32)
    poses = []
    for sample in samples:
        angle = float(sample) * float(yaw_deg) * math.pi / 180.0
        cosine, sine = math.cos(angle), math.sin(angle)
        world_to_camera = torch.eye(4, dtype=torch.float32)
        world_to_camera[:3, :3] = torch.tensor(
            [
                [cosine, 0.0, sine],
                [0.0, 1.0, 0.0],
                [-sine, 0.0, cosine],
            ],
            dtype=torch.float32,
        )
        world_to_camera[0, 3] = float(sample) * float(translation)
        poses.append(world_to_camera)
    return torch.stack(poses)


def load_or_create_poses(args: argparse.Namespace) -> tuple[torch.Tensor, str]:
    if args.poses is None:
        poses = make_small_baseline_trajectory(args.views, args.translation, args.yaw)
        return poses, "generated_small_baseline_w2c"
    loaded = _load_torch(args.poses)
    if isinstance(loaded, dict):
        for key in ("poses", "world_to_camera", "w2c", "camera_to_world", "c2w"):
            if torch.is_tensor(loaded.get(key)):
                loaded = loaded[key]
                break
    if not torch.is_tensor(loaded):
        raise TypeError("--poses must contain a tensor or a dict with a pose tensor.")
    poses = loaded.detach().float().cpu()
    if poses.ndim == 2:
        poses = poses.unsqueeze(0)
    if poses.ndim != 3 or tuple(poses.shape[-2:]) != (4, 4):
        raise ValueError(f"Expected poses [V,4,4], got {tuple(poses.shape)}")
    if args.pose_convention == "c2w":
        poses = torch.linalg.inv(poses)
    return poses.contiguous(), str(args.poses.resolve())


def _source_image_info(payload: dict[str, Any]) -> tuple[int | None, int | None, str | None]:
    image = payload.get("image", {})
    camera = payload.get("camera", {})
    height = int(image["height"]) if isinstance(image, dict) and image.get("height") else None
    width = int(image["width"]) if isinstance(image, dict) and image.get("width") else None
    kind = str(camera.get("kind")) if isinstance(camera, dict) and camera.get("kind") else None
    return height, width, kind


def resolve_output_size(
    payload: dict[str, Any],
    requested_height: int | None,
    requested_width: int | None,
) -> tuple[int, int]:
    source_height, source_width, source_kind = _source_image_info(payload)
    default_height = source_height if source_kind == "perspective" and source_height else 256
    default_width = source_width if source_kind == "perspective" and source_width else 384
    height = int(requested_height) if requested_height is not None else int(default_height)
    width = int(requested_width) if requested_width is not None else int(default_width)
    if height < 1 or width < 1:
        raise ValueError("Output height and width must be positive.")
    return height, width


def resolve_intrinsics(
    payload: dict[str, Any],
    args: argparse.Namespace,
    output_height: int,
    output_width: int,
) -> tuple[float, float, float, float, str]:
    source_height, source_width, _ = _source_image_info(payload)
    camera = payload.get("camera", {})
    intrinsics = camera.get("intrinsics") if isinstance(camera, dict) else None
    have_source_intrinsics = torch.is_tensor(intrinsics)

    if have_source_intrinsics:
        matrix = intrinsics.detach().float().cpu()
        if matrix.ndim == 3:
            matrix = matrix[0]
        if tuple(matrix.shape) != (3, 3):
            raise ValueError(f"Stored camera intrinsics must be [1,3,3] or [3,3], got {tuple(matrix.shape)}")
        source_height = source_height or output_height
        source_width = source_width or output_width
        sx = float(output_width) / float(source_width)
        sy = float(output_height) / float(source_height)
        base_fx = float(matrix[0, 0]) * sx
        base_fy = float(matrix[1, 1]) * sy
        base_cx = (float(matrix[0, 2]) + 0.5) * sx - 0.5
        base_cy = (float(matrix[1, 2]) + 0.5) * sy - 0.5
        source = "stored_source_pinhole_intrinsics"
    else:
        fov_radians = float(args.fov_deg) * math.pi / 180.0
        if not 0.0 < fov_radians < math.pi:
            raise ValueError("--fov-deg must be in (0, 180).")
        base_fx = 0.5 * float(output_width) / math.tan(0.5 * fov_radians)
        base_fy = base_fx
        base_cx = (float(output_width) - 1.0) * 0.5
        base_cy = (float(output_height) - 1.0) * 0.5
        source = "fov_fallback"

    if args.fx is not None:
        base_fx = float(args.fx)
        if args.fy is None:
            base_fy = base_fx
        source = "command_line_override"
    if args.fy is not None:
        base_fy = float(args.fy)
        source = "command_line_override"
    if args.cx is not None:
        base_cx = float(args.cx)
        source = "command_line_override"
    if args.cy is not None:
        base_cy = float(args.cy)
        source = "command_line_override"
    base_fx *= float(args.focal_scale)
    base_fy *= float(args.focal_scale)
    if base_fx <= 0.0 or base_fy <= 0.0:
        raise ValueError("Resolved focal lengths must be positive.")
    return base_fx, base_fy, base_cx, base_cy, source


def project_gaussians(
    gaussians: dict[str, torch.Tensor],
    world_to_camera: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    near: float,
    far: float,
    min_variance: float,
    sigma_cutoff: float,
    max_radius: float,
    scale_modifier: float,
    width: int,
    height: int,
) -> dict[str, torch.Tensor]:
    xyz = gaussians["mean_vectors"]
    camera_rotation = world_to_camera[:3, :3].float()
    camera_translation = world_to_camera[:3, 3].float()
    camera_xyz = xyz @ camera_rotation.T + camera_translation
    x, y, z = camera_xyz.unbind(dim=-1)

    rotation = quaternion_to_rotation(gaussians["quaternions"])
    scales = gaussians["singular_values"].clamp_min(1e-7) * float(scale_modifier)
    basis = rotation * scales[:, None, :]
    covariance_world = basis @ basis.transpose(1, 2)
    covariance_camera = camera_rotation[None] @ covariance_world @ camera_rotation.T[None]

    inverse_z = z.clamp_min(float(near)).reciprocal()
    jacobian = torch.zeros((xyz.shape[0], 2, 3), dtype=torch.float32)
    jacobian[:, 0, 0] = float(fx) * inverse_z
    jacobian[:, 0, 2] = -float(fx) * x * inverse_z.square()
    jacobian[:, 1, 1] = float(fy) * inverse_z
    jacobian[:, 1, 2] = -float(fy) * y * inverse_z.square()
    covariance_2d = jacobian @ covariance_camera @ jacobian.transpose(1, 2)
    covariance_2d[:, 0, 0] += float(min_variance)
    covariance_2d[:, 1, 1] += float(min_variance)

    a = covariance_2d[:, 0, 0]
    b = covariance_2d[:, 0, 1]
    c = covariance_2d[:, 1, 1]
    determinant = (a * c - b.square()).clamp_min(1e-10)
    inverse = torch.stack((c / determinant, -b / determinant, a / determinant), dim=-1)
    largest_eigenvalue = 0.5 * (
        a + c + torch.sqrt(((a - c).square() + 4.0 * b.square()).clamp_min(0.0))
    )
    radius = (
        float(sigma_cutoff) * torch.sqrt(largest_eigenvalue.clamp_min(0.0))
    ).clamp(max=float(max_radius))
    u = float(fx) * x * inverse_z + float(cx)
    v = float(fy) * y * inverse_z + float(cy)

    visible = (
        (z > float(near))
        & (z < float(far))
        & torch.isfinite(u)
        & torch.isfinite(v)
        & torch.isfinite(radius)
        & torch.isfinite(inverse).all(dim=-1)
        & (u + radius >= 0.0)
        & (u - radius < float(width))
        & (v + radius >= 0.0)
        & (v - radius < float(height))
    )
    order = torch.argsort(z[visible])
    return {
        "u": u[visible][order].contiguous(),
        "v": v[visible][order].contiguous(),
        "z": z[visible][order].contiguous(),
        "radius": radius[visible][order].contiguous(),
        "inverse": inverse[visible][order].contiguous(),
        "opacity": gaussians["opacities"][visible][order].contiguous(),
        "color": gaussians["colors"][visible][order].contiguous(),
    }


@torch.inference_mode()
def rasterize(
    projected: dict[str, torch.Tensor],
    height: int,
    width: int,
    background_linear: torch.Tensor,
    tile_size: int,
    chunk_size: int,
    sigma_cutoff: float,
    transmittance_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rgb = torch.empty((height, width, 3), dtype=torch.float32)
    alpha_image = torch.empty((height, width), dtype=torch.float32)
    depth_image = torch.empty((height, width), dtype=torch.float32)
    cutoff_squared = float(sigma_cutoff) ** 2
    u, v, radius = projected["u"], projected["v"], projected["radius"]

    for y0 in range(0, int(height), int(tile_size)):
        y1 = min(y0 + int(tile_size), int(height))
        for x0 in range(0, int(width), int(tile_size)):
            x1 = min(x0 + int(tile_size), int(width))
            overlaps = (
                (u + radius >= float(x0))
                & (u - radius < float(x1))
                & (v + radius >= float(y0))
                & (v - radius < float(y1))
            )
            indices = torch.where(overlaps)[0]
            yy, xx = torch.meshgrid(
                torch.arange(y0, y1, dtype=torch.float32),
                torch.arange(x0, x1, dtype=torch.float32),
                indexing="ij",
            )
            pixels_x, pixels_y = xx.reshape(-1), yy.reshape(-1)
            transmittance = torch.ones(pixels_x.numel(), dtype=torch.float32)
            tile_rgb = torch.zeros((pixels_x.numel(), 3), dtype=torch.float32)
            tile_depth = torch.zeros(pixels_x.numel(), dtype=torch.float32)

            for start in range(0, int(indices.numel()), int(chunk_size)):
                selected = indices[start : start + int(chunk_size)]
                dx = pixels_x[None] - projected["u"][selected, None]
                dy = pixels_y[None] - projected["v"][selected, None]
                inverse = projected["inverse"][selected]
                mahalanobis = (
                    inverse[:, 0, None] * dx.square()
                    + 2.0 * inverse[:, 1, None] * dx * dy
                    + inverse[:, 2, None] * dy.square()
                )
                alpha = projected["opacity"][selected, None] * torch.exp(-0.5 * mahalanobis)
                alpha.masked_fill_(mahalanobis > cutoff_squared, 0.0)
                alpha.clamp_(0.0, 0.999)

                one_minus_alpha = 1.0 - alpha
                exclusive = torch.cumprod(
                    torch.cat((torch.ones_like(one_minus_alpha[:1]), one_minus_alpha[:-1]), dim=0),
                    dim=0,
                )
                weights = alpha * exclusive * transmittance[None]
                tile_rgb += weights.T @ projected["color"][selected]
                tile_depth += weights.T @ projected["z"][selected]
                transmittance *= one_minus_alpha.prod(dim=0)
                if transmittance.numel() and bool(torch.all(transmittance <= float(transmittance_eps))):
                    break

            tile_alpha = 1.0 - transmittance
            tile_rgb += transmittance[:, None] * background_linear[None]
            tile_depth = torch.where(
                tile_alpha > 1e-6,
                tile_depth / tile_alpha.clamp_min(1e-6),
                torch.zeros_like(tile_depth),
            )
            rgb[y0:y1, x0:x1] = tile_rgb.reshape(y1 - y0, x1 - x0, 3)
            alpha_image[y0:y1, x0:x1] = tile_alpha.reshape(y1 - y0, x1 - x0)
            depth_image[y0:y1, x0:x1] = tile_depth.reshape(y1 - y0, x1 - x0)
    return rgb.clamp(0.0, 1.0), alpha_image.clamp(0.0, 1.0), depth_image


def srgb_to_linear(value: torch.Tensor) -> torch.Tensor:
    return torch.where(
        value <= 0.04045,
        value / 12.92,
        ((value + 0.055) / 1.055).pow(2.4),
    )


def linear_to_srgb(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(0.0, 1.0)
    return torch.where(
        value <= 0.0031308,
        12.92 * value,
        1.055 * value.pow(1.0 / 2.4) - 0.055,
    ).clamp(0.0, 1.0)


def crop_and_resize(
    image: torch.Tensor,
    output_height: int,
    output_width: int,
    supersample: int,
    crop_margin: int,
    mode: str,
) -> torch.Tensor:
    margin = int(crop_margin) * int(supersample)
    if margin:
        if 2 * margin >= min(int(image.shape[0]), int(image.shape[1])):
            raise ValueError("--crop-margin is too large for the output size.")
        image = image[margin:-margin, margin:-margin]
    if tuple(image.shape[:2]) == (int(output_height), int(output_width)):
        return image
    channels_last = image.ndim == 3
    tensor = image.permute(2, 0, 1)[None] if channels_last else image[None, None]
    kwargs: dict[str, Any] = {"size": (int(output_height), int(output_width)), "mode": mode}
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
        kwargs["antialias"] = True
    resized = F.interpolate(tensor, **kwargs)
    return resized[0].permute(1, 2, 0) if channels_last else resized[0, 0]


def unsharp_mask(image: torch.Tensor, amount: float) -> torch.Tensor:
    if float(amount) <= 0.0:
        return image
    tensor = image.permute(2, 0, 1)[None]
    kernel_1d = torch.tensor([1, 4, 6, 4, 1], dtype=tensor.dtype) / 16.0
    kernel_x = kernel_1d.reshape(1, 1, 1, 5).repeat(3, 1, 1, 1)
    kernel_y = kernel_1d.reshape(1, 1, 5, 1).repeat(3, 1, 1, 1)
    padded = F.pad(tensor, (2, 2, 2, 2), mode="reflect")
    blurred = F.conv2d(padded, kernel_x, groups=3)
    blurred = F.conv2d(blurred, kernel_y, groups=3)
    return (tensor + float(amount) * (tensor - blurred))[0].permute(1, 2, 0).clamp(0.0, 1.0)


def normalise_depth(depth: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    valid = (alpha > 0.01) & torch.isfinite(depth) & (depth > 0.0)
    output = torch.zeros_like(depth)
    if bool(valid.any()):
        low, high = torch.quantile(depth[valid], torch.tensor([0.02, 0.98]))
        output[valid] = 1.0 - (
            (depth[valid] - low) / (high - low).clamp_min(1e-6)
        ).clamp(0.0, 1.0)
    return output


def tensor_to_rgb_u8(image: torch.Tensor) -> np.ndarray:
    return (image * 255.0 + 0.5).clamp(0, 255).to(torch.uint8).cpu().numpy()


def save_rgb(image: torch.Tensor, path: Path) -> np.ndarray:
    array = tensor_to_rgb_u8(image)
    Image.fromarray(array, mode="RGB").save(path)
    return array


def save_gray(image: torch.Tensor, path: Path) -> None:
    array = (image * 255.0 + 0.5).clamp(0, 255).to(torch.uint8).cpu().numpy()
    Image.fromarray(array, mode="L").save(path)


def _rss_mb() -> float | None:
    try:
        import psutil

        return float(psutil.Process().memory_info().rss / 1024**2)
    except Exception:
        return None


def _validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < float(args.keep_ratio) <= 1.0:
        raise ValueError("--keep-ratio must be in (0,1].")
    if int(args.max_gaussians) < 0:
        raise ValueError("--max-gaussians must be >= 0.")
    if int(args.supersample) < 1:
        raise ValueError("--supersample must be >= 1.")
    if int(args.tile_size) < 1 or int(args.chunk_size) < 1:
        raise ValueError("--tile-size and --chunk-size must be >= 1.")
    if float(args.near) <= 0.0 or float(args.far) <= float(args.near):
        raise ValueError("Require 0 < --near < --far.")
    if any(not 0.0 <= float(channel) <= 1.0 for channel in args.background):
        raise ValueError("--background values must be in [0,1].")


def main() -> None:
    args = parse_args()
    _validate_args(args)
    if int(args.threads) > 0:
        torch.set_num_threads(int(args.threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    gaussians, payload, counts = load_gaussians(
        args.gaussians,
        keep_ratio=float(args.keep_ratio),
        max_gaussians=int(args.max_gaussians),
        min_opacity=float(args.min_opacity),
    )
    poses, poses_source = load_or_create_poses(args)
    pose_indices = args.pose_index if args.pose_index is not None else list(range(int(poses.shape[0])))
    for index in pose_indices:
        if int(index) < 0 or int(index) >= int(poses.shape[0]):
            raise IndexError(f"Pose index {index} is outside [0,{int(poses.shape[0]) - 1}].")

    output_height, output_width = resolve_output_size(payload, args.height, args.width)
    base_fx, base_fy, base_cx, base_cy, intrinsics_source = resolve_intrinsics(
        payload,
        args,
        output_height=output_height,
        output_width=output_width,
    )
    render_height = int(output_height) * int(args.supersample)
    render_width = int(output_width) * int(args.supersample)
    fx = base_fx * int(args.supersample)
    fy = base_fy * int(args.supersample)
    cx = (base_cx + 0.5) * int(args.supersample) - 0.5
    cy = (base_cy + 0.5) * int(args.supersample) - 0.5
    background_srgb = torch.tensor(args.background, dtype=torch.float32)
    background_linear = srgb_to_linear(background_srgb)

    args.output.mkdir(parents=True, exist_ok=True)
    rgb_dir = args.output / "rgb"
    alpha_dir = args.output / "alpha"
    depth_dir = args.output / "depth"
    depth_raw_dir = args.output / "depth_raw"
    for directory in (rgb_dir, alpha_dir, depth_dir, depth_raw_dir):
        directory.mkdir(parents=True, exist_ok=True)
    torch.save(poses, args.output / "render_poses_w2c.pt")
    (args.output / "render_poses_w2c.json").write_text(
        json.dumps(poses.tolist(), indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"Loaded {counts['input']} Gaussians; {counts['valid']} valid; {counts['kept']} kept. "
        f"Rendering {len(pose_indices)} view(s) at {output_width}x{output_height} on CPU."
    )
    timings: list[float] = []
    visible_counts: list[int] = []
    gif_frames: list[Image.Image] = []
    for pose_index in pose_indices:
        started = time.perf_counter()
        projected = project_gaussians(
            gaussians=gaussians,
            world_to_camera=poses[int(pose_index)],
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            near=float(args.near),
            far=float(args.far),
            min_variance=float(args.min_variance),
            sigma_cutoff=float(args.sigma_cutoff),
            max_radius=float(args.max_radius) * int(args.supersample),
            scale_modifier=float(args.scale_modifier),
            width=render_width,
            height=render_height,
        )
        rgb_linear, alpha, depth = rasterize(
            projected=projected,
            height=render_height,
            width=render_width,
            background_linear=background_linear,
            tile_size=int(args.tile_size),
            chunk_size=int(args.chunk_size),
            sigma_cutoff=float(args.sigma_cutoff),
            transmittance_eps=float(args.transmittance_eps),
        )
        elapsed = time.perf_counter() - started
        timings.append(float(elapsed))
        visible_count = int(projected["z"].numel())
        visible_counts.append(visible_count)

        rgb_linear = crop_and_resize(
            rgb_linear,
            output_height,
            output_width,
            args.supersample,
            args.crop_margin,
            mode="bicubic",
        ).clamp(0.0, 1.0)
        alpha = crop_and_resize(
            alpha,
            output_height,
            output_width,
            args.supersample,
            args.crop_margin,
            mode="bilinear",
        ).clamp(0.0, 1.0)
        depth = crop_and_resize(
            depth,
            output_height,
            output_width,
            args.supersample,
            args.crop_margin,
            mode="bilinear",
        ).clamp_min(0.0)
        rgb_srgb = unsharp_mask(linear_to_srgb(rgb_linear), float(args.sharpen))

        stem = f"view_{int(pose_index):03d}"
        rgb_array = save_rgb(rgb_srgb, rgb_dir / f"{stem}.png")
        save_gray(alpha, alpha_dir / f"{stem}.png")
        save_gray(normalise_depth(depth, alpha), depth_dir / f"{stem}.png")
        np.save(depth_raw_dir / f"{stem}.npy", depth.cpu().numpy().astype(np.float32))
        if args.gif:
            gif_frames.append(Image.fromarray(rgb_array, mode="RGB"))
        print(
            f"{stem}: {visible_count} visible Gaussians, {elapsed:.3f} s, "
            f"mean alpha={alpha.mean().item():.4f}"
        )

    if args.gif and gif_frames:
        gif_frames[0].save(
            args.output / "render.gif",
            save_all=True,
            append_images=gif_frames[1:],
            duration=int(args.gif_duration_ms),
            loop=0,
            disposal=2,
        )

    report = {
        "device": "cpu",
        "renderer": "PyTorch anisotropic 3D Gaussian front-to-back alpha compositor",
        "target_camera": "pinhole",
        "gaussians_input": str(args.gaussians.resolve()),
        "poses_input": poses_source,
        "pose_convention_used": "world_to_camera",
        "gaussian_counts": counts,
        "visible_gaussians": visible_counts,
        "image_size_hw": [output_height, output_width],
        "render_size_hw": [render_height, render_width],
        "intrinsics_source": intrinsics_source,
        "intrinsics_at_output_resolution": {
            "fx": base_fx,
            "fy": base_fy,
            "cx": base_cx,
            "cy": base_cy,
        },
        "seconds_per_view": timings,
        "mean_seconds_per_view": sum(timings) / max(len(timings), 1),
        "rss_final_mb": _rss_mb(),
        "settings": {
            "keep_ratio": args.keep_ratio,
            "max_gaussians": args.max_gaussians,
            "min_opacity": args.min_opacity,
            "scale_modifier": args.scale_modifier,
            "sigma_cutoff": args.sigma_cutoff,
            "min_variance": args.min_variance,
            "max_radius": args.max_radius,
            "near": args.near,
            "far": args.far,
            "tile_size": args.tile_size,
            "chunk_size": args.chunk_size,
            "transmittance_eps": args.transmittance_eps,
            "background_srgb": list(args.background),
            "supersample": args.supersample,
            "crop_margin": args.crop_margin,
            "sharpen": args.sharpen,
            "threads": torch.get_num_threads(),
        },
    }
    (args.output / "render_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    saved_kinds = "RGB, alpha, depth, poses"
    if args.gif:
        saved_kinds += ", GIF"
    print(f"Saved {saved_kinds}, and report to {args.output.resolve()}")


if __name__ == "__main__":
    main()

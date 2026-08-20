from __future__ import annotations

"""CPU-only UniSHARP Gaussian prediction and native-style pinhole rendering.

This script intentionally does not import gsplat, Triton, or the 3DGEER CUDA
rasterizer.  Its CPU multiview branch mirrors the perspective branch of
``scripts/infer_unisharp.py``: source-to-world conversion, adaptive forward
and orbit trajectories, black-background alpha compositing, sRGB conversion,
and five-percent output cropping.  The rasterizer itself is the project's
PyTorch ``PortableGaussianRenderer`` reference implementation.
"""

import argparse
import dataclasses
import gc
import json
import logging
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from PIL import Image, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from unisharp.models.unisharp_feature import (  # noqa: E402
    UnisharpFeatureConfig,
    UnisharpFeatureModel,
)
from unisharp.utils.camera_projection import build_extrinsics_w2c  # noqa: E402
from unisharp.utils.camera_utils import transform_gaussians_to_world  # noqa: E402
from unisharp.utils.color_space import linearRGB2sRGB  # noqa: E402
from unisharp.utils.gaussians import Gaussians3D, save_ply  # noqa: E402
from unisharp.utils.portable_renderer import PortableGaussianRenderer  # noqa: E402
from unisharp.utils.rayfit_camera import (  # noqa: E402
    fit_fisheye624_params_from_rays,
    fit_pinhole_intrinsics_from_rays,
)


LOGGER = logging.getLogger("infer_unisharp_cpu")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".PNG", ".JPG", ".JPEG", ".WEBP"}
CameraKind = Literal["perspective", "fisheye", "panorama"]

FISHEYE_FOV_THRESHOLD_DEG = 120.0
FISHEYE_DIAG_THRESHOLD_DEG = 150.0
FISHEYE_VFOV_MIN_DEG = 80.0
FISHEYE_MAX_ASPECT = 1.65
PANORAMA_HFOV_THRESHOLD_DEG = 300.0
PANORAMA_VFOV_THRESHOLD_DEG = 120.0
PANORAMA_ASPECT_MIN = 1.9
PANORAMA_ASPECT_MAX = 2.1
FORWARD_VIEWS = 10
FORWARD_DISTANCE_M = 0.2
ROTATE_VIEWS = 10
ROTATE_RADIUS_M = 0.1
GIF_DURATION_MS = 300
VIEW_MOTION_NEAR_SCENE_DEPTH_M = 2.0
VIEW_MOTION_MIN_SCALE = 0.08
VIEW_MOTION_FORWARD_DEPTH_FRAC = 0.04
VIEW_MOTION_ROTATE_DEPTH_FRAC = 0.02
VIEW_MOTION_FAR_SCENE_MEDIAN_M = 2.5
VIEW_MOTION_FOREGROUND_DEPTH_QUANTILE = 0.20


def _configure_caches() -> None:
    cache_root = REPO_ROOT / "checkpoints"
    torchhub_dir = cache_root / "torchhub"
    hf_dir = cache_root / "huggingface"
    torchhub_dir.mkdir(parents=True, exist_ok=True)
    hf_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(torchhub_dir))
    os.environ.setdefault("HF_HOME", str(hf_dir))
    torch.hub.set_dir(str(torchhub_dir))


def _feature_config_from_checkpoint(
    checkpoint_path: Path,
    ckpt: dict[str, Any],
) -> UnisharpFeatureConfig:
    cfg = UnisharpFeatureConfig()
    merged: dict[str, Any] = {}
    cfg_payload = ckpt.get("config", {})
    if isinstance(cfg_payload, dict):
        merged.update(cfg_payload)
    for key in cfg.__dict__:
        if key in ckpt:
            merged[key] = ckpt[key]

    config_path = checkpoint_path.parent / "config.json"
    if config_path.exists():
        try:
            sidecar = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            LOGGER.warning("Ignoring invalid sidecar config %s: %s", config_path, exc)
        else:
            if isinstance(sidecar, dict):
                merged.update({key: value for key, value in sidecar.items() if key in cfg.__dict__})

    for key in cfg.__dict__:
        if key in merged:
            setattr(cfg, key, merged[key])
    return cfg


def _load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[UnisharpFeatureModel, UnisharpFeatureConfig, int]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError(f"Expected a checkpoint dict, got {type(ckpt)!r}")

    cfg = _feature_config_from_checkpoint(checkpoint_path, ckpt)
    LOGGER.info("Building UniSHARP (UniK3D backbone=%s) on CPU", cfg.unik3d_backbone)
    model = UnisharpFeatureModel(cfg)
    missing, unexpected = model.load_from_checkpoint(str(checkpoint_path), strict=False)
    if missing or unexpected:
        LOGGER.warning("Checkpoint keys: missing=%s unexpected=%s", missing[:20], unexpected[:20])
    model.to(device).eval()
    return model, cfg, int(ckpt.get("step", 0))


def _collect_image_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    if args.image is not None:
        paths.append(Path(args.image))
    if args.image_list is not None:
        list_path = Path(args.image_list)
        for raw in list_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            candidate = Path(line)
            if not candidate.is_absolute():
                candidate = list_path.parent / candidate
            paths.append(candidate)
    if args.image_dir is not None:
        root = Path(args.image_dir)
        paths.extend(sorted(path for path in root.iterdir() if path.is_file() and path.suffix in IMAGE_SUFFIXES))
    if not paths:
        raise ValueError("Provide --image, --image-list, or --image-dir.")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
    return paths[: int(args.max_images)] if int(args.max_images) > 0 else paths


def _load_rgb_u8(
    image_path: Path,
    max_long_edge: int,
) -> tuple[torch.Tensor, tuple[int, int]]:
    with Image.open(image_path) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
    native_w, native_h = image.size
    if int(max_long_edge) > 0:
        scale = min(1.0, float(max_long_edge) / float(max(native_h, native_w)))
        if scale < 1.0:
            image = image.resize(
                (
                    max(1, int(round(native_w * scale))),
                    max(1, int(round(native_h * scale))),
                ),
                resample=Image.Resampling.BILINEAR,
            )
    array = np.asarray(image, dtype=np.uint8).copy()
    rgb_u8 = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return rgb_u8, (int(native_h), int(native_w))


def _load_camera_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("--camera-json must contain a JSON object.")
    return payload


def _camera_json_for_image(
    payload: dict[str, Any] | None,
    image_path: Path,
) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    images = payload.get("images")
    if isinstance(images, dict):
        for key in (str(image_path), image_path.as_posix(), image_path.name, image_path.stem):
            value = images.get(key)
            if isinstance(value, dict):
                base = payload.get("default", {})
                merged = dict(base) if isinstance(base, dict) else {}
                merged.update(value)
                return merged
    if isinstance(payload.get("default"), dict):
        return dict(payload["default"])
    return dict(payload)


def _values_from_camera_json(entry: dict[str, Any] | None, *names: str) -> list[float] | None:
    if not isinstance(entry, dict):
        return None
    for name in names:
        value = entry.get(name)
        if value is None:
            continue
        if isinstance(value, dict):
            if "K" in value:
                value = value["K"]
            elif all(key in value for key in ("fx", "fy", "cx", "cy")):
                distortion = [
                    float(value[key])
                    for key in (
                        "k1",
                        "k2",
                        "k3",
                        "k4",
                        "k5",
                        "k6",
                        "p1",
                        "p2",
                        "s1",
                        "s2",
                        "s3",
                        "s4",
                    )
                    if key in value
                ]
                return [float(value[key]) for key in ("fx", "fy", "cx", "cy")] + distortion
            else:
                continue
        if isinstance(value, (list, tuple)):
            if len(value) == 3 and all(isinstance(row, (list, tuple)) for row in value):
                return [float(item) for row in value for item in row]
            return [float(item) for item in value]
    return None


def _camera_name_from_json(entry: dict[str, Any] | None) -> str | None:
    if not isinstance(entry, dict):
        return None
    value = entry.get("camera", entry.get("camera_model", entry.get("type")))
    return str(value).strip().lower() if value is not None and str(value).strip() else None


def _canonical_camera_name(value: str | None) -> CameraKind | None:
    if value is None or value == "auto":
        return None
    canonical = {
        "pinhole": "perspective",
        "perspective": "perspective",
        "fisheye": "fisheye",
        "fisheye624": "fisheye",
        "opencv_fisheye": "fisheye",
        "erp": "panorama",
        "spherical": "panorama",
        "panorama": "panorama",
    }.get(str(value).strip().lower())
    if canonical is None:
        raise ValueError(f"Unsupported camera type: {value!r}")
    return canonical  # type: ignore[return-value]


def _pinhole_intrinsics_from_values(
    values: list[float] | None,
    device: torch.device,
) -> torch.Tensor | None:
    if values is None:
        return None
    vals = [float(value) for value in values]
    if len(vals) == 4:
        fx, fy, cx, cy = vals
        matrix = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
        intrinsics = torch.tensor(matrix, dtype=torch.float32, device=device)
    elif len(vals) == 9:
        intrinsics = torch.tensor(vals, dtype=torch.float32, device=device).reshape(3, 3)
    else:
        raise ValueError("Camera intrinsics must have 4 values (fx fy cx cy) or 9 row-major values.")
    return intrinsics.unsqueeze(0)


def _fisheye624_params_from_values(
    values: list[float] | None,
    device: torch.device,
) -> torch.Tensor | None:
    if values is None:
        return None
    vals = [float(value) for value in values]
    if len(vals) == 8:
        vals += [0.0] * 8
    if len(vals) != 16:
        raise ValueError("Fisheye624 parameters must contain 8 or 16 values.")
    return torch.tensor(vals, dtype=torch.float32, device=device).reshape(1, 16)


def _scale_calibration(
    intrinsics: torch.Tensor | None,
    camera_params: torch.Tensor | None,
    native_hw: tuple[int, int],
    resized_hw: tuple[int, int],
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    native_h, native_w = native_hw
    resized_h, resized_w = resized_hw
    sx = float(resized_w) / float(native_w)
    sy = float(resized_h) / float(native_h)
    if intrinsics is not None and (sx != 1.0 or sy != 1.0):
        intrinsics = intrinsics.clone()
        intrinsics[:, 0, 0] *= sx
        intrinsics[:, 0, 2] *= sx
        intrinsics[:, 1, 1] *= sy
        intrinsics[:, 1, 2] *= sy
    if camera_params is not None and (sx != 1.0 or sy != 1.0):
        camera_params = camera_params.clone()
        camera_params[:, 0] *= sx
        camera_params[:, 2] *= sx
        camera_params[:, 1] *= sy
        camera_params[:, 3] *= sy
    return intrinsics, camera_params


def _normalize_rays(rays: torch.Tensor) -> torch.Tensor:
    rays_f = rays.detach().to(torch.float32)
    return rays_f / torch.linalg.vector_norm(rays_f, dim=1, keepdim=True).clamp(min=1e-6)


def _angular_span_deg(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size < 2:
        return 0.0
    return float(np.degrees(np.nanpercentile(values, 99.0) - np.nanpercentile(values, 1.0)))


def _angle_between_deg(first: np.ndarray, second: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(first) * np.linalg.norm(second)), 1e-8)
    cosine = np.clip(float(np.dot(first, second)) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _ray_fov_stats(rays_b3hw: torch.Tensor) -> dict[str, float]:
    rays = _normalize_rays(rays_b3hw)[0].cpu().numpy()
    _, height, width = rays.shape
    rows = [max(0, min(height - 1, int(round(height * q)))) for q in (0.25, 0.5, 0.75)]
    columns = [max(0, min(width - 1, int(round(width * q)))) for q in (0.25, 0.5, 0.75)]
    horizontal_spans = []
    for row in rows:
        longitude = np.unwrap(np.arctan2(rays[0, row], rays[2, row]))
        horizontal_spans.append(_angular_span_deg(longitude))
    vertical_spans = []
    for column in columns:
        x, y, z = rays[0, :, column], rays[1, :, column], rays[2, :, column]
        latitude = np.arctan2(y, np.sqrt(x * x + z * z))
        vertical_spans.append(_angular_span_deg(latitude))
    corners = [
        rays[:, 0, 0],
        rays[:, 0, width - 1],
        rays[:, height - 1, 0],
        rays[:, height - 1, width - 1],
    ]
    diagonal = max(
        _angle_between_deg(corners[i], corners[j])
        for i in range(4)
        for j in range(i + 1, 4)
    )
    return {
        "horizontal_fov_deg": float(np.median(horizontal_spans)),
        "vertical_fov_deg": float(np.median(vertical_spans)),
        "diagonal_fov_deg": float(diagonal),
        "aspect": float(width) / float(max(height, 1)),
    }


def _classify_camera(stats: dict[str, float]) -> CameraKind:
    aspect = float(stats["aspect"])
    horizontal_fov = float(stats["horizontal_fov_deg"])
    vertical_fov = float(stats["vertical_fov_deg"])
    diagonal_fov = float(stats["diagonal_fov_deg"])
    if (
        PANORAMA_ASPECT_MIN <= aspect <= PANORAMA_ASPECT_MAX
        and horizontal_fov >= PANORAMA_HFOV_THRESHOLD_DEG
        and vertical_fov >= PANORAMA_VFOV_THRESHOLD_DEG
    ):
        return "panorama"
    if aspect <= FISHEYE_MAX_ASPECT and (
        max(horizontal_fov, vertical_fov) >= FISHEYE_FOV_THRESHOLD_DEG
        or (diagonal_fov >= FISHEYE_DIAG_THRESHOLD_DEG and vertical_fov >= FISHEYE_VFOV_MIN_DEG)
    ):
        return "fisheye"
    return "perspective"


def _release_probe_features(model: UnisharpFeatureModel) -> None:
    extractor = model.feature_extractor
    extractor._unisharp_last_unik3d_output = None
    encoder = getattr(extractor.unik3d, "pixel_encoder", None)
    if encoder is not None and hasattr(encoder, "_unisharp_last_encoder_output"):
        encoder._unisharp_last_encoder_output = None
    decoder = getattr(extractor.unik3d, "pixel_decoder", None)
    if decoder is not None and hasattr(decoder, "_unisharp_last_pred_rays_flat"):
        decoder._unisharp_last_pred_rays_flat = None
    radial = getattr(decoder, "radial_module", None)
    if radial is not None:
        for name in ("_unisharp_last_out_features", "_unisharp_last_init_latents"):
            if hasattr(radial, name):
                setattr(radial, name, None)
    gc.collect()


@torch.inference_mode()
def _predict_unik3d_rays(
    model: UnisharpFeatureModel,
    image_u8: torch.Tensor,
) -> torch.Tensor:
    _, _, height, width = image_u8.shape
    model.feature_extractor.forward(
        rgb_u8=image_u8,
        target_h=int(height),
        target_w=int(width),
        use_predicted_rays=True,
    )
    output = model.feature_extractor._unisharp_last_unik3d_output
    if not isinstance(output, dict) or not torch.is_tensor(output.get("rays")):
        raise RuntimeError("UniK3D did not return predicted camera rays.")
    rays = output["rays"].detach().clone()
    _release_probe_features(model)
    return rays


@torch.inference_mode()
def _run_model(
    model: UnisharpFeatureModel,
    image: torch.Tensor,
    image_u8: torch.Tensor,
    camera_kind: CameraKind,
    intrinsics: torch.Tensor | None,
    camera_params: torch.Tensor | None,
    distance_init_cap_m: float,
    save_aux: bool,
) -> tuple[Gaussians3D, dict[str, torch.Tensor]]:
    camera_model = {
        "perspective": "pinhole",
        "fisheye": "fisheye624",
        "panorama": "spherical",
    }[camera_kind]
    output = model(
        image=image,
        image_u8=image_u8,
        camera_intrinsics=intrinsics,
        camera_params=camera_params,
        camera_model=camera_model,
        depth_gt=None,
        distance_init_cap_m=(float(distance_init_cap_m) if float(distance_init_cap_m) > 0.0 else None),
        return_aux=bool(save_aux),
    )
    if save_aux:
        if not isinstance(output, dict) or not isinstance(output.get("gaussians"), Gaussians3D):
            raise RuntimeError("UniSHARP returned an unexpected auxiliary output.")
        gaussians = output["gaussians"]
        aux = {
            key: value.detach().cpu().contiguous()
            for key, value in output.items()
            if key
            in {
                "geometry_rays",
                "unik3d_rays",
                "unik3d_gt_rays",
                "unik3d_distance",
                "distance_layers",
            }
            and torch.is_tensor(value)
        }
        return gaussians, aux
    if not isinstance(output, Gaussians3D):
        raise RuntimeError(f"UniSHARP returned {type(output)!r}, expected Gaussians3D.")
    return output, {}


def _gaussians_to_cpu_dict(gaussians: Gaussians3D) -> dict[str, torch.Tensor]:
    return {
        "mean_vectors": gaussians.mean_vectors.detach().cpu().contiguous(),
        "singular_values": gaussians.singular_values.detach().cpu().contiguous(),
        "quaternions": gaussians.quaternions.detach().cpu().contiguous(),
        "colors": gaussians.colors.detach().cpu().contiguous(),
        "opacities": gaussians.opacities.detach().cpu().contiguous(),
    }


def _json_safe_stats(stats: dict[str, float] | None) -> dict[str, float | None] | None:
    if stats is None:
        return None
    return {key: (float(value) if math.isfinite(float(value)) else None) for key, value in stats.items()}


def _slug_from_path(image_path: Path) -> str:
    raw = f"{image_path.parent.name}_{image_path.stem}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)


def _resolve_camera(
    args: argparse.Namespace,
    model: UnisharpFeatureModel,
    image_u8: torch.Tensor,
    native_hw: tuple[int, int],
    entry: dict[str, Any] | None,
    device: torch.device,
) -> tuple[CameraKind, torch.Tensor | None, torch.Tensor | None, dict[str, float] | None]:
    _, _, height, width = image_u8.shape
    json_intrinsics = _values_from_camera_json(entry, "intrinsics", "camera_intrinsics", "K")
    json_camera_params = _values_from_camera_json(entry, "camera_params", "fisheye624_params", "params")
    intrinsics = _pinhole_intrinsics_from_values(json_intrinsics or args.camera_intrinsics, device)
    camera_params = _fisheye624_params_from_values(json_camera_params or args.camera_params, device)
    if intrinsics is not None and camera_params is not None:
        raise ValueError("Use only one of camera intrinsics or fisheye camera parameters.")
    intrinsics, camera_params = _scale_calibration(
        intrinsics,
        camera_params,
        native_hw=native_hw,
        resized_hw=(int(height), int(width)),
    )

    forced_kind = _canonical_camera_name(args.camera)
    json_kind = _canonical_camera_name(_camera_name_from_json(entry))
    if intrinsics is not None:
        if forced_kind not in (None, "perspective") or json_kind not in (None, "perspective"):
            raise ValueError("Pinhole intrinsics conflict with the selected camera type.")
        return "perspective", intrinsics, None, None
    if camera_params is not None:
        if forced_kind not in (None, "fisheye") or json_kind not in (None, "fisheye"):
            raise ValueError("Fisheye parameters conflict with the selected camera type.")
        return "fisheye", None, camera_params, None

    declared_kind = forced_kind or json_kind
    aspect = float(width) / float(max(height, 1))
    if declared_kind == "panorama" or (
        declared_kind is None and PANORAMA_ASPECT_MIN <= aspect <= PANORAMA_ASPECT_MAX
    ):
        return "panorama", None, None, None

    LOGGER.info("No calibration supplied; predicting camera rays with UniK3D")
    rays = _predict_unik3d_rays(model, image_u8)
    stats = _ray_fov_stats(rays)
    camera_kind = declared_kind or _classify_camera(stats)
    if camera_kind == "panorama":
        return camera_kind, None, None, stats
    if camera_kind == "fisheye":
        camera_params = fit_fisheye624_params_from_rays(rays).detach().to(device=device, dtype=torch.float32)
        return camera_kind, None, camera_params, stats
    intrinsics = fit_pinhole_intrinsics_from_rays(rays).detach().to(device=device, dtype=torch.float32)
    return camera_kind, intrinsics, None, stats


def _focal_for_ply(
    camera_kind: CameraKind,
    width: int,
    intrinsics: torch.Tensor | None,
    camera_params: torch.Tensor | None,
) -> float:
    if intrinsics is not None:
        return float(0.5 * (intrinsics[0, 0, 0].item() + intrinsics[0, 1, 1].item()))
    if camera_params is not None:
        return float(0.5 * (camera_params[0, 0].item() + camera_params[0, 1].item()))
    if camera_kind == "panorama":
        return float(width) / (2.0 * math.pi)
    return float(width)


def _to_u8_hwc(image_chw: torch.Tensor) -> np.ndarray:
    """Match the native inference script's image conversion."""
    if image_chw.dtype == torch.uint8:
        return image_chw.permute(1, 2, 0).detach().cpu().numpy()
    image_f = image_chw.detach().to(torch.float32).clamp(0.0, 1.0)
    return (image_f * 255.0).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()


def _crop_border_u8(frame: np.ndarray, fraction: float) -> np.ndarray:
    """Copy of the native output-frame crop helper."""
    if float(fraction) <= 0.0 or frame.ndim < 2:
        return frame
    height, width = int(frame.shape[0]), int(frame.shape[1])
    crop_y = int(round(float(height) * float(fraction)))
    crop_x = int(round(float(width) * float(fraction)))
    if crop_y <= 0 and crop_x <= 0:
        return frame
    if crop_y * 2 >= height or crop_x * 2 >= width:
        return frame
    return frame[crop_y : height - crop_y, crop_x : width - crop_x].copy()


def _save_gif(frames: list[np.ndarray], output: Path, duration_ms: int) -> None:
    if not frames:
        raise ValueError(f"No frames to save for {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    pil_frames = [Image.fromarray(frame) for frame in frames]
    pil_frames[0].save(
        output,
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(duration_ms),
        loop=0,
        disposal=2,
    )


def _build_forward_poses(num_views: int, distance_m: float, device: torch.device) -> list[torch.Tensor]:
    poses: list[torch.Tensor] = []
    r_c2w = torch.eye(3, dtype=torch.float32, device=device)
    views = max(1, int(num_views))
    for index in range(views):
        alpha = float(index + 1) / float(views)
        eye = torch.tensor([0.0, 0.0, float(distance_m) * alpha], dtype=torch.float32, device=device)
        poses.append(build_extrinsics_w2c(r_c2w, eye, "c2w"))
    return poses


def _build_rotate_poses(num_views: int, radius_m: float, device: torch.device) -> list[torch.Tensor]:
    poses: list[torch.Tensor] = []
    r_c2w = torch.eye(3, dtype=torch.float32, device=device)
    views = max(1, int(num_views))
    for index in range(views):
        theta = -2.0 * math.pi * float(index) / float(views)
        eye = torch.tensor(
            [float(radius_m) * math.sin(theta), float(radius_m) * math.cos(theta), 0.0],
            dtype=torch.float32,
            device=device,
        )
        poses.append(build_extrinsics_w2c(r_c2w, eye, "c2w"))
    return poses


def _predicted_depth_samples_m(model_output: dict[str, Any]) -> torch.Tensor | None:
    depth = model_output.get("unik3d_distance")
    if not torch.is_tensor(depth):
        layers = model_output.get("distance_layers")
        if torch.is_tensor(layers) and layers.ndim >= 4 and int(layers.shape[1]) >= 1:
            depth = layers[:, 0:1]
    if torch.is_tensor(depth) and depth.numel() > 0:
        values = depth.detach().reshape(-1).to(torch.float32)
        valid = values[torch.isfinite(values) & (values > 1e-3) & (values < 1e4)]
        if int(valid.numel()) > 0:
            return valid
    gaussians = model_output.get("gaussians")
    if isinstance(gaussians, Gaussians3D):
        values = gaussians.mean_vectors.detach().reshape(-1, 3)[..., 2].to(torch.float32)
        valid = values[torch.isfinite(values) & (values > 1e-3) & (values < 1e4)]
        if int(valid.numel()) > 0:
            return valid
    return None


def _adaptive_view_motion_distances(
    model_output: dict[str, Any],
    *,
    default_forward_m: float,
    default_radius_m: float,
) -> tuple[float, float, float | None, float, float | None, float | None]:
    """Copy of the native near-scene motion safeguard."""
    valid = _predicted_depth_samples_m(model_output)
    if valid is None or int(valid.numel()) == 0:
        effective_depth_m = median_depth_m = foreground_depth_m = None
    else:
        median_depth_m = float(torch.median(valid).item())
        foreground_depth_m = float(torch.quantile(valid, VIEW_MOTION_FOREGROUND_DEPTH_QUANTILE).item())
        effective_depth_m = (
            median_depth_m
            if median_depth_m >= VIEW_MOTION_FAR_SCENE_MEDIAN_M
            else min(median_depth_m, foreground_depth_m)
        )
    if (
        effective_depth_m is None
        or not math.isfinite(effective_depth_m)
        or effective_depth_m >= VIEW_MOTION_NEAR_SCENE_DEPTH_M
    ):
        return default_forward_m, default_radius_m, effective_depth_m, 1.0, median_depth_m, foreground_depth_m
    scale = max(VIEW_MOTION_MIN_SCALE, effective_depth_m / VIEW_MOTION_NEAR_SCENE_DEPTH_M)
    forward_m = min(default_forward_m * scale, effective_depth_m * VIEW_MOTION_FORWARD_DEPTH_FRAC)
    radius_m = min(default_radius_m * scale, effective_depth_m * VIEW_MOTION_ROTATE_DEPTH_FRAC)
    return forward_m, radius_m, effective_depth_m, scale, median_depth_m, foreground_depth_m


def _render_pinhole_frame_cpu(
    renderer: PortableGaussianRenderer,
    gaussians: Gaussians3D,
    *,
    extr_w2c: torch.Tensor,
    intrinsics: torch.Tensor,
    image_h: int,
    image_w: int,
) -> np.ndarray:
    """CPU equivalent of ``_render_pinhole_frame`` in infer_unisharp.py."""
    output = renderer(
        gaussians,
        extrinsics=extr_w2c[None],
        intrinsics=intrinsics[None],
        image_width=int(image_w),
        image_height=int(image_h),
    )
    alpha = output.alpha.detach().to(torch.float32).clamp(0.0, 1.0)
    rgb = linearRGB2sRGB((output.color / alpha.clamp(min=1e-4)).clamp(0.0, 1.0)).clamp(0.0, 1.0)
    return _to_u8_hwc(rgb[0])


def _render_multiview_cpu(
    args: argparse.Namespace,
    gaussians: Gaussians3D,
    aux: dict[str, torch.Tensor],
    intrinsics: torch.Tensor | None,
    sample_dir: Path,
    camera_kind: CameraKind,
    image_h: int,
    image_w: int,
) -> Path | None:
    """CPU port of UniSHARP's native perspective multiview inference branch."""
    if not bool(args.render_multiview):
        return None
    if camera_kind != "perspective" or intrinsics is None:
        LOGGER.warning(
            "Skipping CPU multiview rendering for %s input: the native fisheye/"
            "panorama render paths require CUDA-specific renderers. gaussians.pt was still exported.",
            camera_kind,
        )
        return None

    render_threads = int(args.render_threads) if int(args.render_threads) > 0 else int(args.threads)
    if render_threads > 0:
        torch.set_num_threads(render_threads)
    render_h = int(args.render_height) or int(image_h)
    render_w = int(args.render_width) or int(image_w)
    if (render_h, render_w) != (int(image_h), int(image_w)):
        LOGGER.warning("Custom render resolution changes the native inference output geometry: %dx%d -> %dx%d", image_w, image_h, render_w, render_h)

    device = gaussians.mean_vectors.device
    src_w2c = torch.eye(4, dtype=torch.float32, device=device)
    gaussians_world = transform_gaussians_to_world(gaussians, src_w2c)
    motion_input: dict[str, Any] = dict(aux)
    motion_input["gaussians"] = gaussians_world
    forward_m, radius_m, scene_depth_m, motion_scale, median_depth_m, foreground_depth_m = _adaptive_view_motion_distances(
        motion_input,
        default_forward_m=FORWARD_DISTANCE_M,
        default_radius_m=ROTATE_RADIUS_M,
    )
    if motion_scale < 0.999:
        LOGGER.info(
            "Near-scene view motion | depth_eff=%.3fm median=%.3fm p25=%.3fm scale=%.3f forward=%.3fm orbit=%.3fm",
            float(scene_depth_m) if scene_depth_m is not None else float("nan"),
            float(median_depth_m) if median_depth_m is not None else float("nan"),
            float(foreground_depth_m) if foreground_depth_m is not None else float("nan"),
            motion_scale,
            forward_m,
            radius_m,
        )
    renderer = PortableGaussianRenderer(
        background_color="black",
        low_pass_filter_eps=float(args.low_pass_filter_eps),
    ).to(device)
    k3 = intrinsics.detach().to(device=device, dtype=torch.float32)[0]
    forward_frames = [
        _render_pinhole_frame_cpu(renderer, gaussians_world, extr_w2c=pose, intrinsics=k3, image_h=render_h, image_w=render_w)
        for pose in _build_forward_poses(FORWARD_VIEWS, forward_m, device)
    ]
    rotate_frames = [
        _render_pinhole_frame_cpu(renderer, gaussians_world, extr_w2c=pose, intrinsics=k3, image_h=render_h, image_w=render_w)
        for pose in _build_rotate_poses(ROTATE_VIEWS, radius_m, device)
    ]
    # Native perspective inference crops the peripheral border before GIF encoding.
    forward_frames = [_crop_border_u8(frame, 0.05) for frame in forward_frames]
    rotate_frames = [_crop_border_u8(frame, 0.05) for frame in rotate_frames]
    render_dir = sample_dir / "multiview"
    _save_gif(forward_frames, render_dir / "forward.gif", GIF_DURATION_MS)
    _save_gif(rotate_frames, render_dir / "rotate.gif", GIF_DURATION_MS)
    (render_dir / "metadata.json").write_text(
        json.dumps(
            {
                "renderer": "unisharp_native_pinhole_cpu_port",
                "rasterizer": "unisharp.utils.portable_renderer.PortableGaussianRenderer",
                "forward_views": FORWARD_VIEWS,
                "rotate_views": ROTATE_VIEWS,
                "forward_distance_m": forward_m,
                "rotate_radius_m": radius_m,
                "scene_depth_for_motion_m": scene_depth_m,
                "median_predicted_depth_m": median_depth_m,
                "foreground_depth_p25_m": foreground_depth_m,
                "view_motion_scale": motion_scale,
                "low_pass_filter_eps": float(args.low_pass_filter_eps),
                "output_crop_border_fraction": 0.05,
                "height": render_h,
                "width": render_w,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return render_dir


def _process_one(
    args: argparse.Namespace,
    model: UnisharpFeatureModel,
    cfg: UnisharpFeatureConfig,
    checkpoint_step: int,
    image_path: Path,
    camera_json: dict[str, Any] | None,
    device: torch.device,
) -> None:
    started = time.perf_counter()
    rgb_u8, native_hw = _load_rgb_u8(image_path, int(args.max_long_edge))
    _, height, width = rgb_u8.shape
    image_u8 = rgb_u8.unsqueeze(0).to(device=device)
    image = image_u8.to(dtype=torch.float32).div_(255.0)
    entry = _camera_json_for_image(camera_json, image_path)

    camera_kind, intrinsics, camera_params, ray_stats = _resolve_camera(
        args=args,
        model=model,
        image_u8=image_u8,
        native_hw=native_hw,
        entry=entry,
        device=device,
    )
    LOGGER.info(
        "%s | camera=%s | input=%dx%d (native=%dx%d)",
        image_path,
        camera_kind,
        width,
        height,
        native_hw[1],
        native_hw[0],
    )

    gaussians, aux = _run_model(
        model=model,
        image=image,
        image_u8=image_u8,
        camera_kind=camera_kind,
        intrinsics=intrinsics,
        camera_params=camera_params,
        distance_init_cap_m=float(args.distance_init_cap_m),
        # Native multiview inference uses UniK3D distance to shorten motion in
        # close scenes.  Keep it in memory even when the user does not request
        # auxiliary tensors in the exported checkpoint.
        save_aux=bool(args.save_aux) or bool(args.render_multiview),
    )
    gaussian_tensors = _gaussians_to_cpu_dict(gaussians)
    num_gaussians = int(gaussian_tensors["mean_vectors"].shape[1])
    sample_dir = Path(args.out_dir) / _slug_from_path(image_path)
    sample_dir.mkdir(parents=True, exist_ok=True)

    camera_payload = {
        "kind": camera_kind,
        "model": {"perspective": "pinhole", "fisheye": "fisheye624", "panorama": "spherical"}[
            camera_kind
        ],
        "intrinsics": intrinsics.detach().cpu().contiguous() if intrinsics is not None else None,
        "camera_params": camera_params.detach().cpu().contiguous() if camera_params is not None else None,
        "source_w2c": torch.eye(4, dtype=torch.float32),
    }
    payload: dict[str, Any] = {
        "format": "unisharp_gaussians",
        "format_version": 1,
        "gaussians": gaussian_tensors,
        "camera": camera_payload,
        "image": {
            "path": str(image_path.resolve()),
            "height": int(height),
            "width": int(width),
            "native_height": int(native_hw[0]),
            "native_width": int(native_hw[1]),
        },
        "checkpoint": {
            "path": str(Path(args.checkpoint).resolve()),
            "step": int(checkpoint_step),
        },
        "model_config": dataclasses.asdict(cfg),
        "conventions": {
            "coordinates": "source_camera_opencv_x_right_y_down_z_forward",
            "quaternion_order": "wxyz",
            "colors": "linearRGB_0_to_1",
            "opacities": "activated_0_to_1",
            "singular_values": "world_scale_not_log_scale",
            "distance_unit": "meter",
        },
        "ray_fov_stats": _json_safe_stats(ray_stats),
        "aux": aux if bool(args.save_aux) else {},
    }
    gaussian_path = sample_dir / "gaussians.pt"
    torch.save(payload, gaussian_path)

    inference_elapsed_seconds = time.perf_counter() - started
    render_started = time.perf_counter()
    multiview_dir = _render_multiview_cpu(
        args=args,
        gaussians=gaussians,
        aux=aux,
        intrinsics=intrinsics,
        sample_dir=sample_dir,
        camera_kind=camera_kind,
        image_h=int(height),
        image_w=int(width),
    )
    render_elapsed_seconds = time.perf_counter() - render_started if multiview_dir is not None else None
    elapsed_seconds = time.perf_counter() - started

    metadata = {
        "format": payload["format"],
        "format_version": payload["format_version"],
        "image": payload["image"],
        "checkpoint": payload["checkpoint"],
        "model_config": payload["model_config"],
        "camera_kind": camera_kind,
        "camera_model": camera_payload["model"],
        "camera_intrinsics": camera_payload["intrinsics"].tolist()
        if torch.is_tensor(camera_payload["intrinsics"])
        else None,
        "camera_params": camera_payload["camera_params"].tolist()
        if torch.is_tensor(camera_payload["camera_params"])
        else None,
        "ray_fov_stats": payload["ray_fov_stats"],
        "num_gaussians": num_gaussians,
        "inference_elapsed_seconds": float(inference_elapsed_seconds),
        "render_elapsed_seconds": float(render_elapsed_seconds) if render_elapsed_seconds is not None else None,
        "elapsed_seconds": float(elapsed_seconds),
        "conventions": payload["conventions"],
        "outputs": {
            "pt": "gaussians.pt",
            "ply": "gaussians.ply" if bool(args.save_ply) else None,
            "multiview": "multiview" if multiview_dir is not None else None,
        },
    }
    (sample_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if args.save_ply:
        gaussians_cpu = Gaussians3D(**gaussian_tensors)
        focal_px = _focal_for_ply(camera_kind, int(width), intrinsics, camera_params)
        save_ply(
            gaussians_cpu,
            f_px=focal_px,
            image_shape=(int(height), int(width)),
            path=sample_dir / "gaussians.ply",
        )

    LOGGER.info(
        "Saved %d Gaussians to %s (%.1f s)",
        num_gaussians,
        sample_dir,
        elapsed_seconds,
    )
    del image, image_u8, gaussians, gaussian_tensors, aux, payload
    gc.collect()


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run UniSHARP on CPU and export 3D Gaussians without CUDA rendering dependencies."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--image", type=Path)
    inputs.add_argument("--image-list", type=Path)
    inputs.add_argument("--image-dir", type=Path)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "cpu_inference")
    parser.add_argument("--max-images", type=int, default=0, help="0 processes every input image.")
    parser.add_argument(
        "--max-long-edge",
        type=int,
        default=384,
        help="Resize the image for CPU inference; 0 keeps the native resolution.",
    )
    parser.add_argument(
        "--camera",
        choices=["auto", "perspective", "pinhole", "fisheye", "panorama", "erp"],
        default="auto",
    )
    parser.add_argument("--camera-json", type=Path, default=None)
    parser.add_argument(
        "--camera-intrinsics",
        type=float,
        nargs="+",
        default=None,
        help="Original-image fx fy cx cy, or a row-major 3x3 K. Values are scaled after resizing.",
    )
    parser.add_argument(
        "--camera-params",
        type=float,
        nargs="+",
        default=None,
        help="Original-image Fisheye624 parameters (8 or 16 values). Values are scaled after resizing.",
    )
    parser.add_argument("--distance-init-cap-m", type=float, default=0.0)
    parser.add_argument("--save-ply", action="store_true", help="Also export the project's SHARP-style PLY.")
    parser.add_argument(
        "--save-aux",
        action="store_true",
        help="Include rays and predicted distance tensors in gaussians.pt (uses more RAM and disk).",
    )
    parser.add_argument("--threads", type=int, default=0, help="PyTorch CPU threads; 0 keeps its default.")
    parser.add_argument(
        "--render-multiview",
        action="store_true",
        help="Render native UniSHARP-style pinhole forward/orbit GIFs with the CPU reference rasterizer.",
    )
    parser.add_argument(
        "--render-rig",
        choices=["cross5", "arc5", "grid9"],
        default=None,
        help="Deprecated Flash3D option, accepted for command compatibility and ignored.",
    )
    parser.add_argument("--render-height", type=int, default=0, help="0 reuses the inference image height.")
    parser.add_argument("--render-width", type=int, default=0, help="0 reuses the inference image width.")
    parser.add_argument(
        "--low-pass-filter-eps",
        type=float,
        default=0.0,
        help="Matches scripts/infer_unisharp.py; 0.0 is its native inference default.",
    )
    parser.add_argument(
        "--render-threads",
        type=int,
        default=0,
        help="CPU renderer threads; 0 reuses --threads (or its default).",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _build_argparser().parse_args()
    if int(args.max_long_edge) < 0:
        raise ValueError("--max-long-edge must be >= 0.")
    if int(args.render_height) < 0 or int(args.render_width) < 0:
        raise ValueError("--render-height and --render-width must be >= 0.")
    if (int(args.render_height) == 0) != (int(args.render_width) == 0):
        raise ValueError("Set both --render-height and --render-width, or leave both at 0.")
    if float(args.low_pass_filter_eps) < 0.0:
        raise ValueError("--low-pass-filter-eps must be >= 0.")
    if args.render_rig is not None:
        LOGGER.warning(
            "--render-rig is ignored: native UniSHARP multiview always renders 10 forward and 10 orbit frames."
        )
    if int(args.threads) > 0:
        torch.set_num_threads(int(args.threads))
    torch.set_float32_matmul_precision("high")
    device = torch.device("cpu")
    _configure_caches()
    camera_json = _load_camera_json(args.camera_json)
    image_paths = _collect_image_paths(args)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    model, cfg, checkpoint_step = _load_model(Path(args.checkpoint), device)
    LOGGER.info("Processing %d image(s) with %d CPU thread(s)", len(image_paths), torch.get_num_threads())
    for image_path in image_paths:
        _process_one(
            args=args,
            model=model,
            cfg=cfg,
            checkpoint_step=checkpoint_step,
            image_path=Path(image_path),
            camera_json=camera_json,
            device=device,
        )


if __name__ == "__main__":
    main()

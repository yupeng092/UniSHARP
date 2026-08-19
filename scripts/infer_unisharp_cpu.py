from __future__ import annotations

"""CPU-only UniSHARP Gaussian prediction.

This script intentionally does not import gsplat, Triton, or the 3DGEER CUDA
rasterizer.  It runs the UniK3D/UniSHARP network and stores the predicted 3D
Gaussians for a separate renderer.
"""

import argparse
import dataclasses
import gc
import json
import logging
import math
import os
import re
import subprocess
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
from unisharp.utils.gaussians import Gaussians3D, save_ply  # noqa: E402
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
DEFAULT_FLASH3D_ROOT = Path(os.environ.get("FLASH3D_ROOT", r"D:\PythonFiles\flash3d-main"))


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


def _render_multiview_cpu(
    args: argparse.Namespace,
    gaussian_path: Path,
    sample_dir: Path,
    camera_kind: CameraKind,
) -> Path | None:
    """Render the exported perspective scene with Flash3D's CPU entry point.

    UniSHARP's native renderer has CUDA-only gsplat/GEER dependencies.  The
    Flash3D renderer understands the ``unisharp_gaussians`` payload directly,
    so keeping rendering in a subprocess also avoids introducing those
    optional dependencies into the CPU inference environment.
    """
    if not bool(args.render_multiview):
        return None
    if camera_kind != "perspective":
        LOGGER.warning(
            "Skipping CPU multiview rendering for %s input: the configured Flash3D "
            "renderer accepts pinhole scenes only. gaussians.pt was still exported.",
            camera_kind,
        )
        return None

    flash3d_root = Path(args.flash3d_root)
    renderer_script = flash3d_root / "render_cpu_multiview.py"
    if not renderer_script.is_file():
        raise FileNotFoundError(
            "Flash3D CPU renderer not found: "
            f"{renderer_script}. Set --flash3d-root or FLASH3D_ROOT."
        )
    renderer_python = Path(args.renderer_python) if args.renderer_python is not None else Path(sys.executable)
    if not renderer_python.is_file():
        raise FileNotFoundError(f"Renderer Python executable not found: {renderer_python}")

    render_dir = sample_dir / "multiview"
    command = [
        str(renderer_python),
        str(renderer_script),
        "--gaussians",
        str(gaussian_path),
        "--output",
        str(render_dir),
        "--backend",
        "torch",
        "--rig",
        str(args.render_rig),
        "--baseline",
        str(float(args.render_baseline)),
        "--height",
        str(int(args.render_height)),
        "--width",
        str(int(args.render_width)),
        "--keep-ratio",
        str(float(args.render_keep_ratio)),
        "--use-source-intrinsics",
        "--linear-to-srgb",
    ]
    render_threads = int(args.render_threads) if int(args.render_threads) > 0 else int(args.threads)
    if render_threads > 0:
        command.extend(["--threads", str(render_threads)])

    LOGGER.info("Rendering CPU multiview images with %s", renderer_script)
    subprocess.run(command, check=True, cwd=str(flash3d_root))
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
        save_aux=bool(args.save_aux),
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
        "aux": aux,
    }
    gaussian_path = sample_dir / "gaussians.pt"
    torch.save(payload, gaussian_path)

    inference_elapsed_seconds = time.perf_counter() - started
    render_started = time.perf_counter()
    multiview_dir = _render_multiview_cpu(
        args=args,
        gaussian_path=gaussian_path,
        sample_dir=sample_dir,
        camera_kind=camera_kind,
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
        help="After inference, render a CPU multiview rig with Flash3D's render_cpu_multiview.py.",
    )
    parser.add_argument(
        "--flash3d-root",
        type=Path,
        default=DEFAULT_FLASH3D_ROOT,
        help="Flash3D directory; defaults to FLASH3D_ROOT or D:/PythonFiles/flash3d-main.",
    )
    parser.add_argument(
        "--renderer-python",
        type=Path,
        default=None,
        help="Python executable for Flash3D rendering; defaults to this Python interpreter.",
    )
    parser.add_argument("--render-rig", choices=["cross5", "arc5", "grid9"], default="cross5")
    parser.add_argument("--render-baseline", type=float, default=0.5)
    parser.add_argument("--render-height", type=int, default=256)
    parser.add_argument("--render-width", type=int, default=384)
    parser.add_argument(
        "--render-keep-ratio",
        type=float,
        default=1.0,
        help="Fraction of exported Gaussians to keep for CPU rendering, in (0, 1].",
    )
    parser.add_argument(
        "--render-threads",
        type=int,
        default=0,
        help="Flash3D renderer CPU threads; 0 reuses --threads (or its default).",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _build_argparser().parse_args()
    if int(args.max_long_edge) < 0:
        raise ValueError("--max-long-edge must be >= 0.")
    if int(args.render_height) <= 0 or int(args.render_width) <= 0:
        raise ValueError("--render-height and --render-width must be positive.")
    if not 0.0 < float(args.render_keep_ratio) <= 1.0:
        raise ValueError("--render-keep-ratio must be in (0, 1].")
    if float(args.render_baseline) < 0.0:
        raise ValueError("--render-baseline must be >= 0.")
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

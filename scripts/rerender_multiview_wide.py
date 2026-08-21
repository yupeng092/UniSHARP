"""Re-render UniSHARP multiview GIFs from an exported gaussians.pt with larger
camera-position separation, to produce clearly different viewpoints.

This mirrors the perspective CPU rendering branch of
``scripts/infer_unisharp_cpu.py`` (same source->world transform, depth-free
fixed forward/orbit trajectories, black background, alpha normalisation, sRGB
conversion and 5% output crop) but lets you set the forward dolly distance and
orbit radius directly. The Gaussian prediction is deterministic, so re-rendering
from the saved gaussians.pt is equivalent to a full re-run with the same
trajectory, only much faster because UniK3D/UniSHARP inference is skipped.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from unisharp.utils.camera_projection import build_extrinsics_w2c  # noqa: E402
from unisharp.utils.camera_utils import transform_gaussians_to_world  # noqa: E402
from unisharp.utils.color_space import linearRGB2sRGB  # noqa: E402
from unisharp.utils.gaussians import Gaussians3D  # noqa: E402
from unisharp.utils.portable_renderer import PortableGaussianRenderer  # noqa: E402

FORWARD_VIEWS = 10
ROTATE_VIEWS = 10
GIF_DURATION_MS = 300


def _load_payload(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _to_u8_hwc(image_chw: torch.Tensor) -> np.ndarray:
    image_f = image_chw.detach().to(torch.float32).clamp(0.0, 1.0)
    return (image_f * 255.0).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()


def _crop_native_border(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    crop_y, crop_x = int(round(height * 0.05)), int(round(width * 0.05))
    if crop_y * 2 >= height or crop_x * 2 >= width:
        return frame
    return frame[crop_y : height - crop_y, crop_x : width - crop_x].copy()


def _scale_render_intrinsics(intrinsics: torch.Tensor, src_h: int, src_w: int, out_h: int, out_w: int) -> torch.Tensor:
    result = intrinsics.detach().to(torch.float32).clone()
    result[..., 0, 0] *= float(out_w) / float(src_w)
    result[..., 0, 2] *= float(out_w) / float(src_w)
    result[..., 1, 1] *= float(out_h) / float(src_h)
    result[..., 1, 2] *= float(out_h) / float(src_h)
    return result


def _build_forward_poses(num_views: int, distance_m: float, device: torch.device) -> list[torch.Tensor]:
    rotation = torch.eye(3, dtype=torch.float32, device=device)
    poses = []
    for idx in range(num_views):
        alpha = float(idx + 1) / float(num_views)
        eye = torch.tensor([0.0, 0.0, float(distance_m) * alpha], dtype=torch.float32, device=device)
        poses.append(build_extrinsics_w2c(rotation, eye, "c2w"))
    return poses


def _build_orbit_poses(num_views: int, radius_m: float, device: torch.device) -> list[torch.Tensor]:
    rotation = torch.eye(3, dtype=torch.float32, device=device)
    poses = []
    for idx in range(num_views):
        theta = -2.0 * math.pi * float(idx) / float(num_views)
        eye = torch.tensor(
            [float(radius_m) * math.sin(theta), float(radius_m) * math.cos(theta), 0.0],
            dtype=torch.float32,
            device=device,
        )
        poses.append(build_extrinsics_w2c(rotation, eye, "c2w"))
    return poses


def _render_frame(renderer, gaussians, pose, k3, height, width) -> np.ndarray:
    output = renderer(gaussians, extrinsics=pose[None], intrinsics=k3[None], image_width=width, image_height=height)
    rgb = linearRGB2sRGB((output.color / output.alpha.clamp(min=1e-4)).clamp(0.0, 1.0)).clamp(0.0, 1.0)
    return _to_u8_hwc(rgb[0])


def _save_gif(frames: list[np.ndarray], out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    pil_frames = [Image.fromarray(frame) for frame in frames]
    pil_frames[0].save(
        out_file,
        save_all=True,
        append_images=pil_frames[1:],
        duration=GIF_DURATION_MS,
        loop=0,
        disposal=2,
    )


def _build_combined(forward_frames: list[np.ndarray], rotate_frames: list[np.ndarray], pad: int = 1) -> np.ndarray:
    n_cols = max(len(forward_frames), len(rotate_frames))
    n_rows = 2
    cell_h, cell_w = forward_frames[0].shape[:2]
    total_w = n_cols * cell_w + (n_cols + 1) * pad
    total_h = n_rows * cell_h + (n_rows + 1) * pad
    canvas = np.full((total_h, total_w, 3), 255, dtype=np.uint8)
    for c, frame in enumerate(forward_frames):
        x = pad + c * (cell_w + pad)
        y = pad
        canvas[y : y + cell_h, x : x + cell_w] = frame
    for c, frame in enumerate(rotate_frames):
        x = pad + c * (cell_w + pad)
        y = pad + cell_h + pad
        canvas[y : y + cell_h, x : x + cell_w] = frame
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-render UniSHARP multiview GIFs with larger viewpoint separation.")
    parser.add_argument("--gaussians", type=Path, required=True, help="Exported gaussians.pt from infer_unisharp_cpu.py.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for the new multiview.")
    parser.add_argument("--forward-distance-m", type=float, default=0.6, help="Total forward dolly distance in metres (default 0.6 = 3x native 0.2).")
    parser.add_argument("--rotate-radius-m", type=float, default=0.3, help="Orbit radius in metres (default 0.3 = 3x native 0.1).")
    parser.add_argument("--render-width", type=int, default=0, help="0 reuses the inference image width.")
    parser.add_argument("--render-height", type=int, default=0, help="0 reuses the inference image height.")
    parser.add_argument("--threads", type=int, default=0, help="CPU threads; 0 uses all cores.")
    parser.add_argument("--low-pass-filter-eps", type=float, default=0.0)
    args = parser.parse_args()

    if int(args.render_width) < 0 or int(args.render_height) < 0:
        raise ValueError("--render-width and --render-height must be >= 0.")
    if (int(args.render_width) == 0) != (int(args.render_height) == 0):
        raise ValueError("Set both --render-width and --render-height, or leave both at 0.")
    if int(args.threads) > 0:
        torch.set_num_threads(int(args.threads))
    else:
        torch.set_num_threads(max(1, torch.get_num_threads()))
    torch.set_float32_matmul_precision("high")

    device = torch.device("cpu")
    payload = _load_payload(args.gaussians)
    g = payload["gaussians"]
    gaussians = Gaussians3D(
        mean_vectors=g["mean_vectors"].to(device),
        singular_values=g["singular_values"].to(device),
        quaternions=g["quaternions"].to(device),
        colors=g["colors"].to(device),
        opacities=g["opacities"].to(device),
    )
    intrinsics = payload["camera"]["intrinsics"].to(device=device, dtype=torch.float32)
    src_w2c = payload["camera"].get("source_w2c", torch.eye(4, dtype=torch.float32)).to(device=device, dtype=torch.float32)
    src_h = int(payload["image"]["height"])
    src_w = int(payload["image"]["width"])
    out_h = int(args.render_height) or src_h
    out_w = int(args.render_width) or src_w

    world = transform_gaussians_to_world(gaussians, src_w2c)
    k3 = _scale_render_intrinsics(intrinsics, src_h, src_w, out_h, out_w).to(device=device)[0]
    renderer = PortableGaussianRenderer(
        background_color="black",
        low_pass_filter_eps=float(args.low_pass_filter_eps),
    ).to(device)

    forward_poses = _build_forward_poses(FORWARD_VIEWS, float(args.forward_distance_m), device)
    orbit_poses = _build_orbit_poses(ROTATE_VIEWS, float(args.rotate_radius_m), device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = args.out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    forward_frames: list[np.ndarray] = []
    for idx, pose in enumerate(forward_poses):
        t0 = time.perf_counter()
        frame = _crop_native_border(_render_frame(renderer, world, pose, k3, out_h, out_w))
        Image.fromarray(frame).save(frames_dir / f"forward_{idx:02d}.png")
        forward_frames.append(frame)
        print(f"forward {idx:02d}/{FORWARD_VIEWS} done in {time.perf_counter() - t0:.1f}s", flush=True)

    rotate_frames: list[np.ndarray] = []
    for idx, pose in enumerate(orbit_poses):
        t0 = time.perf_counter()
        frame = _crop_native_border(_render_frame(renderer, world, pose, k3, out_h, out_w))
        Image.fromarray(frame).save(frames_dir / f"rotate_{idx:02d}.png")
        rotate_frames.append(frame)
        print(f"rotate {idx:02d}/{ROTATE_VIEWS} done in {time.perf_counter() - t0:.1f}s", flush=True)

    _save_gif(forward_frames, args.out_dir / "forward.gif")
    _save_gif(rotate_frames, args.out_dir / "rotate.gif")
    combined = _build_combined(forward_frames, rotate_frames)
    Image.fromarray(combined).save(args.out_dir / "combined.png")

    metadata = {
        "renderer": "unisharp_gsplat_cpu_reference",
        "source_gaussians": str(args.gaussians.resolve()),
        "forward_distance_m": float(args.forward_distance_m),
        "rotate_radius_m": float(args.rotate_radius_m),
        "forward_distance_m_native_default": 0.2,
        "rotate_radius_m_native_default": 0.1,
        "motion_multiplier": float(args.forward_distance_m) / 0.2,
        "forward_views": FORWARD_VIEWS,
        "rotate_views": ROTATE_VIEWS,
        "height": int(forward_frames[0].shape[0]),
        "width": int(forward_frames[0].shape[1]),
        "inference_resolution": [src_h, src_w],
        "low_pass_filter_eps": float(args.low_pass_filter_eps),
        "threads": int(torch.get_num_threads()),
        "note": "Re-rendered from saved gaussians.pt with enlarged camera trajectory; Gaussian prediction unchanged.",
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()

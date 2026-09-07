#!/usr/bin/env python3
"""Encode named UniSHARP multiview RGB renders as an animated GIF.

The script consumes ``multiview_report.json`` written by
``scripts/render_unisharp_cpu.py``.  That report is the source of truth for
the rendered camera names and RGB paths, so no filename sorting assumptions
are needed.  Alongside the GIF it writes a JSON manifest recording the source
image, camera rig, view sequence and every file used to make the animation.

Example
-------
python scripts/make_multiview_gif.py --config configs/multiview_gif_demo_20260831094111_73_2.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image


FORMAT_NAME = "unisharp_multiview_gif"
FORMAT_VERSION = 1
DEFAULT_DURATION_MS = 260


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    """Read and validate one JSON object from disk."""
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid {label} JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _path_from_value(value: str | Path | None, base_dir: Path, label: str, required: bool = True) -> Path | None:
    """Resolve a command-line/config path, allowing config-relative values."""
    if value is None:
        if required:
            raise ValueError(f"Missing required setting: {label}")
        return None
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def _choose(args_value: Any, config: dict[str, Any], key: str, default: Any = None) -> Any:
    """Use an explicit CLI value first, then config, then the supplied default."""
    if args_value is not None:
        return args_value
    return config.get(key, default)


def _sha256(path: Path) -> str:
    """Return a streaming SHA-256 checksum without loading an entire file at once."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _camera_index(report: dict[str, Any], report_path: Path) -> dict[str, dict[str, Any]]:
    """Index report cameras by name and resolve their RGB image paths."""
    cameras = report.get("cameras")
    if not isinstance(cameras, list) or not cameras:
        raise ValueError("render report must contain a non-empty 'cameras' list")
    indexed: dict[str, dict[str, Any]] = {}
    for camera in cameras:
        if not isinstance(camera, dict):
            raise ValueError("every render-report camera entry must be an object")
        name = camera.get("name")
        rgb = camera.get("rgb")
        if not isinstance(name, str) or not name:
            raise ValueError("every render-report camera needs a non-empty 'name'")
        if not isinstance(rgb, str) or not rgb:
            raise ValueError(f"camera '{name}' has no RGB render path")
        if name in indexed:
            raise ValueError(f"render report repeats camera name: {name}")
        rgb_path = Path(rgb)
        camera_copy = dict(camera)
        camera_copy["_rgb_path"] = rgb_path if rgb_path.is_absolute() else (report_path.parent / rgb_path).resolve()
        indexed[name] = camera_copy
    return indexed


def _load_frames(cameras: list[dict[str, Any]]) -> tuple[list[Image.Image], tuple[int, int]]:
    """Load RGB frames and ensure a GIF has one consistent canvas size."""
    frames: list[Image.Image] = []
    size: tuple[int, int] | None = None
    for camera in cameras:
        image_path = camera["_rgb_path"]
        if not image_path.is_file():
            raise FileNotFoundError(f"rendered RGB image does not exist: {image_path}")
        with Image.open(image_path) as image:
            frame = image.convert("RGB").copy()
        if size is None:
            size = frame.size
        elif frame.size != size:
            raise ValueError(f"GIF frames must have one size; {image_path} is {frame.size}, expected {size}")
        frames.append(frame)
    if size is None:
        raise ValueError("No frames selected")
    return frames, size


def _ping_pong(frames: list[Image.Image], cameras: list[dict[str, Any]]) -> tuple[list[Image.Image], list[dict[str, Any]]]:
    """Append the reverse interior sequence for a loop without a hard turn-around cut."""
    if len(frames) < 2:
        return frames, cameras
    return frames + frames[-2:0:-1], cameras + cameras[-2:0:-1]


def _save_gif(frames: list[Image.Image], output_path: Path, duration_ms: int, loop: int) -> None:
    """Write an animated, indefinitely looping GIF with disposal between frames."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=loop,
        disposal=2,
        optimize=False,
    )


def _camera_manifest_entry(camera: dict[str, Any]) -> dict[str, Any]:
    """Select the reproducibility-relevant camera fields for the output manifest."""
    return {
        "name": camera["name"],
        "rgb": str(camera["_rgb_path"].resolve()),
        "rgb_sha256": _sha256(camera["_rgb_path"]),
        "position_xyz": camera.get("position_xyz"),
        "look_at_xyz": camera.get("look_at_xyz"),
        "roll_deg": camera.get("roll_deg"),
        "source_camera": camera.get("source_camera"),
    }


def _build_parser() -> argparse.ArgumentParser:
    """Define both standalone CLI use and config-file driven use."""
    parser = argparse.ArgumentParser(description="Create a GIF and JSON manifest from UniSHARP multiview RGB renders.")
    parser.add_argument("--config", type=Path, help="JSON configuration file. CLI options override its values.")
    parser.add_argument("--render-report", type=Path, default=None, help="multiview_report.json from render_unisharp_cpu.py.")
    parser.add_argument("--source-image", type=Path, default=None, help="Original image used to infer the Gaussian scene.")
    parser.add_argument("--camera-rig", type=Path, default=None, help="Camera rig JSON used for the render, if available.")
    parser.add_argument("--output", type=Path, default=None, help="Destination .gif path.")
    parser.add_argument("--manifest", type=Path, default=None, help="Destination JSON manifest; default is beside the GIF.")
    parser.add_argument("--duration-ms", type=int, default=None, help=f"Per-frame duration; default {DEFAULT_DURATION_MS} ms.")
    parser.add_argument("--loop", type=int, default=None, help="GIF loop count; 0 means loop forever.")
    parser.add_argument("--views", nargs="+", default=None, help="Camera names in animation order; default follows the render report.")
    loop_mode = parser.add_mutually_exclusive_group()
    loop_mode.add_argument("--ping-pong", dest="ping_pong", action="store_true", default=None, help="Append reverse interior frames for a smooth loop.")
    loop_mode.add_argument("--no-ping-pong", dest="ping_pong", action="store_false", help="Use each selected frame exactly once.")
    return parser


def main() -> None:
    """Resolve configuration, encode the selected renders, then write its manifest."""
    args = _build_parser().parse_args()
    config_path = args.config.resolve() if args.config is not None else None
    config = _read_json_object(config_path, "config") if config_path is not None else {}
    config_dir = config_path.parent if config_path is not None else Path.cwd()

    report_path = _path_from_value(_choose(args.render_report, config, "render_report"), config_dir, "render_report")
    source_image = _path_from_value(_choose(args.source_image, config, "source_image"), config_dir, "source_image", required=False)
    camera_rig = _path_from_value(_choose(args.camera_rig, config, "camera_rig"), config_dir, "camera_rig", required=False)
    output_path = _path_from_value(_choose(args.output, config, "output_gif"), config_dir, "output_gif")
    manifest_path = _path_from_value(_choose(args.manifest, config, "output_manifest"), config_dir, "output_manifest", required=False)
    if manifest_path is None:
        manifest_path = output_path.with_suffix(".json")
    duration_ms = int(_choose(args.duration_ms, config, "frame_duration_ms", DEFAULT_DURATION_MS))
    loop = int(_choose(args.loop, config, "loop", 0))
    ping_pong = bool(_choose(args.ping_pong, config, "ping_pong", False))
    requested_views = _choose(args.views, config, "views", None)
    if duration_ms <= 0:
        raise ValueError("frame_duration_ms must be positive")
    if loop < 0:
        raise ValueError("loop must be >= 0")
    if requested_views is not None and (not isinstance(requested_views, list) or not all(isinstance(name, str) for name in requested_views)):
        raise ValueError("views must be a JSON array of camera-name strings")

    report = _read_json_object(report_path, "render report")
    camera_by_name = _camera_index(report, report_path)
    view_names = requested_views or list(camera_by_name)
    if not view_names:
        raise ValueError("at least one view must be selected")
    missing = [name for name in view_names if name not in camera_by_name]
    if missing:
        raise ValueError(f"requested views absent from render report: {', '.join(missing)}")
    selected_cameras = [camera_by_name[name] for name in view_names]
    frames, (width, height) = _load_frames(selected_cameras)
    encoded_frames, encoded_cameras = _ping_pong(frames, selected_cameras) if ping_pong else (frames, selected_cameras)
    _save_gif(encoded_frames, output_path, duration_ms, loop)

    manifest = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "source_image": str(source_image.resolve()) if source_image is not None else None,
        "render_report": str(report_path.resolve()),
        "camera_rig": str(camera_rig.resolve()) if camera_rig is not None else None,
        "gif": {
            "path": str(output_path.resolve()),
            "sha256": _sha256(output_path),
            "width": width,
            "height": height,
            "duration_ms": duration_ms,
            "loop": loop,
            "ping_pong": ping_pong,
            "selected_frame_count": len(selected_cameras),
            "encoded_frame_count": len(encoded_cameras),
        },
        "render": {
            "backend": report.get("backend"),
            "renderer": report.get("renderer"),
            "gaussians_input": report.get("gaussians_input"),
            "image_size_hw": report.get("image_size_hw"),
            "camera_orientation": report.get("camera_orientation"),
        },
        "selected_views": [_camera_manifest_entry(camera) for camera in selected_cameras],
        "encoded_view_order": [camera["name"] for camera in encoded_cameras],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"GIF saved: {output_path}")
    print(f"Manifest saved: {manifest_path}")


if __name__ == "__main__":
    main()

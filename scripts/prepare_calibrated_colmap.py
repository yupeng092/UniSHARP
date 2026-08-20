from __future__ import annotations

"""Convert publicly released COLMAP text reconstructions to UniSHARP JSONL.

This is used by the direct-download NeuMan package and is also useful for any
open image/video corpus that contains ``cameras.txt``, ``images.txt`` and the
corresponding RGB images.  The output uses OpenCV/COLMAP world-to-camera poses.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def _intrinsics(tokens: list[str]) -> list[list[float]]:
    model = tokens[1]
    params = [float(value) for value in tokens[4:]]
    if model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
        fx = fy = params[0]
        cx, cy = params[1:3]
    elif model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "FOV", "THIN_PRISM_FISHEYE"}:
        fx, fy, cx, cy = params[:4]
    else:
        raise ValueError(f"Unsupported COLMAP camera model: {model}")
    return [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]


def _quaternion_to_rotation(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    q = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _read_cameras(path: Path) -> dict[int, list[list[float]]]:
    cameras: dict[int, list[list[float]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        cameras[int(tokens[0])] = _intrinsics(tokens)
    return cameras


def _resolve_image(name: str, *, model_dir: Path, source_root: Path, image_root: Path | None) -> Path | None:
    roots = [image_root] if image_root is not None else []
    roots.extend((model_dir / "images", model_dir.parent / "images", model_dir.parent.parent / "images", source_root / "images"))
    for root in roots:
        if root is None:
            continue
        path = root / name
        if path.is_file():
            return path.resolve()
    return None


def _read_frames(model_dir: Path, *, source_root: Path, image_root: Path | None) -> list[dict[str, object]]:
    cameras = _read_cameras(model_dir / "cameras.txt")
    lines = (model_dir / "images.txt").read_text(encoding="utf-8").splitlines()
    frames: list[dict[str, object]] = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        if len(tokens) < 10:
            raise ValueError(f"Malformed COLMAP image row in {model_dir / 'images.txt'}: {line}")
        # COLMAP stores one following row of 2D points (which can be blank).
        if index < len(lines):
            index += 1
        qw, qx, qy, qz, tx, ty, tz = (float(value) for value in tokens[1:8])
        camera_id, image_name = int(tokens[8]), tokens[9]
        image_path = _resolve_image(image_name, model_dir=model_dir, source_root=source_root, image_root=image_root)
        if image_path is None or camera_id not in cameras:
            continue
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = _quaternion_to_rotation(qw, qx, qy, qz)
        w2c[:3, 3] = (tx, ty, tz)
        frames.append({"image": str(image_path), "intrinsics": cameras[camera_id], "w2c": w2c.tolist()})
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=None, help="Optional root used before automatic image-root discovery.")
    parser.add_argument("--max-scenes", type=int, default=0, help="0 keeps every valid COLMAP reconstruction.")
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    model_dirs = sorted(path.parent for path in source_root.rglob("cameras.txt") if (path.parent / "images.txt").is_file())
    if int(args.max_scenes) > 0:
        model_dirs = model_dirs[: int(args.max_scenes)]
    rows: list[str] = []
    for model_dir in model_dirs:
        frames = _read_frames(model_dir, source_root=source_root, image_root=args.image_root)
        if len(frames) >= 2:
            rows.append(json.dumps({"scene": str(model_dir.relative_to(source_root)), "frames": frames}))
    if not rows:
        raise SystemExit("No COLMAP text reconstructions with at least two resolvable RGB frames were found.")
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} calibrated scenes: {args.manifest}")


if __name__ == "__main__":
    main()

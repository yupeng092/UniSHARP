from __future__ import annotations

"""Pack a frame-and-camera RE10K export into UniSHARP ``train/*.torch`` chunks.

Expected input (for example, the result of the public pixelSplat/LVSM RE10K
preprocessor):
  <source>/<split>/images/<scene>/*.{png,jpg,jpeg}
  <source>/<split>/metadata/<scene>.json

Each metadata frame must contain pixel intrinsics (``fxfycxcy`` or ``fx``,
``fy``, ``cx``, ``cy``) and a 4x4 ``w2c`` / ``world_to_camera`` matrix. A
``c2w`` / ``camera_to_world`` matrix is accepted and inverted explicitly.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch

# Keep this preparation script directly executable via
# ``python scripts/prepare_re10k_chunks.py``.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from unisharp.utils.rigid_transform import invert_rigid_transform


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


def _frames(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("frames", "views", "images", "data"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    if all(isinstance(value, dict) for value in payload.values()):
        out: list[dict[str, Any]] = []
        for key, value in payload.items():
            item = dict(value)
            item.setdefault("frame_id", key)
            out.append(item)
        return out
    return []


def _image_path(frame: dict[str, Any], scene_images: Path, fallback: Path | None) -> Path | None:
    for key in ("file_path", "image_path", "path", "filename", "image_name"):
        value = frame.get(key)
        if not value:
            continue
        candidate = Path(str(value))
        candidates = (candidate, scene_images / candidate, scene_images / candidate.name)
        for path in candidates:
            if path.is_file():
                return path
    return fallback if fallback is not None and fallback.is_file() else None


def _intrinsics(frame: dict[str, Any]) -> tuple[float, float, float, float] | None:
    values = frame.get("fxfycxcy", frame.get("intrinsics"))
    if isinstance(values, dict):
        values = [values.get("fx"), values.get("fy"), values.get("cx"), values.get("cy")]
    if isinstance(values, (list, tuple)) and len(values) >= 4:
        fx, fy, cx, cy = (float(values[i]) for i in range(4))
    elif all(key in frame for key in ("fx", "fy", "cx", "cy")):
        fx, fy, cx, cy = (float(frame[key]) for key in ("fx", "fy", "cx", "cy"))
    else:
        return None
    return (fx, fy, cx, cy) if fx > 0.0 and fy > 0.0 else None


def _w2c(frame: dict[str, Any]) -> torch.Tensor | None:
    for key, invert in (("w2c", False), ("world_to_camera", False), ("c2w", True), ("camera_to_world", True), ("transform_matrix", True)):
        value = frame.get(key)
        if value is None:
            continue
        matrix = torch.as_tensor(value, dtype=torch.float32)
        if matrix.numel() != 16:
            continue
        matrix = matrix.reshape(4, 4)
        try:
            return invert_rigid_transform(matrix) if invert else matrix
        except (RuntimeError, ValueError):
            continue
    return None


def _camera_row(k: tuple[float, float, float, float], h: int, w: int, w2c: torch.Tensor) -> torch.Tensor:
    fx, fy, cx, cy = k
    row = torch.zeros(18, dtype=torch.float32)
    # Re10KDataset converts these back as fx*w, fy*h, cx*w-0.5, cy*h-0.5.
    row[:4] = torch.tensor((fx / w, fy / h, (cx + 0.5) / w, (cy + 0.5) / h), dtype=torch.float32)
    row[6:] = w2c[:3, :].reshape(-1)
    return row


def _encoded_bytes(path: Path) -> torch.Tensor:
    return torch.tensor(bytearray(path.read_bytes()), dtype=torch.uint8)


def _scene_example(scene: str, image_dir: Path, metadata_path: Path) -> dict[str, Any] | None:
    raw_images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not raw_images:
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"skip {scene}: cannot read {metadata_path}: {exc}")
        return None
    frames = _frames(metadata)
    if not frames:
        print(f"skip {scene}: no frame list in {metadata_path}")
        return None
    packed_images: list[torch.Tensor] = []
    cameras: list[torch.Tensor] = []
    for index, frame in enumerate(frames):
        image = _image_path(frame, image_dir, raw_images[index] if index < len(raw_images) else None)
        k = _intrinsics(frame)
        pose = _w2c(frame)
        if image is None or k is None or pose is None:
            continue
        try:
            from PIL import Image

            with Image.open(image) as im:
                w, h = im.size
            packed_images.append(_encoded_bytes(image))
            cameras.append(_camera_row(k, h, w, pose))
        except Exception as exc:
            print(f"skip frame {image}: {exc}")
    if len(packed_images) < 2:
        print(f"skip {scene}: fewer than two valid frames")
        return None
    return {"key": str(scene), "images": packed_images, "cameras": torch.stack(cameras, dim=0)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True, help="Output RE10K root; chunks go in <output-root>/<split>.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--scenes-per-chunk", type=int, default=64)
    parser.add_argument("--max-scenes", type=int, default=0, help="0 packs every available scene.")
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing chunk files in the explicit output directory.")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if int(args.scenes_per_chunk) < 1 or int(args.max_scenes) < 0:
        raise SystemExit("--scenes-per-chunk must be positive and --max-scenes non-negative.")
    source_split = Path(args.source_root) / str(args.split)
    images_root, metadata_root = source_split / "images", source_split / "metadata"
    if not images_root.is_dir() or not metadata_root.is_dir():
        raise SystemExit(f"Expected {images_root} and {metadata_root}; run the public RE10K frame/metadata preprocessor first.")
    scene_ids = sorted(path.stem for path in metadata_root.glob("*.json") if (images_root / path.stem).is_dir())
    if int(args.max_scenes) > 0:
        scene_ids = scene_ids[: int(args.max_scenes)]
    if not scene_ids:
        raise SystemExit("No matching RE10K image directories and metadata JSON files were found.")
    output_split = Path(args.output_root) / str(args.split)
    if output_split.exists() and any(output_split.glob("*.torch")) and not args.overwrite:
        raise SystemExit(f"Chunk files already exist in {output_split}; use a new output root or pass --overwrite.")
    output_split.mkdir(parents=True, exist_ok=True)
    chunks: list[Path] = []
    current: list[dict[str, Any]] = []
    packed = 0
    for scene in scene_ids:
        example = _scene_example(scene, images_root / scene, metadata_root / f"{scene}.json")
        if example is None:
            continue
        current.append(example)
        packed += 1
        if len(current) == int(args.scenes_per_chunk):
            path = output_split / f"chunk_{len(chunks):05d}.torch"
            torch.save(current, path)
            chunks.append(path)
            current = []
    if current:
        path = output_split / f"chunk_{len(chunks):05d}.torch"
        torch.save(current, path)
        chunks.append(path)
    if not chunks:
        raise SystemExit("No valid scenes were packed; inspect metadata keys and camera conventions.")
    manifest = Path(args.manifest_path) if args.manifest_path else Path(args.output_root) / f"{args.split}_chunks.txt"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(path.name for path in chunks) + "\n", encoding="utf-8")
    print(f"Packed {packed} scenes into {len(chunks)} chunk(s): {output_split}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()

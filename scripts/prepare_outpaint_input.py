#!/usr/bin/env python3
"""Prepare a border-expanded image and matching camera JSON for UniSHARP.

The original image is pasted into the enlarged canvas *without resizing*.
Accordingly, only the principal point changes:

    fx' = fx, fy' = fy, cx' = cx + left_pad, cy' = cy + top_pad

``reflect`` is a dependency-free smoke-test fill.  It is useful for validating
the geometry and the UniSHARP handoff, but it does not invent a semantically
correct background.  For a real outpainting model, pass its full-canvas result
with ``--filled-image``; this script will still restore the source rectangle
pixel-for-pixel before writing the final input.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    def shifted(self, left: int, top: int) -> "Intrinsics":
        return Intrinsics(self.fx, self.fy, self.cx + left, self.cy + top)

    def as_dict(self) -> dict[str, float]:
        return {"fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy}


def _positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return number


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Expand an image border for UniSHARP without changing the source pixels."
    )
    parser.add_argument("--image", required=True, type=Path, help="Source RGB image.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Directory for generated files.")

    padding = parser.add_mutually_exclusive_group()
    padding.add_argument(
        "--expand-ratio",
        type=_positive_float,
        default=0.15,
        help="Padding on every side as a fraction of the original short edge (default: 0.15).",
    )
    padding.add_argument(
        "--pad-px",
        type=_positive_int,
        help="Padding in pixels on every side. Overrides --expand-ratio.",
    )

    camera = parser.add_mutually_exclusive_group(required=True)
    camera.add_argument(
        "--intrinsics",
        metavar=("FX", "FY", "CX", "CY"),
        nargs=4,
        type=float,
        help="Original image intrinsics in pixel units.",
    )
    camera.add_argument(
        "--camera-json",
        type=Path,
        help="UniSHARP-style JSON containing camera and intrinsics.",
    )
    parser.add_argument(
        "--camera",
        choices=("perspective", "pinhole"),
        default="perspective",
        help="Camera type used with --intrinsics (default: perspective).",
    )

    parser.add_argument(
        "--fill-mode",
        choices=("reflect", "edge", "black"),
        default="reflect",
        help="Dependency-free border fill for a geometry smoke test (default: reflect).",
    )
    parser.add_argument(
        "--filled-image",
        type=Path,
        help="Optional externally outpainted full-canvas image. It must match the target size.",
    )
    return parser


def read_rgb_image(path: Path) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(f"Image does not exist: {path}")
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def read_intrinsics_from_json(path: Path) -> tuple[str, Intrinsics]:
    if not path.is_file():
        raise FileNotFoundError(f"Camera JSON does not exist: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Camera JSON must contain an object at its top level.")

    camera = str(data.get("camera", "perspective")).lower()
    if camera == "pinhole":
        camera = "perspective"
    if camera != "perspective":
        raise ValueError(
            "This script currently supports perspective/pinhole cameras only. "
            f"Received camera={camera!r}; fisheye and panorama require a camera-specific expansion."
        )

    raw = data.get("intrinsics")
    if isinstance(raw, dict):
        try:
            values = [raw[key] for key in ("fx", "fy", "cx", "cy")]
        except KeyError as exc:
            raise ValueError("intrinsics object must contain fx, fy, cx, cy") from exc
    elif isinstance(raw, list) and len(raw) in (4, 9):
        values = raw[:4]
    else:
        raise ValueError(
            "Camera JSON must contain intrinsics as {fx, fy, cx, cy} or a list with 4/9 values."
        )
    try:
        intrinsics = Intrinsics(*(float(value) for value in values))
    except (TypeError, ValueError) as exc:
        raise ValueError("Camera intrinsics must be numeric.") from exc
    validate_intrinsics(intrinsics)
    return camera, intrinsics


def validate_intrinsics(intrinsics: Intrinsics) -> None:
    values = (intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Camera intrinsics must be finite numbers.")
    if intrinsics.fx <= 0 or intrinsics.fy <= 0:
        raise ValueError("fx and fy must be positive.")


def calculate_padding(width: int, height: int, args: argparse.Namespace) -> int:
    if args.pad_px is not None:
        return args.pad_px
    return max(1, int(round(min(width, height) * args.expand_ratio)))


def dependency_free_fill(image: Image.Image, pad: int, mode: str) -> Image.Image:
    if mode == "black":
        return ImageOps.expand(image, border=pad, fill=(0, 0, 0))

    array = np.asarray(image)
    if mode == "edge" or image.width == 1 or image.height == 1:
        padded = np.pad(array, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    else:
        padded = np.pad(array, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    return Image.fromarray(padded, mode="RGB")


def make_mask(width: int, height: int, pad: int) -> Image.Image:
    mask = Image.new("L", (width + 2 * pad, height + 2 * pad), color=255)
    mask.paste(0, (pad, pad, pad + width, pad + height))
    return mask


def make_output_image(
    source: Image.Image, pad: int, fill_mode: str, filled_image: Path | None
) -> Image.Image:
    target_size = (source.width + 2 * pad, source.height + 2 * pad)
    if filled_image is None:
        canvas = dependency_free_fill(source, pad, fill_mode)
    else:
        canvas = read_rgb_image(filled_image)
        if canvas.size != target_size:
            raise ValueError(
                f"--filled-image has size {canvas.size}, but expected full canvas {target_size}."
            )
    # This is intentional even for an external fill: never let a generator alter source pixels.
    canvas.paste(source, (pad, pad))
    return canvas


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    if args.expand_ratio is not None and args.expand_ratio > 1.0:
        raise ValueError("--expand-ratio must not exceed 1.0 (100% of the short edge per side).")

    source = read_rgb_image(args.image)
    if args.camera_json is not None:
        camera, original_k = read_intrinsics_from_json(args.camera_json)
    else:
        camera = "perspective" if args.camera == "pinhole" else args.camera
        original_k = Intrinsics(*(float(value) for value in args.intrinsics))
        validate_intrinsics(original_k)

    pad = calculate_padding(source.width, source.height, args)
    expanded = make_output_image(source, pad, args.fill_mode, args.filled_image)
    mask = make_mask(source.width, source.height, pad)
    updated_k = original_k.shifted(pad, pad)

    output_dir = args.out_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "outpaint_input.png"
    mask_path = output_dir / "outpaint_mask.png"
    camera_path = output_dir / "camera.json"
    metadata_path = output_dir / "metadata.json"

    expanded.save(image_path)
    mask.save(mask_path)
    save_json(camera_path, {"camera": camera, "intrinsics": updated_k.as_dict()})
    save_json(
        metadata_path,
        {
            "source_image": str(args.image.resolve()),
            "source_size": {"width": source.width, "height": source.height},
            "canvas_size": {"width": expanded.width, "height": expanded.height},
            "source_bbox_xyxy": [pad, pad, pad + source.width, pad + source.height],
            "padding": {"left": pad, "top": pad, "right": pad, "bottom": pad},
            "camera": camera,
            "original_intrinsics": original_k.as_dict(),
            "output_intrinsics": updated_k.as_dict(),
            "border_fill": (
                {"type": "external", "path": str(args.filled_image.resolve())}
                if args.filled_image is not None
                else {
                    "type": "deterministic_test_fill",
                    "mode": args.fill_mode,
                    "note": "This fill validates the geometry only; use semantic outpainting for production.",
                }
            ),
        },
    )

    print(f"Prepared image: {image_path}")
    print(f"Outpaint mask:  {mask_path}  (white=generate, black=preserve)")
    print(f"Camera JSON:    {camera_path}")
    print("UniSHARP handoff (replace <checkpoint>):")
    print(
        "  python scripts/infer_unisharp.py "
        f"--checkpoint <checkpoint> --image \"{image_path}\" --camera-json \"{camera_path}\" "
        f"--out-dir \"{output_dir / 'unisharp'}\" --save-ply"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)

from __future__ import annotations

"""Generate UniK3D pseudo z-depth files for downloaded DL3DV RGB/pose scenes."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from unisharp.utils.unik3d_adapter import infer_unik3d_pinhole, load_unik3d_model


def _scene_dirs(root: Path) -> list[tuple[str, Path, str, str]]:
    result: list[tuple[str, Path, str, str]] = []
    for bucket in sorted(path for path in root.iterdir() if path.is_dir()):
        for stub in sorted(path for path in bucket.iterdir() if path.is_dir()):
            inner = sorted(path for path in stub.iterdir() if path.is_dir())
            scene = inner[0] if inner else stub
            if (scene / "transforms.json").is_file() and (scene / "images_4").is_dir():
                result.append((f"{bucket.name}/{stub.name}", scene, bucket.name, stub.name))
    return result


def _frame_id(name: str) -> int:
    return int(Path(name).stem.split("_")[-1])


def _load_scene(scene_dir: Path) -> tuple[dict[int, Path], dict[int, torch.Tensor]]:
    meta = json.loads((scene_dir / "transforms.json").read_text(encoding="utf-8"))
    orig_w, orig_h = int(meta["w"]), int(meta["h"])
    fx, fy, cx, cy = (float(meta[key]) for key in ("fl_x", "fl_y", "cx", "cy"))
    image_paths = {_frame_id(path.name): path for path in (scene_dir / "images_4").glob("*.png")}
    intrinsics: dict[int, torch.Tensor] = {}
    for frame in meta.get("frames", []):
        fid = _frame_id(Path(str(frame.get("file_path", ""))).name)
        image = image_paths.get(fid)
        if image is None:
            continue
        with Image.open(image) as im:
            cur_w, cur_h = im.size
        sx, sy = float(cur_w) / orig_w, float(cur_h) / orig_h
        k = torch.eye(3, dtype=torch.float32)
        k[0, 0], k[1, 1] = fx * sx, fy * sy
        k[0, 2], k[1, 2] = (cx + 0.5) * sx - 0.5, (cy + 0.5) * sy - 0.5
        intrinsics[fid] = k
    return image_paths, intrinsics


def _rgb(path: Path) -> torch.Tensor:
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _distance_to_z(distance: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Convert radial UniK3D distance to camera-space +Z depth."""
    _, _, h, w = distance.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=distance.device, dtype=distance.dtype),
        torch.arange(w, device=distance.device, dtype=distance.dtype),
        indexing="ij",
    )
    kk = k.to(device=distance.device, dtype=distance.dtype)
    x = (xx - kk[0, 2]) / kk[0, 0]
    y = (yy - kk[1, 2]) / kk[1, 1]
    ray_z = torch.rsqrt(x.square() + y.square() + 1.0)
    return distance * ray_z[None, None]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--depth-root", type=Path, required=True)
    parser.add_argument("--backbone", choices=("vits", "vitb", "vitl"), default="vits")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--max-frames-per-scene", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--manifest-path", type=Path, required=True, help="Writes the exact DL3DVDataset scene-spec manifest.")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if str(args.device).startswith("npu"):
        try:
            import torch_npu  # noqa: F401
        except ImportError as exc:
            raise SystemExit("--device npu requires the matched torch_npu package.") from exc
    device = torch.device(args.device)
    scenes = _scene_dirs(Path(args.rgb_root))
    if int(args.max_scenes) > 0:
        scenes = scenes[: int(args.max_scenes)]
    if not scenes:
        raise SystemExit("No DL3DV scenes with transforms.json and images_4 were found.")
    model = load_unik3d_model(backbone=args.backbone, pretrained=True, device=device)
    specs: list[str] = []
    saved = 0
    for scene_name, scene_dir, bucket, stub in scenes:
        try:
            images, intrinsics = _load_scene(scene_dir)
        except Exception as exc:
            print(f"skip {scene_name}: {exc}")
            continue
        out_dir = Path(args.depth_root) / bucket / stub / "exports" / "mini_npz" / "per_image"
        valid = sorted(set(images).intersection(intrinsics))
        if int(args.max_frames_per_scene) > 0:
            valid = valid[: int(args.max_frames_per_scene)]
        scene_saved = 0
        for fid in valid:
            target = out_dir / f"frame_{fid:05d}.npz"
            if target.exists() and not args.overwrite:
                scene_saved += 1
                continue
            rgb = _rgb(images[fid]).unsqueeze(0).to(device=device, non_blocking=True)
            with torch.no_grad():
                result = infer_unik3d_pinhole(model, rgb_u8=rgb, intrinsics=intrinsics[fid].unsqueeze(0).to(device))
            distance = result.get("distance")
            if not torch.is_tensor(distance) or tuple(distance.shape[-2:]) != tuple(rgb.shape[-2:]):
                raise RuntimeError(f"UniK3D returned no full-resolution distance for {images[fid]}")
            zdepth = _distance_to_z(distance.to(torch.float32), intrinsics[fid]).squeeze().clamp(min=0.0).cpu().numpy()
            out_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, depth=zdepth.astype(np.float32))
            scene_saved += 1
            saved += 1
        if scene_saved >= 2:
            specs.append(f"{scene_name}|{scene_dir.resolve()}|{out_dir.resolve()}")
            print(f"prepared {scene_name}: {scene_saved} frame(s)")
    if not specs:
        raise SystemExit("No DL3DV scene produced at least two usable depths.")
    manifest = Path(args.manifest_path)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(specs) + "\n", encoding="utf-8")
    print(f"Saved {saved} depth files; manifest: {manifest}")


if __name__ == "__main__":
    main()

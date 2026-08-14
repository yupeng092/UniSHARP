
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import IterableDataset

from unisharp import DEFAULT_MAX_DEPTH_M


LOGGER = logging.getLogger(__name__)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
DEPTH_DIR_NAMES = ("depth", "depths", "distance", "distances", "depth_maps")
MASK_DIR_NAMES = ("masks", "mask")
DEPTH_MAX_M = DEFAULT_MAX_DEPTH_M


def _qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    q = np.asarray(qvec, dtype=np.float64)
    return np.array(
        [
            [1 - 2 * q[2] ** 2 - 2 * q[3] ** 2, 2 * q[1] * q[2] - 2 * q[0] * q[3], 2 * q[3] * q[1] + 2 * q[0] * q[2]],
            [2 * q[1] * q[2] + 2 * q[0] * q[3], 1 - 2 * q[1] ** 2 - 2 * q[3] ** 2, 2 * q[2] * q[3] - 2 * q[0] * q[1]],
            [2 * q[3] * q[1] - 2 * q[0] * q[2], 2 * q[2] * q[3] + 2 * q[0] * q[1], 1 - 2 * q[1] ** 2 - 2 * q[2] ** 2],
        ],
        dtype=np.float64,
    )


def _read_colmap_w2c(images_txt: Path) -> dict[str, torch.Tensor]:
    poses: dict[str, torch.Tensor] = {}
    if not images_txt.exists():
        return poses
    with images_txt.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            try:
                qvec = np.asarray([float(x) for x in parts[1:5]], dtype=np.float64)
                tvec = np.asarray([float(x) for x in parts[5:8]], dtype=np.float64)
                image_name = parts[9]
            except Exception:
                continue
            w2c = np.eye(4, dtype=np.float32)
            w2c[:3, :3] = _qvec_to_rotmat(qvec).astype(np.float32)
            w2c[:3, 3] = tvec.astype(np.float32)
            poses[Path(image_name).name] = torch.from_numpy(w2c)
    return poses


def _opencv_fisheye_to_fisheye624_params(meta: dict[str, object]) -> torch.Tensor:
    if str(meta.get("camera_model", "")) != "OPENCV_FISHEYE":
        raise RuntimeError(f"Unsupported ScanNet++ camera_model={meta.get('camera_model')!r}; expected OPENCV_FISHEYE.")
    return torch.tensor(
        [
            float(meta["fl_x"]),
            float(meta["fl_y"]),
            float(meta["cx"]),
            float(meta["cy"]),
            float(meta.get("k1", 0.0)),
            float(meta.get("k2", 0.0)),
            float(meta.get("k3", 0.0)),
            float(meta.get("k4", 0.0)),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=torch.float32,
    )


def _camera_hw_from_meta(meta: dict[str, object]) -> tuple[int, int] | None:
    h = meta.get("h", meta.get("height", None))
    w = meta.get("w", meta.get("width", None))
    if h is None or w is None:
        return None
    try:
        h_i, w_i = int(h), int(w)
    except Exception:
        return None
    return (h_i, w_i) if h_i > 0 and w_i > 0 else None


def _scale_fisheye624_params(
    params: torch.Tensor,
    *,
    src_hw: tuple[int, int],
    dst_hw: tuple[int, int],
) -> torch.Tensor:
    if tuple(int(x) for x in src_hw) == tuple(int(x) for x in dst_hw):
        return params.clone()
    src_h, src_w = int(src_hw[0]), int(src_hw[1])
    dst_h, dst_w = int(dst_hw[0]), int(dst_hw[1])
    sx = float(dst_w) / float(max(src_w, 1))
    sy = float(dst_h) / float(max(src_h, 1))
    out = params.clone()
    out[..., 0] *= sx
    out[..., 1] *= sy
    out[..., 2] = (out[..., 2] + 0.5) * sx - 0.5
    out[..., 3] = (out[..., 3] + 0.5) * sy - 0.5
    return out


def _stack_batch(batch: list["ScannetppFisheyePairSample"]) -> "ScannetppFisheyePairSample":
    return ScannetppFisheyePairSample(
        src_rgb_u8=torch.stack([b.src_rgb_u8 for b in batch], dim=0),
        tgt_rgb_u8=torch.stack([b.tgt_rgb_u8 for b in batch], dim=0),
        src_depth_m=torch.stack([b.src_depth_m for b in batch], dim=0),
        tgt_depth_m=torch.stack([b.tgt_depth_m for b in batch], dim=0),
        src_valid_mask=torch.stack([b.src_valid_mask for b in batch], dim=0),
        tgt_valid_mask=torch.stack([b.tgt_valid_mask for b in batch], dim=0),
        src_w2c=torch.stack([b.src_w2c for b in batch], dim=0),
        tgt_w2c=torch.stack([b.tgt_w2c for b in batch], dim=0),
        src_camera_params=torch.stack([b.src_camera_params for b in batch], dim=0),
        tgt_camera_params=torch.stack([b.tgt_camera_params for b in batch], dim=0),
        src_idx=[b.src_idx for b in batch],  # type: ignore[arg-type]
        tgt_idx=[b.tgt_idx for b in batch],  # type: ignore[arg-type]
        scene=[b.scene for b in batch],  # type: ignore[arg-type]
        camera_model="fisheye624",
    )


@dataclass(frozen=True)
class ScannetppFisheyePairSample:
    src_rgb_u8: torch.Tensor
    tgt_rgb_u8: torch.Tensor
    src_depth_m: torch.Tensor
    tgt_depth_m: torch.Tensor
    src_valid_mask: torch.Tensor
    tgt_valid_mask: torch.Tensor
    src_w2c: torch.Tensor
    tgt_w2c: torch.Tensor
    src_camera_params: torch.Tensor
    tgt_camera_params: torch.Tensor
    src_idx: int
    tgt_idx: int
    scene: str
    camera_model: str = "fisheye624"


def scannetpp_fisheye_passthrough(batch: ScannetppFisheyePairSample) -> ScannetppFisheyePairSample:
    return batch


class ScannetppFisheyeDataset(IterableDataset):

    def __init__(
        self,
        root: Path,
        scene_list_file: Path | None = None,
        min_frame_gap: int = 1,
        max_frame_gap: int = 10,
        pair_max_translation_m: float = 0.5,
        shuffle_scene: bool = True,
        shuffle_frame: bool = True,
        skip_bad: bool = True,
        ddp_rank: int = 0,
        ddp_world_size: int = 1,
        batch_size_hint: int = 1,
        depth_max_m: float = DEFAULT_MAX_DEPTH_M,
        far_depth_invalid_m: float = 30.0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.min_frame_gap = int(min_frame_gap)
        self.max_frame_gap = int(max_frame_gap)
        self.pair_max_translation_m = float(pair_max_translation_m)
        self.shuffle_scene = bool(shuffle_scene)
        self.shuffle_frame = bool(shuffle_frame)
        self.skip_bad = bool(skip_bad)
        self.ddp_rank = int(ddp_rank)
        self.ddp_world_size = int(ddp_world_size)
        self.batch_size_hint = int(max(1, batch_size_hint))
        self.depth_max_m = float(depth_max_m)
        self.far_depth_invalid_m = float(far_depth_invalid_m)
        self.seed = int(seed)
        self.epoch = 0
        self.scene_specs = self._load_scene_specs(scene_list_file)
        if not self.scene_specs:
            raise RuntimeError(f"No ScanNet++ fisheye scenes found under {self.root}")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _load_scene_specs(self, scene_list_file: Path | None) -> list[tuple[str, Path]]:
        specs: list[tuple[str, Path]] = []
        if scene_list_file is not None and Path(scene_list_file).exists():
            for raw in Path(scene_list_file).read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) == 1:
                    scene_dir = Path(parts[0])
                    scene_id = scene_dir.name
                else:
                    scene_id = parts[0]
                    scene_dir = Path(parts[1])
                if not scene_dir.is_absolute():
                    scene_dir = self.root / scene_dir
                specs.append((scene_id, scene_dir))
            return specs
        for transforms in sorted(self.root.glob("*/nerfstudio/transforms.json")):
            specs.append((transforms.parent.parent.name, transforms.parent.parent))
        for transforms in sorted(self.root.glob("*/*/nerfstudio/transforms.json")):
            specs.append((f"{transforms.parent.parent.parent.name}/{transforms.parent.parent.name}", transforms.parent.parent))
        return specs

    @staticmethod
    def _load_rgb(path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            arr = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    @staticmethod
    def _load_mask(path: Path, image_hw: tuple[int, int]) -> torch.Tensor | None:
        if not path.exists():
            return None
        with Image.open(path) as image:
            arr = np.asarray(image.convert("L"), dtype=np.uint8).copy()
        mask = torch.from_numpy(arr).unsqueeze(0).to(torch.float32) / 255.0
        if tuple(mask.shape[-2:]) != tuple(image_hw):
            mask = torch.nn.functional.interpolate(mask.unsqueeze(0), size=image_hw, mode="nearest").squeeze(0)
        return (mask > 0.5).to(torch.float32)

    def _load_depth_map(self, path: Path) -> tuple[torch.Tensor, str]:
        depth_kind = "distance"
        if path.suffix.lower() == ".npz":
            payload = np.load(path, allow_pickle=False)
            for key in ("distance_m", "depth_m", "distance", "depth"):
                if key in payload:
                    arr = payload[key]
                    if key in {"distance_m", "distance"}:
                        depth_kind = "distance"
                    elif "depth_kind" in payload:
                        depth_kind = str(np.asarray(payload["depth_kind"]).item()).strip().lower()
                    break
            else:
                raise RuntimeError(f"Unsupported ScanNet++ depth payload keys at {path}")
        else:
            arr = np.load(path)
        depth = torch.from_numpy(np.asarray(arr, dtype=np.float32).copy())
        if depth.ndim == 3 and depth.shape[0] == 1:
            depth = depth[0]
        if depth.ndim != 2:
            raise RuntimeError(f"Expected 2D fisheye depth at {path}, got shape={tuple(depth.shape)}")
        depth = depth.unsqueeze(0)
        valid = torch.isfinite(depth) & (depth > 0.0)
        if self.far_depth_invalid_m > 0.0:
            valid = valid & (depth <= self.far_depth_invalid_m)
        depth = torch.where(valid, depth, torch.zeros_like(depth))
        if depth_kind in {"radial", "radius", "dist"}:
            depth_kind = "distance"
        if depth_kind not in {"distance", "z"}:
            raise RuntimeError(f"Unsupported fisheye depth_kind={depth_kind!r} at {path}")
        return depth.clamp(min=0.0, max=self.depth_max_m), depth_kind

    @staticmethod
    def _fisheye_z_depth_to_distance(z_depth: torch.Tensor, camera_params: torch.Tensor) -> torch.Tensor:
        from unisharp.utils.fisheye_geer import build_fisheye624_raymap

        h, w = int(z_depth.shape[-2]), int(z_depth.shape[-1])
        rays = build_fisheye624_raymap(
            camera_params.unsqueeze(0),
            image_h=h,
            image_w=w,
            device=z_depth.device,
            dtype=torch.float32,
        )
        ray_z = rays[:, 2:3].squeeze(0).to(device=z_depth.device, dtype=z_depth.dtype)
        valid = torch.isfinite(z_depth) & (z_depth > 0.0) & torch.isfinite(ray_z) & (ray_z > 1e-4)
        distance = z_depth / ray_z.clamp(min=1e-4)
        return torch.where(valid, distance, torch.zeros_like(z_depth))

    def _resolve_image_path(self, scene_dir: Path, image_name: str) -> Path | None:
        rel = Path(image_name)
        candidates = [
            scene_dir / rel,
            scene_dir / "images" / rel.name,
            scene_dir / "resized_images" / rel.name,
            scene_dir / "dslr" / rel,
            scene_dir / "dslr" / "images" / rel.name,
            scene_dir / "dslr" / "resized_images" / rel.name,
        ]
        for path in candidates:
            if path.exists() and path.suffix in IMAGE_SUFFIXES:
                return path
        return None

    def _resolve_depth_path(self, scene_dir: Path, image_name: str) -> Path | None:
        stem = Path(image_name).stem
        names = [stem, Path(image_name).name]
        bases = [scene_dir, scene_dir / "dslr"]
        for base in bases:
            for depth_dir_name in DEPTH_DIR_NAMES:
                depth_dir = base / depth_dir_name
                for name in names:
                    for suffix in (".npz", ".npy"):
                        path = depth_dir / f"{name}{suffix}"
                        if path.exists():
                            return path
        return None

    def _resolve_mask_path(self, scene_dir: Path, image_name: str, mask_name: str | None) -> Path | None:
        names = []
        if mask_name:
            names.append(Path(mask_name).name)
        names.append(f"{Path(image_name).stem}.png")
        bases = [scene_dir, scene_dir / "dslr"]
        for base in bases:
            for name in names:
                direct = base / name
                if direct.exists():
                    return direct
                for mask_dir_name in MASK_DIR_NAMES:
                    path = base / mask_dir_name / name
                    if path.exists():
                        return path
        return None

    def _load_scene_frames(self, scene_id: str, scene_dir: Path) -> tuple[torch.Tensor, list[dict[str, object]]]:
        transforms_path = scene_dir / "nerfstudio" / "transforms.json"
        if not transforms_path.exists():
            transforms_path = scene_dir / "dslr" / "nerfstudio" / "transforms.json"
        meta = json.loads(transforms_path.read_text(encoding="utf-8"))
        camera_params = _opencv_fisheye_to_fisheye624_params(meta)
        camera_hw = _camera_hw_from_meta(meta)
        w2c_by_name = _read_colmap_w2c(scene_dir / "colmap" / "images.txt")
        if not w2c_by_name:
            w2c_by_name = _read_colmap_w2c(scene_dir / "dslr" / "colmap" / "images.txt")

        raw_frames = list(meta.get("frames", [])) + list(meta.get("test_frames", []))
        frames: list[dict[str, object]] = []
        for frame in raw_frames:
            image_name = Path(str(frame.get("file_path", ""))).name
            if not image_name:
                continue
            if self.skip_bad and bool(frame.get("is_bad", False)):
                continue
            image_path = self._resolve_image_path(scene_dir, image_name)
            depth_path = self._resolve_depth_path(scene_dir, image_name)
            if image_path is None or depth_path is None:
                continue
            w2c = w2c_by_name.get(image_name)
            if w2c is None and frame.get("transform_matrix") is not None:
                c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
                w2c = torch.linalg.inv(c2w)
            if w2c is None:
                continue
            center = torch.linalg.inv(w2c)[:3, 3]
            frames.append(
                {
                    "image_name": image_name,
                    "image_path": image_path,
                    "depth_path": depth_path,
                    "mask_path": self._resolve_mask_path(scene_dir, image_name, frame.get("mask_path")),
                    "w2c": w2c.to(torch.float32),
                    "center": center.to(torch.float32),
                    "idx": len(frames),
                    "scene": scene_id,
                    "camera_hw": _camera_hw_from_meta(frame) or camera_hw,
                }
            )
        return camera_params, sorted(frames, key=lambda x: str(x["image_name"]))

    def _load_frame_tensor(self, frame: dict[str, object], camera_params: torch.Tensor) -> dict[str, torch.Tensor]:
        rgb = self._load_rgb(frame["image_path"])  # type: ignore[arg-type]
        rgb_hw = (int(rgb.shape[-2]), int(rgb.shape[-1]))
        camera_hw = frame.get("camera_hw", None)
        params = camera_params.clone()
        if isinstance(camera_hw, tuple):
            params = _scale_fisheye624_params(params, src_hw=camera_hw, dst_hw=rgb_hw)
        depth, depth_kind = self._load_depth_map(frame["depth_path"])  # type: ignore[arg-type]
        if tuple(depth.shape[-2:]) != tuple(rgb.shape[-2:]):
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(0),
                size=(int(rgb.shape[-2]), int(rgb.shape[-1])),
                mode="nearest",
            ).squeeze(0)
        if depth_kind == "z":
            depth = self._fisheye_z_depth_to_distance(depth, params)
        valid = (torch.isfinite(depth) & (depth > 0.0)).to(torch.float32)
        mask_path = frame.get("mask_path", None)
        if isinstance(mask_path, Path):
            mask = self._load_mask(mask_path, (int(rgb.shape[-2]), int(rgb.shape[-1])))
            if mask is not None:
                valid = valid * mask
        else:
            valid = valid * (rgb.to(torch.float32).sum(dim=0, keepdim=True) > 1.0).to(torch.float32)
        return {
            "rgb_u8": rgb,
            "depth_m": depth.clamp(min=0.0, max=self.depth_max_m),
            "valid_mask": valid,
            "camera_params": params,
        }

    def _iter_scene_pairs(self, scene_id: str, scene_dir: Path, rng: random.Random):
        try:
            camera_params, frames = self._load_scene_frames(scene_id, scene_dir)
        except Exception as exc:
            LOGGER.debug("Skip ScanNet++ scene %s: %s", str(scene_id), str(exc))
            return
        if len(frames) < 2:
            return
        loaded: dict[int, dict[str, torch.Tensor]] = {}

        def get_loaded(pos: int) -> dict[str, torch.Tensor]:
            if pos not in loaded:
                loaded[pos] = self._load_frame_tensor(frames[pos], camera_params)
            return loaded[pos]

        order = list(range(len(frames)))
        if self.shuffle_frame:
            rng.shuffle(order)
        for src_pos in order:
            src_item = frames[src_pos]
            src_center = src_item["center"]
            assert torch.is_tensor(src_center)
            candidates: list[int] = []
            for tgt_pos in range(max(0, src_pos - self.max_frame_gap), min(len(frames), src_pos + self.max_frame_gap + 1)):
                if tgt_pos == src_pos:
                    continue
                gap = abs(tgt_pos - src_pos)
                if gap < self.min_frame_gap:
                    continue
                tgt_center = frames[tgt_pos]["center"]
                assert torch.is_tensor(tgt_center)
                if float(torch.norm(tgt_center - src_center, p=2).item()) > self.pair_max_translation_m:
                    continue
                candidates.append(tgt_pos)
            if not candidates:
                continue
            tgt_pos = rng.choice(candidates)
            try:
                src_loaded = get_loaded(src_pos)
                tgt_loaded = get_loaded(tgt_pos)
            except Exception:
                continue
            yield ScannetppFisheyePairSample(
                src_rgb_u8=src_loaded["rgb_u8"],
                tgt_rgb_u8=tgt_loaded["rgb_u8"],
                src_depth_m=src_loaded["depth_m"],
                tgt_depth_m=tgt_loaded["depth_m"],
                src_valid_mask=src_loaded["valid_mask"],
                tgt_valid_mask=tgt_loaded["valid_mask"],
                src_w2c=src_item["w2c"],  # type: ignore[arg-type]
                tgt_w2c=frames[tgt_pos]["w2c"],  # type: ignore[arg-type]
                src_camera_params=src_loaded["camera_params"],
                tgt_camera_params=tgt_loaded["camera_params"],
                src_idx=int(src_item["idx"]),
                tgt_idx=int(frames[tgt_pos]["idx"]),
                scene=str(scene_id),
            )

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        worker_id = 0 if worker is None else int(worker.id)
        num_workers = 1 if worker is None else int(worker.num_workers)
        rng = random.Random(self.seed + 1009 * self.epoch + 97 * self.ddp_rank + 17 * worker_id)
        specs = list(self.scene_specs)
        if self.shuffle_scene:
            rng.shuffle(specs)
        specs = specs[self.ddp_rank :: max(self.ddp_world_size, 1)]
        specs = specs[worker_id :: num_workers]
        pending: dict[tuple[int, int], list[ScannetppFisheyePairSample]] = {}
        for scene_id, scene_dir in specs:
            for sample in self._iter_scene_pairs(scene_id, scene_dir, rng):
                hw = (int(sample.src_rgb_u8.shape[-2]), int(sample.src_rgb_u8.shape[-1]))
                bucket = pending.setdefault(hw, [])
                bucket.append(sample)
                while len(bucket) >= self.batch_size_hint:
                    packed = bucket[: self.batch_size_hint]
                    del bucket[: self.batch_size_hint]
                    yield _stack_batch(packed)

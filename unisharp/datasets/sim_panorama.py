
from __future__ import annotations

import csv
from dataclasses import dataclass
import os
from pathlib import Path
import random
import re

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import IterableDataset

from unisharp.datasets.panogs import PanOGSSample
from unisharp import DEFAULT_MAX_DEPTH_M

try:
    import h5py
except ImportError:
    h5py = None


_NUM_RE = re.compile(r"(\d+)(?!.*\d)")
_SIM_CACHE_VERSION = 6


def _default_dataset_manifest_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    parent_path = repo_root.parent / "dataset_manifests"
    if parent_path.exists():
        return parent_path
    return repo_root / "dataset_manifests"


def _frame_index_from_name(name: str) -> int | None:
    match = _NUM_RE.search(Path(name).stem)
    if match is None:
        return None
    return int(match.group(1))


def _sim_csv_xyz_to_training_position(x: float, y: float, z: float) -> torch.Tensor:
    return torch.tensor([float(y), -float(z), float(x)], dtype=torch.float32)


class _EquirecToCube:
    def __init__(self, equ_h: int, equ_w: int, face_w: int) -> None:
        self.equ_h = int(equ_h)
        self.equ_w = int(equ_w)
        self.face_w = int(face_w)
        self.grid = self._build_grid()
        rng = torch.linspace(-0.5, 0.5, steps=self.face_w, dtype=torch.float32)
        xx, yy = torch.meshgrid(rng, -rng, indexing="xy")
        self.ray_z = (1.0 / torch.sqrt((2.0 * xx) ** 2 + (2.0 * yy) ** 2 + 1.0)).contiguous()

    def _build_grid(self) -> torch.Tensor:
        face_w = self.face_w
        rng = torch.linspace(-0.5, 0.5, steps=face_w, dtype=torch.float32)
        grid = torch.stack(torch.meshgrid(rng, -rng, indexing="xy"), dim=-1)
        xyz = torch.zeros((6, face_w, face_w, 3), dtype=torch.float32)

        xyz[0, :, :, 0] = grid[:, :, 0]
        xyz[0, :, :, 1] = grid[:, :, 1]
        xyz[0, :, :, 2] = 0.5

        xyz[1, :, :, 2] = torch.flip(grid[:, :, 0], dims=[1])
        xyz[1, :, :, 1] = torch.flip(grid[:, :, 1], dims=[1])
        xyz[1, :, :, 0] = 0.5

        xyz[2, :, :, 0] = torch.flip(grid[:, :, 0], dims=[1])
        xyz[2, :, :, 1] = torch.flip(grid[:, :, 1], dims=[1])
        xyz[2, :, :, 2] = -0.5

        xyz[3, :, :, 2] = grid[:, :, 0]
        xyz[3, :, :, 1] = grid[:, :, 1]
        xyz[3, :, :, 0] = -0.5

        xyz[4, :, :, 0] = torch.flip(grid[:, :, 0], dims=[0])
        xyz[4, :, :, 2] = torch.flip(grid[:, :, 1], dims=[0])
        xyz[4, :, :, 1] = 0.5

        xyz[5, :, :, 0] = grid[:, :, 0]
        xyz[5, :, :, 2] = grid[:, :, 1]
        xyz[5, :, :, 1] = -0.5

        xyz = xyz[[4, 2, 3, 0, 1, 5]]
        x = xyz[..., 0]
        y = xyz[..., 1]
        z = xyz[..., 2]
        lon = torch.atan2(x, z)
        c = torch.sqrt(x * x + z * z).clamp(min=1e-8)
        lat = torch.atan2(y, c)
        grid_x = lon / np.pi
        grid_y = (-2.0 * lat / np.pi).clamp(min=-1.0, max=1.0)
        return torch.stack([grid_x, grid_y], dim=-1).contiguous()

    def run_depth(self, depth_1hw: torch.Tensor) -> torch.Tensor:
        depth = depth_1hw.unsqueeze(0).to(torch.float32)
        if tuple(depth.shape[-2:]) != (self.equ_h, self.equ_w):
            depth = F.interpolate(depth, size=(self.equ_h, self.equ_w), mode="nearest")
        depth_faces = F.grid_sample(
            depth.expand(6, -1, -1, -1),
            self.grid,
            mode="nearest",
            padding_mode="border",
            align_corners=True,
        )
        depth_faces = depth_faces[:, 0] * self.ray_z.to(depth_faces.device, depth_faces.dtype)
        return depth_faces.unsqueeze(-1).to(torch.float32).cpu()

    def run(self, rgb_chw: torch.Tensor, depth_1hw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rgb = rgb_chw.unsqueeze(0).to(torch.float32) / 255.0
        if tuple(rgb.shape[-2:]) != (self.equ_h, self.equ_w):
            rgb = F.interpolate(rgb, size=(self.equ_h, self.equ_w), mode="bilinear", align_corners=True)
        rgb_faces = F.grid_sample(
            rgb.expand(6, -1, -1, -1),
            self.grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        cube_rgb = (rgb_faces.permute(0, 2, 3, 1).clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
        cube_depth = self.run_depth(depth_1hw)
        return cube_rgb.cpu(), cube_depth

    def run_rgb(self, rgb_chw: torch.Tensor) -> torch.Tensor:
        rgb = rgb_chw.unsqueeze(0).to(torch.float32) / 255.0
        if tuple(rgb.shape[-2:]) != (self.equ_h, self.equ_w):
            rgb = F.interpolate(rgb, size=(self.equ_h, self.equ_w), mode="bilinear", align_corners=True)
        rgb_faces = F.grid_sample(
            rgb.expand(6, -1, -1, -1),
            self.grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return (rgb_faces.permute(0, 2, 3, 1).clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu()


@dataclass(frozen=True)
class _SimFrame:
    frame_idx: int
    rgb_path: Path
    depth_path: Path
    position_xyz: torch.Tensor


class SimPanoramaDataset(IterableDataset):
    def __init__(
        self,
        root: Path,
        pose_root: Path,
        scene_names: list[str] | None = None,
        scene_list_file: Path | None = None,
        position_scale: float = 0.01,
        max_index_gap: int = 10,
        pair_max_translation_m: float = 0.5,
        pair_min_depth_overlap: float = 0.6,
        pair_overlap_margin: float = 1.05,
        pairs_per_chunk: int = 15,
        chunk_size: int = 30,
        shuffle_scene: bool = True,
        ddp_rank: int = 0,
        ddp_world_size: int = 1,
        depth_max_m: float = DEFAULT_MAX_DEPTH_M,
        far_depth_invalid_m: float = 30.0,
        far_depth_invalid_max_frac: float = 1.0,
        max_long_edge: int = 0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.pose_root = Path(pose_root)
        self.scene_list_file = Path(scene_list_file) if scene_list_file is not None else None
        requested_scene_names = [str(name).strip() for name in (scene_names or []) if str(name).strip()]
        if self.scene_list_file is not None:
            if not self.scene_list_file.exists():
                raise FileNotFoundError(self.scene_list_file)
            manifest_scene_names = [
                line.strip()
                for line in self.scene_list_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if requested_scene_names:
                requested = set(requested_scene_names)
                self.scene_names = [name for name in manifest_scene_names if name in requested]
            else:
                self.scene_names = manifest_scene_names
        else:
            self.scene_names = requested_scene_names
        if not self.scene_names:
            raise ValueError("SimPanoramaDataset requires scene_names or scene_list_file.")
        self.position_scale = float(position_scale)
        self.max_index_gap = int(max_index_gap)
        self.pair_max_translation_m = float(pair_max_translation_m)
        self.pair_min_depth_overlap = float(pair_min_depth_overlap)
        self.pair_overlap_margin = float(pair_overlap_margin)
        self.pairs_per_chunk = int(pairs_per_chunk)
        self.chunk_size = int(chunk_size)
        self.shuffle_scene = bool(shuffle_scene)
        self.ddp_rank = int(ddp_rank)
        self.ddp_world_size = int(ddp_world_size)
        self.seed = int(seed)
        self.depth_max_m = float(depth_max_m)
        self.far_depth_invalid_m = float(far_depth_invalid_m)
        self.far_depth_invalid_max_frac = float(far_depth_invalid_max_frac)
        self.max_long_edge = max(int(max_long_edge), 0)
        self.epoch = 0
        self.cache_dir = _default_dataset_manifest_dir() / "sim_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._scene_frames_cache: dict[str, list[_SimFrame]] = {}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @staticmethod
    def _is_depth_path(path: Path) -> bool:
        tokens = [part.lower() for part in path.parts]
        name = path.name.lower()
        return ("depth" in name) or any("depth" in token for token in tokens)

    @staticmethod
    def _is_image_path(path: Path) -> bool:
        return path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")

    @staticmethod
    def _load_rgb(path: Path) -> torch.Tensor:
        with Image.open(path) as img:
            img = img.convert("RGB")
            arr = np.asarray(img, dtype=np.uint8).copy()
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    @staticmethod
    def _image_hw(path: Path) -> tuple[int, int]:
        with Image.open(path) as img:
            width, height = img.size
        return int(height), int(width)

    def _load_depth(self, path: Path) -> torch.Tensor:
        suffix = path.suffix.lower()
        if suffix == ".npy":
            dep = np.load(path)
        elif suffix == ".npz":
            payload = np.load(path)
            key = "depth" if "depth" in payload.files else payload.files[0]
            dep = payload[key]
        elif suffix in (".h5", ".hdf5"):
            if h5py is None:
                raise ImportError("h5py is required to read sim .h5 depth files but is not installed.")
            with h5py.File(path, "r") as f:
                keys = list(f.keys())
                if not keys:
                    raise RuntimeError(f"Empty sim depth file: {path}")
                dep = f[keys[0]][()]
        else:
            with Image.open(path) as img:
                dep = np.asarray(img)
        dep = dep.astype(np.float32)
        if dep.ndim == 3:
            dep = dep[..., 0]
        dep[~np.isfinite(dep)] = 0.0
        if self.far_depth_invalid_m > 0.0:
            far = dep > self.far_depth_invalid_m
            if 0.0 < float(far.mean()) <= self.far_depth_invalid_max_frac:
                dep[far] = 0.0
        dep = np.clip(dep, a_min=0.0, a_max=self.depth_max_m)
        return torch.from_numpy(dep).unsqueeze(0)

    def _resize_erp_if_needed(self, rgb: torch.Tensor, depth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.max_long_edge <= 0:
            return rgb, depth
        h = int(rgb.shape[-2])
        w = int(rgb.shape[-1])
        long_edge = max(h, w)
        if long_edge <= self.max_long_edge:
            return rgb, depth
        scale = float(self.max_long_edge) / float(long_edge)
        new_h = max(2, int(round(float(h) * scale)))
        new_w = max(2, int(round(float(w) * scale)))
        rgb_f = rgb.unsqueeze(0).to(dtype=torch.float32)
        rgb_resized = F.interpolate(rgb_f, size=(new_h, new_w), mode="bilinear", align_corners=False)
        rgb_out = rgb_resized[0].round().clamp(0.0, 255.0).to(dtype=torch.uint8).contiguous()
        depth_f = depth.unsqueeze(0).to(dtype=torch.float32)
        depth_out = F.interpolate(depth_f, size=(new_h, new_w), mode="nearest")[0].contiguous()
        return rgb_out, depth_out

    def _pose_csv_for_scene(self, scene_name: str) -> Path:
        direct = self.pose_root / f"{scene_name}.csv"
        if direct.exists():
            return direct
        matches = sorted(self.pose_root.glob(f"*{scene_name}*.csv"))
        if matches:
            return matches[0]
        raise FileNotFoundError(f"No pose csv found for sim scene={scene_name} under {self.pose_root}")

    def _parse_pose_csv(self, csv_path: Path) -> list[tuple[int, torch.Tensor]]:
        with csv_path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise RuntimeError(f"Empty sim pose csv: {csv_path}")
        poses: list[tuple[int, torch.Tensor]] = []
        for row_idx, row in enumerate(rows):
            lower = {str(k).strip().lower(): v for k, v in row.items()}
            frame_val = None
            for key in ("frame", "frame_idx", "idx", "index", "id", "image", "filename", "name"):
                if key in lower and str(lower[key]).strip():
                    frame_val = _frame_index_from_name(str(lower[key]))
                    if frame_val is None:
                        try:
                            frame_val = int(float(str(lower[key]).strip()))
                        except Exception:
                            frame_val = None
                    break
            x = next((lower[k] for k in lower if k in ("x", "tx", "pos_x", "world_x")), None)
            y = next((lower[k] for k in lower if k in ("y", "ty", "pos_y", "world_y")), None)
            z = next((lower[k] for k in lower if k in ("z", "tz", "pos_z", "world_z")), None)
            if x is None or y is None or z is None:
                numeric_vals = []
                for val in row.values():
                    try:
                        numeric_vals.append(float(str(val).strip()))
                    except Exception:
                        continue
                if len(numeric_vals) < 3:
                    raise ValueError(f"Failed to parse xyz from sim csv row: {row}")
                x, y, z = numeric_vals[:3]
            pos = _sim_csv_xyz_to_training_position(float(x), float(y), float(z)) * self.position_scale
            poses.append((int(frame_val if frame_val is not None else row_idx), pos))
        return poses

    def _scan_scene_frames(self, scene_name: str) -> list[_SimFrame]:
        scene_dir = self.root / scene_name
        if not scene_dir.exists():
            raise FileNotFoundError(scene_dir)
        all_files = [p for p in scene_dir.rglob("*") if p.is_file()]
        image_map: dict[int, Path] = {}
        depth_map: dict[int, Path] = {}
        for path in all_files:
            idx = _frame_index_from_name(path.name)
            if idx is None:
                continue
            if self._is_depth_path(path) and path.suffix.lower() in (".png", ".npy", ".npz", ".exr", ".h5", ".hdf5"):
                depth_map.setdefault(idx, path)
            elif self._is_image_path(path):
                image_map.setdefault(idx, path)
        pose_entries = self._parse_pose_csv(self._pose_csv_for_scene(scene_name))
        frames: list[_SimFrame] = []
        for frame_idx, pos in pose_entries:
            rgb_path = image_map.get(int(frame_idx))
            depth_path = depth_map.get(int(frame_idx))
            if rgb_path is None or depth_path is None:
                continue
            frames.append(_SimFrame(frame_idx=int(frame_idx), rgb_path=rgb_path, depth_path=depth_path, position_xyz=pos))
        return frames

    @staticmethod
    def _atomic_torch_save(path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)

    def _scene_index_cache_path(self, scene_name: str) -> Path:
        scene_key = scene_name.replace("/", "__")
        return self.cache_dir / f"{scene_key}_ps{self.position_scale:g}_frames_v{_SIM_CACHE_VERSION}.pt"

    def _load_or_build_scene_frames(self, scene_name: str) -> list[_SimFrame]:
        cached = self._scene_frames_cache.get(scene_name)
        if cached is not None:
            return cached
        cache_path = self._scene_index_cache_path(scene_name)
        frames: list[_SimFrame]
        if cache_path.exists():
            try:
                payload = torch.load(cache_path, map_location="cpu")
                frames = [
                    _SimFrame(
                        frame_idx=int(item["frame_idx"]),
                        rgb_path=Path(str(item["rgb_path"])),
                        depth_path=Path(str(item["depth_path"])),
                        position_xyz=torch.tensor(item["position_xyz"], dtype=torch.float32),
                    )
                    for item in payload["frames"]
                ]
            except Exception:
                frames = self._scan_scene_frames(scene_name)
                payload = {
                    "scene": scene_name,
                    "frames": [
                        {
                            "frame_idx": int(frame.frame_idx),
                            "rgb_path": str(frame.rgb_path),
                            "depth_path": str(frame.depth_path),
                            "position_xyz": frame.position_xyz.tolist(),
                        }
                        for frame in frames
                    ],
                }
                self._atomic_torch_save(cache_path, payload)
        else:
            frames = self._scan_scene_frames(scene_name)
            payload = {
                "scene": scene_name,
                "frames": [
                    {
                        "frame_idx": int(frame.frame_idx),
                        "rgb_path": str(frame.rgb_path),
                        "depth_path": str(frame.depth_path),
                        "position_xyz": frame.position_xyz.tolist(),
                    }
                    for frame in frames
                ],
            }
            self._atomic_torch_save(cache_path, payload)
        self._scene_frames_cache[scene_name] = frames
        return frames

    def _random_chunk_pairs(self, chunk: list[_SimFrame], rng: random.Random) -> list[tuple[int, int]]:
        if len(chunk) < self.chunk_size:
            return []
        indices = list(range(len(chunk)))
        rng.shuffle(indices)
        max_pairs = min(self.pairs_per_chunk, len(indices) // 2)
        return [(indices[2 * i], indices[2 * i + 1]) for i in range(max_pairs)]

    def __iter__(self):
        scene_names = list(self.scene_names)
        order_rng = random.Random(self.seed + self.epoch)
        if self.shuffle_scene:
            order_rng.shuffle(scene_names)
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        total_shards = max(1, self.ddp_world_size * num_workers)
        shard_id = self.ddp_rank * num_workers + worker_id
        pair_unit_index = 0

        for scene_order_idx, scene_name in enumerate(scene_names):
            try:
                frames = self._load_or_build_scene_frames(scene_name)
            except Exception:
                continue
            if len(frames) < self.chunk_size:
                continue
            for start in range(0, len(frames), self.chunk_size):
                chunk = frames[start : start + self.chunk_size]
                if len(chunk) < self.chunk_size:
                    break
                try:
                    equ_h, equ_w = self._image_hw(chunk[0].rgb_path)
                    if self.max_long_edge > 0 and max(equ_h, equ_w) > self.max_long_edge:
                        scale = float(self.max_long_edge) / float(max(equ_h, equ_w))
                        equ_h = max(2, int(round(float(equ_h) * scale)))
                        equ_w = max(2, int(round(float(equ_w) * scale)))
                    face_w = max(1, equ_h // 2)
                    converter = _EquirecToCube(equ_h=equ_h, equ_w=equ_w, face_w=face_w)
                except Exception:
                    continue
                def load_frame(local_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                    frame = chunk[local_idx]
                    rgb = self._load_rgb(frame.rgb_path)
                    depth = self._load_depth(frame.depth_path)
                    rgb, depth = self._resize_erp_if_needed(rgb, depth)
                    cube_rgb, cube_depth = converter.run(rgb, depth)
                    return rgb, depth, cube_rgb, cube_depth

                chunk_rng = random.Random(
                    self.seed + self.epoch * 1000003 + scene_order_idx * 1009 + start
                )
                pairs = self._random_chunk_pairs(chunk, chunk_rng)
                for src_local, tgt_local in pairs:
                    if pair_unit_index % total_shards != shard_id:
                        pair_unit_index += 1
                        continue
                    pair_unit_index += 1
                    src_rgb, src_depth, src_cube_rgb, src_cube_depth = load_frame(src_local)
                    tgt_rgb, tgt_depth, tgt_cube_rgb, tgt_cube_depth = load_frame(tgt_local)
                    yield PanOGSSample(
                        src_erp_rgb_u8=src_rgb,
                        tgt_erp_rgb_u8=tgt_rgb,
                        src_erp_depth_m=src_depth,
                        tgt_erp_depth_m=tgt_depth,
                        src_cube_rgb_u8=src_cube_rgb,
                        tgt_cube_rgb_u8=tgt_cube_rgb,
                        src_cube_depth_m=src_cube_depth,
                        tgt_cube_depth_m=tgt_cube_depth,
                        src_R=torch.eye(3, dtype=torch.float32),
                        src_t=chunk[src_local].position_xyz.clone(),
                        tgt_R=torch.eye(3, dtype=torch.float32),
                        tgt_t=chunk[tgt_local].position_xyz.clone(),
                        src_idx=int(chunk[src_local].frame_idx),
                        tgt_idx=int(chunk[tgt_local].frame_idx),
                        scene=str(scene_name),
                    )

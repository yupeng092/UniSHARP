from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import IterableDataset

from unisharp.datasets.pair_sampling import project_overlap_ratio, resize_k3_align_corners_false, resize_rgb_u8_chw_high_quality
from unisharp import DEFAULT_MAX_DEPTH_M


@dataclass(frozen=True)
class WildRGBDPairSample:
    src_rgb_u8: torch.Tensor
    tgt_rgb_u8: torch.Tensor
    src_depth_m: torch.Tensor
    tgt_depth_m: torch.Tensor
    src_w2c: torch.Tensor
    tgt_w2c: torch.Tensor
    src_intrinsics: torch.Tensor
    tgt_intrinsics: torch.Tensor
    src_idx: int
    tgt_idx: int
    scene: str


class WildRGBDDataset(IterableDataset):

    def __init__(
        self,
        root: Path | None = None,
        scene_list_file: Path | None = None,
        split: str = "train",
        min_frame_gap: int = 1,
        max_frame_gap: int = 32,
        pair_max_translation_m: float = 0.5,
        pair_min_overlap: float = 0.6,
        pair_overlap_sample_h: int = 32,
        pair_overlap_sample_w: int = 56,
        pair_max_tries: int = 32,
        output_h: int | None = None,
        output_w: int | None = None,
        shuffle_scene: bool = True,
        shuffle_frame: bool = True,
        ddp_rank: int = 0,
        ddp_world_size: int = 1,
        roots: list[Path] | None = None,
        depth_max_m: float = DEFAULT_MAX_DEPTH_M,
        seed: int = 0,
        verify_manifest_paths: bool = False,
    ) -> None:
        super().__init__()
        self.root = root
        self.split = split
        self.min_frame_gap = int(min_frame_gap)
        self.max_frame_gap = int(max_frame_gap)
        self.pair_max_translation_m = float(pair_max_translation_m)
        self.pair_min_overlap = float(pair_min_overlap)
        self.pair_overlap_sample_h = int(pair_overlap_sample_h)
        self.pair_overlap_sample_w = int(pair_overlap_sample_w)
        self.pair_max_tries = int(pair_max_tries)
        self.output_h = int(output_h) if output_h is not None else None
        self.output_w = int(output_w) if output_w is not None else None
        self.shuffle_scene = bool(shuffle_scene)
        self.shuffle_frame = bool(shuffle_frame)
        self.ddp_rank = int(ddp_rank)
        self.ddp_world_size = int(ddp_world_size)
        self.depth_max_m = float(depth_max_m)
        self.seed = int(seed)
        self.epoch = 0
        self.verify_manifest_paths = bool(verify_manifest_paths)
        self.roots = [Path(p) for p in roots] if roots is not None else ([Path(root)] if root is not None else [])
        if not self.roots:
            raise ValueError("WildRGBDDataset requires at least one root path.")
        self.scene_dirs: list[Path] = []
        self.scene_list_file = Path(scene_list_file) if scene_list_file is not None else None
        if self.scene_list_file is not None:
            if not self.scene_list_file.exists():
                raise FileNotFoundError(self.scene_list_file)
            for raw in self.scene_list_file.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line:
                    continue
                scene_dir = Path(line)
                if (not self.verify_manifest_paths) or scene_dir.is_dir():
                    self.scene_dirs.append(scene_dir)
        else:
            for ds_root in self.roots:
                split_dir = ds_root / self.split
                if not split_dir.exists():
                    raise FileNotFoundError(split_dir)
                self.scene_dirs.extend(sorted([p for p in split_dir.iterdir() if p.is_dir()]))
        if not self.scene_dirs:
            raise RuntimeError("No scene folders found in the configured WildRGBD roots.")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @staticmethod
    def _load_scene_pose_and_k(scene_dir: Path) -> tuple[np.ndarray, dict[int, np.ndarray], torch.Tensor]:
        metadata_path = scene_dir / "metadata"
        with metadata_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        k_raw = np.asarray(meta["K"], dtype=np.float32).reshape(3, 3).T
        k = torch.from_numpy(k_raw.copy()).to(torch.float32)

        pose_path = scene_dir / "cam_poses.txt"
        pose_rows = np.genfromtxt(str(pose_path), dtype=np.float32)
        if pose_rows.ndim == 1:
            pose_rows = pose_rows[None, :]
        if pose_rows.shape[1] < 17:
            raise ValueError(f"Bad cam_poses.txt shape={pose_rows.shape} at {pose_path}")
        frame_ids = pose_rows[:, 0].astype(np.int64)
        c2w = pose_rows[:, 1:17].reshape(-1, 4, 4).astype(np.float32)
        w2c = np.linalg.inv(c2w).astype(np.float32)
        w2c_map = {int(fid): w2c[i] for i, fid in enumerate(frame_ids.tolist())}
        return frame_ids, w2c_map, k

    @staticmethod
    def _collect_frame_ids(folder: Path) -> set[int]:
        ids: set[int] = set()
        if not folder.exists():
            return ids
        for p in folder.iterdir():
            if not p.is_file():
                continue
            if p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue
            try:
                ids.add(int(p.stem))
            except ValueError:
                continue
        return ids

    @staticmethod
    def _resolve_img_path(folder: Path, idx: int) -> Path:
        for ext in (".png", ".jpg", ".jpeg"):
            p = folder / f"{idx:05d}{ext}"
            if p.exists():
                return p
        raise FileNotFoundError(folder / f"{idx:05d}.png")

    @staticmethod
    def _load_rgb_u8(path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        arr = np.asarray(img, dtype=np.uint8).copy()
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    def _load_depth_m(self, depth_path: Path) -> torch.Tensor:
        dep = np.asarray(Image.open(depth_path))
        if dep.ndim != 2:
            raise ValueError(f"Expected single-channel depth at {depth_path}, got {dep.shape}")
        depth = dep.astype(np.float32)
        if float(np.nanmax(depth)) > 200.0:
            depth = depth / 1000.0
        depth[~np.isfinite(depth)] = 0.0
        depth = np.clip(depth, a_min=0.0, a_max=self.depth_max_m)
        return torch.from_numpy(depth).unsqueeze(0).to(torch.float32)

    @staticmethod
    def _scene_name(scene_dir: Path) -> str:
        parent = scene_dir.parent.parent.name if scene_dir.parent.name == "scenes" else scene_dir.parent.name
        return f"{parent}/{scene_dir.name}"

    def _sample_target_for_src(
        self,
        src_idx: int,
        valid_ids: list[int],
        w2c_map: dict[int, np.ndarray],
        intr: torch.Tensor,
        h: int,
        w: int,
        rng: random.Random,
    ) -> int | None:
        src_w2c = torch.from_numpy(w2c_map[int(src_idx)]).to(torch.float32)
        src_center = torch.linalg.inv(src_w2c)[:3, 3]
        candidates: list[int] = []
        for j in valid_ids:
            if int(j) == int(src_idx):
                continue
            gap = abs(int(j) - int(src_idx))
            if gap < self.min_frame_gap or gap > self.max_frame_gap:
                continue
            jw2c = torch.from_numpy(w2c_map[int(j)]).to(torch.float32)
            jcenter = torch.linalg.inv(jw2c)[:3, 3]
            trans = torch.norm(jcenter - src_center, p=2).item()
            if trans > self.pair_max_translation_m:
                continue
            candidates.append(int(j))
        if not candidates:
            return None

        rng.shuffle(candidates)
        tries = min(self.pair_max_tries, len(candidates))
        src_k = intr.to(torch.float32)
        src_w2c_t = src_w2c.to(torch.float32)
        for j in candidates[:tries]:
            tgt_w2c_t = torch.from_numpy(w2c_map[int(j)]).to(torch.float32)
            ov_st = project_overlap_ratio(
                src_w2c=src_w2c_t,
                tgt_w2c=tgt_w2c_t,
                src_k=src_k,
                tgt_k=src_k,
                h=h,
                w=w,
                sample_h=self.pair_overlap_sample_h,
                sample_w=self.pair_overlap_sample_w,
            )
            ov_ts = project_overlap_ratio(
                src_w2c=tgt_w2c_t,
                tgt_w2c=src_w2c_t,
                src_k=src_k,
                tgt_k=src_k,
                h=h,
                w=w,
                sample_h=self.pair_overlap_sample_h,
                sample_w=self.pair_overlap_sample_w,
            )
            if 0.5 * (ov_st + ov_ts) >= self.pair_min_overlap:
                return int(j)
        return None

    def __iter__(self):
        scenes = list(self.scene_dirs)
        order_rng = random.Random(self.seed + self.epoch)
        if self.shuffle_scene:
            order_rng.shuffle(scenes)

        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        total_shards = max(1, self.ddp_world_size * num_workers)
        shard_id = self.ddp_rank * num_workers + worker_id
        src_unit_index = 0

        for scene_order_idx, scene_dir in enumerate(scenes):
            try:
                pose_ids_np, w2c_map, intr = self._load_scene_pose_and_k(scene_dir)
            except Exception:
                continue
            pose_ids = {int(x) for x in pose_ids_np.tolist()}
            rgb_ids = self._collect_frame_ids(scene_dir / "rgb")
            dep_ids = self._collect_frame_ids(scene_dir / "depth")
            valid_ids = sorted(list(pose_ids & rgb_ids & dep_ids))
            if len(valid_ids) < 2:
                continue

            src_order = list(valid_ids)
            scene_rng = random.Random(self.seed + self.epoch * 1000003 + scene_order_idx)
            if self.shuffle_frame:
                scene_rng.shuffle(src_order)

            for src_idx in src_order:
                if src_unit_index % total_shards != shard_id:
                    src_unit_index += 1
                    continue
                src_unit_index += 1
                try:
                    rgb_src_path = self._resolve_img_path(scene_dir / "rgb", int(src_idx))
                    dep_src_path = self._resolve_img_path(scene_dir / "depth", int(src_idx))
                    src_img = self._load_rgb_u8(rgb_src_path)
                    src_depth = self._load_depth_m(dep_src_path)
                except Exception:
                    continue

                h, w = int(src_img.shape[1]), int(src_img.shape[2])
                tgt_idx = self._sample_target_for_src(
                    src_idx=int(src_idx),
                    valid_ids=valid_ids,
                    w2c_map=w2c_map,
                    intr=intr,
                    h=h,
                    w=w,
                    rng=scene_rng,
                )
                if tgt_idx is None:
                    continue

                try:
                    rgb_tgt_path = self._resolve_img_path(scene_dir / "rgb", int(tgt_idx))
                    dep_tgt_path = self._resolve_img_path(scene_dir / "depth", int(tgt_idx))
                    tgt_img = self._load_rgb_u8(rgb_tgt_path)
                    tgt_depth = self._load_depth_m(dep_tgt_path)
                except Exception:
                    continue

                if src_img.shape != tgt_img.shape:
                    continue

                src_intr = intr.clone()
                tgt_intr = intr.clone()
                if self.output_h is not None and self.output_w is not None:
                    oh, ow = int(src_img.shape[1]), int(src_img.shape[2])
                    if oh > 0 and ow > 0 and (oh != self.output_h or ow != self.output_w):
                        sx = float(self.output_w) / float(ow)
                        sy = float(self.output_h) / float(oh)
                        src_img = resize_rgb_u8_chw_high_quality(src_img, size=(self.output_h, self.output_w))
                        tgt_img = resize_rgb_u8_chw_high_quality(tgt_img, size=(self.output_h, self.output_w))
                        src_depth = F.interpolate(
                            src_depth.unsqueeze(0),
                            size=(self.output_h, self.output_w),
                            mode="nearest",
                        ).squeeze(0)
                        tgt_depth = F.interpolate(
                            tgt_depth.unsqueeze(0),
                            size=(self.output_h, self.output_w),
                            mode="nearest",
                        ).squeeze(0)
                        src_intr = resize_k3_align_corners_false(src_intr, sx=sx, sy=sy)
                        tgt_intr = resize_k3_align_corners_false(tgt_intr, sx=sx, sy=sy)

                yield WildRGBDPairSample(
                    src_rgb_u8=src_img,
                    tgt_rgb_u8=tgt_img,
                    src_depth_m=src_depth,
                    tgt_depth_m=tgt_depth,
                    src_w2c=torch.from_numpy(w2c_map[int(src_idx)]).to(torch.float32),
                    tgt_w2c=torch.from_numpy(w2c_map[int(tgt_idx)]).to(torch.float32),
                    src_intrinsics=src_intr,
                    tgt_intrinsics=tgt_intr,
                    src_idx=int(src_idx),
                    tgt_idx=int(tgt_idx),
                    scene=self._scene_name(scene_dir),
                )


def wildrgbd_collate(batch: list[WildRGBDPairSample]) -> WildRGBDPairSample:
    def stack(xs):
        if isinstance(xs[0], torch.Tensor):
            return torch.stack(xs, dim=0)
        return xs

    return WildRGBDPairSample(
        src_rgb_u8=stack([b.src_rgb_u8 for b in batch]),
        tgt_rgb_u8=stack([b.tgt_rgb_u8 for b in batch]),
        src_depth_m=stack([b.src_depth_m for b in batch]),
        tgt_depth_m=stack([b.tgt_depth_m for b in batch]),
        src_w2c=stack([b.src_w2c for b in batch]),
        tgt_w2c=stack([b.tgt_w2c for b in batch]),
        src_intrinsics=stack([b.src_intrinsics for b in batch]),
        tgt_intrinsics=stack([b.tgt_intrinsics for b in batch]),
        src_idx=[b.src_idx for b in batch],  # type: ignore[arg-type]
        tgt_idx=[b.tgt_idx for b in batch],  # type: ignore[arg-type]
        scene=[b.scene for b in batch],  # type: ignore[arg-type]
    )


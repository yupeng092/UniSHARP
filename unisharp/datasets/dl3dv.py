
from __future__ import annotations

from collections import defaultdict, deque
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import IterableDataset

from unisharp.datasets.pair_sampling import (
    project_overlap_ratio,
    resize_k3_align_corners_false,
    resize_rgb_u8_chw_high_quality,
    select_targets_for_source,
)
from unisharp.datasets.re10k import Re10KPairSample, re10k_collate
from unisharp import DEFAULT_MAX_DEPTH_M


class DL3DVDataset(IterableDataset):
    def __init__(
        self,
        root: Path,
        depth_root: Path,
        scene_specs_file: Path | None = None,
        min_frame_gap: int = 1,
        max_frame_gap: int = 32,
        pair_max_translation_m: float = 0.5,
        pair_min_overlap: float = 0.6,
        pair_overlap_sample_h: int = 32,
        pair_overlap_sample_w: int = 56,
        output_h: int | None = None,
        output_w: int | None = None,
        shuffle_scene: bool = True,
        shuffle_frame: bool = False,
        ddp_rank: int = 0,
        ddp_world_size: int = 1,
        batch_size_hint: int = 1,
        depth_max_m: float = DEFAULT_MAX_DEPTH_M,
        seed: int = 0,
        verify_manifest_paths: bool = False,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.depth_root = Path(depth_root)
        self.min_frame_gap = int(min_frame_gap)
        self.max_frame_gap = int(max_frame_gap)
        self.pair_max_translation_m = float(pair_max_translation_m)
        self.pair_min_overlap = float(pair_min_overlap)
        self.pair_overlap_sample_h = int(pair_overlap_sample_h)
        self.pair_overlap_sample_w = int(pair_overlap_sample_w)
        self.output_h = int(output_h) if output_h is not None else None
        self.output_w = int(output_w) if output_w is not None else None
        self.shuffle_scene = bool(shuffle_scene)
        self.shuffle_frame = bool(shuffle_frame)
        self.ddp_rank = int(ddp_rank)
        self.ddp_world_size = int(ddp_world_size)
        self.batch_size_hint = int(max(1, batch_size_hint))
        self.depth_max_m = float(depth_max_m)
        self.seed = int(seed)
        self.epoch = 0
        self.verify_manifest_paths = bool(verify_manifest_paths)
        self.scene_specs_file = Path(scene_specs_file) if scene_specs_file is not None else None
        self.scene_specs = self._load_scene_specs()
        if not self.scene_specs:
            raise RuntimeError(f"No valid DL3DV scenes found under {self.root}")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _load_scene_specs(self) -> list[tuple[str, Path, Path]]:
        if self.scene_specs_file is None:
            return self._scan_scenes()
        if not self.scene_specs_file.exists():
            raise FileNotFoundError(self.scene_specs_file)
        out: list[tuple[str, Path, Path]] = []
        for raw in self.scene_specs_file.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) != 3:
                continue
            scene_name, scene_dir_raw, depth_dir_raw = parts
            scene_dir = Path(scene_dir_raw)
            depth_dir = Path(depth_dir_raw)
            if (not self.verify_manifest_paths) or (scene_dir.exists() and depth_dir.exists()):
                out.append((scene_name, scene_dir, depth_dir))
        return out

    def _scan_scenes(self) -> list[tuple[str, Path, Path]]:
        out: list[tuple[str, Path, Path]] = []
        for bucket_dir in sorted([p for p in self.root.iterdir() if p.is_dir()]):
            for scene_stub in sorted([p for p in bucket_dir.iterdir() if p.is_dir()]):
                inner_dirs = [p for p in scene_stub.iterdir() if p.is_dir()]
                scene_dir = inner_dirs[0] if inner_dirs else scene_stub
                transforms_path = scene_dir / "transforms.json"
                image_dir = scene_dir / "images_4"
                depth_dir = self.depth_root / bucket_dir.name / scene_stub.name / "exports" / "mini_npz" / "per_image"
                if transforms_path.exists() and image_dir.exists() and depth_dir.exists():
                    scene_name = f"{bucket_dir.name}/{scene_stub.name}"
                    out.append((scene_name, scene_dir, depth_dir))
        return out

    @staticmethod
    def _load_rgb_u8(path: Path) -> torch.Tensor:
        arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8).copy()
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    def _load_depth_m(self, path: Path) -> torch.Tensor:
        payload = np.load(path)
        depth = payload["depth"].astype(np.float32)
        depth[~np.isfinite(depth)] = 0.0
        depth = np.clip(depth, a_min=0.0, a_max=self.depth_max_m)
        return torch.from_numpy(depth).unsqueeze(0)

    @staticmethod
    def _resize_depth_to_image(depth: torch.Tensor, image_hw: tuple[int, int]) -> torch.Tensor:
        target_h, target_w = int(image_hw[0]), int(image_hw[1])
        if depth.shape[-2:] == (target_h, target_w):
            return depth
        return F.interpolate(
            depth.unsqueeze(0),
            size=(target_h, target_w),
            mode="nearest",
        ).squeeze(0)

    @staticmethod
    def _frame_id_from_name(name: str) -> int:
        stem = Path(name).stem
        return int(stem.split("_")[-1])

    def _load_scene(
        self,
        scene_name: str,
        scene_dir: Path,
        depth_dir: Path,
    ) -> tuple[list[int], dict[int, Path], dict[int, Path], dict[int, torch.Tensor], dict[int, torch.Tensor], torch.Tensor]:
        meta = json.loads((scene_dir / "transforms.json").read_text())
        orig_w = int(meta["w"])
        orig_h = int(meta["h"])
        k = torch.eye(3, dtype=torch.float32)
        k[0, 0] = float(meta["fl_x"])
        k[1, 1] = float(meta["fl_y"])
        k[0, 2] = float(meta["cx"])
        k[1, 2] = float(meta["cy"])

        image_dir = scene_dir / "images_4"
        image_paths = {self._frame_id_from_name(p.name): p for p in image_dir.glob("*.png")}
        depth_paths = {self._frame_id_from_name(p.name): p for p in depth_dir.glob("*.npz")}
        w2c_map: dict[int, torch.Tensor] = {}
        intr_map: dict[int, torch.Tensor] = {}
        valid_ids: list[int] = []

        example_img = None
        for frame in meta.get("frames", []):
            rel_path = str(frame.get("file_path", ""))
            frame_name = Path(rel_path).name
            frame_id = self._frame_id_from_name(frame_name)
            if frame_id not in image_paths or frame_id not in depth_paths:
                continue
            c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
            c2w[:3, 1:3] *= -1.0
            if example_img is None:
                example_img = self._load_rgb_u8(image_paths[frame_id])
            cur_h, cur_w = int(example_img.shape[1]), int(example_img.shape[2])
            k_cur = k.clone()
            if cur_h != orig_h or cur_w != orig_w:
                sx = float(cur_w) / float(orig_w)
                sy = float(cur_h) / float(orig_h)
                k_cur = resize_k3_align_corners_false(k_cur, sx=sx, sy=sy)
            w2c_map[frame_id] = torch.linalg.inv(c2w)
            intr_map[frame_id] = k_cur
            valid_ids.append(frame_id)
        valid_ids = sorted(valid_ids)
        return valid_ids, image_paths, depth_paths, w2c_map, intr_map, k

    def __iter__(self):
        scenes = list(self.scene_specs)
        order_rng = random.Random(self.seed + self.epoch)
        if self.shuffle_scene:
            order_rng.shuffle(scenes)
        pending_by_hw: dict[tuple[int, int], deque[Re10KPairSample]] = defaultdict(deque)
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        total_shards = max(1, self.ddp_world_size * num_workers)
        shard_id = self.ddp_rank * num_workers + worker_id
        src_unit_index = 0

        for scene_order_idx, (scene_name, scene_dir, depth_dir) in enumerate(scenes):
            try:
                valid_ids, image_paths, depth_paths, w2c_map, intr_map, _ = self._load_scene(scene_name, scene_dir, depth_dir)
            except Exception:
                continue
            if len(valid_ids) < 2:
                continue
            src_order = list(valid_ids)
            scene_rng = random.Random(self.seed + self.epoch * 1000003 + scene_order_idx)
            if self.shuffle_frame:
                scene_rng.shuffle(src_order)
            centers = torch.stack([torch.linalg.inv(w2c_map[i])[:3, 3] for i in valid_ids], dim=0)
            frame_to_pos = {fid: pos for pos, fid in enumerate(valid_ids)}

            def overlap_avg(src_pos: int, tgt_pos: int) -> float:
                src_fid = int(valid_ids[src_pos])
                tgt_fid = int(valid_ids[tgt_pos])
                src_img_path = image_paths[src_fid]
                with Image.open(src_img_path) as img:
                    w = int(img.size[0])
                    h = int(img.size[1])
                return float(
                    0.5
                    * (
                        project_overlap_ratio(
                            src_w2c=w2c_map[src_fid],
                            tgt_w2c=w2c_map[tgt_fid],
                            src_k=intr_map[src_fid],
                            tgt_k=intr_map[tgt_fid],
                            h=h,
                            w=w,
                            sample_h=self.pair_overlap_sample_h,
                            sample_w=self.pair_overlap_sample_w,
                        )
                        + project_overlap_ratio(
                            src_w2c=w2c_map[tgt_fid],
                            tgt_w2c=w2c_map[src_fid],
                            src_k=intr_map[tgt_fid],
                            tgt_k=intr_map[src_fid],
                            h=h,
                            w=w,
                            sample_h=self.pair_overlap_sample_h,
                            sample_w=self.pair_overlap_sample_w,
                        )
                    )
                )

            for src_idx in src_order:
                if src_unit_index % total_shards != shard_id:
                    src_unit_index += 1
                    continue
                src_unit_index += 1
                src_pos = int(frame_to_pos[int(src_idx)])
                tgt_pos_list = select_targets_for_source(
                    src_idx=src_pos,
                    candidate_indices=list(range(len(valid_ids))),
                    centers=centers,
                    min_index_gap=int(self.min_frame_gap),
                    max_index_gap=int(self.max_frame_gap),
                    pair_max_translation_m=float(self.pair_max_translation_m),
                    pair_min_overlap=float(self.pair_min_overlap),
                    overlap_score_fn=overlap_avg,
                )
                if not tgt_pos_list:
                    continue
                tgt_idx = int(valid_ids[scene_rng.choice(tgt_pos_list)])
                try:
                    src_img = self._load_rgb_u8(image_paths[int(src_idx)])
                    tgt_img = self._load_rgb_u8(image_paths[int(tgt_idx)])
                    src_depth = self._load_depth_m(depth_paths[int(src_idx)])
                    tgt_depth = self._load_depth_m(depth_paths[int(tgt_idx)])
                except Exception:
                    continue
                src_depth = self._resize_depth_to_image(src_depth, (int(src_img.shape[1]), int(src_img.shape[2])))
                tgt_depth = self._resize_depth_to_image(tgt_depth, (int(tgt_img.shape[1]), int(tgt_img.shape[2])))
                src_intr = intr_map[int(src_idx)].clone()
                tgt_intr = intr_map[int(tgt_idx)].clone()
                if self.output_h is not None and self.output_w is not None:
                    oh, ow = int(src_img.shape[1]), int(src_img.shape[2])
                    if oh != self.output_h or ow != self.output_w:
                        sx = float(self.output_w) / float(ow)
                        sy = float(self.output_h) / float(oh)
                        src_img = resize_rgb_u8_chw_high_quality(src_img, size=(self.output_h, self.output_w))
                        tgt_img = resize_rgb_u8_chw_high_quality(tgt_img, size=(self.output_h, self.output_w))
                        src_depth = F.interpolate(src_depth[None], size=(self.output_h, self.output_w), mode="nearest")[0]
                        tgt_depth = F.interpolate(tgt_depth[None], size=(self.output_h, self.output_w), mode="nearest")[0]
                        src_intr = resize_k3_align_corners_false(src_intr, sx=sx, sy=sy)
                        tgt_intr = resize_k3_align_corners_false(tgt_intr, sx=sx, sy=sy)
                sample = Re10KPairSample(
                    src_rgb_u8=src_img,
                    tgt_rgb_u8=tgt_img,
                    src_w2c=w2c_map[int(src_idx)],
                    tgt_w2c=w2c_map[int(tgt_idx)],
                    src_intrinsics=src_intr,
                    tgt_intrinsics=tgt_intr,
                    src_idx=int(src_idx),
                    tgt_idx=int(tgt_idx),
                    scene=scene_name,
                    src_depth_m=src_depth,
                    tgt_depth_m=tgt_depth,
                )
                hw_key = (int(sample.src_rgb_u8.shape[1]), int(sample.src_rgb_u8.shape[2]))
                bucket = pending_by_hw[hw_key]
                bucket.append(sample)
                if self.batch_size_hint <= 1:
                    yield bucket.popleft()
                    continue
                while len(bucket) >= self.batch_size_hint:
                    packed = [bucket.popleft() for _ in range(self.batch_size_hint)]
                    yield re10k_collate(packed)

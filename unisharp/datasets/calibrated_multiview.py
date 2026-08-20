from __future__ import annotations

"""Generic calibrated human multi-view JSONL dataset.

Each JSONL row is one scene::

  {"scene": "subject/expression", "frames": [
    {"image": "/abs/a.jpg", "intrinsics": [[...]], "w2c": [[...]]}, ...
  ]}

``w2c`` is an OpenCV world-to-camera 4x4 transform and intrinsics are pixel
3x3.  HUMBI, NeRSemble, AIST++ and rendered THuman sequences can be
converted to this small explicit format without changing the training code.
"""

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info

from unisharp.datasets.pair_sampling import resize_k3_align_corners_false, resize_rgb_u8_chw_high_quality
from unisharp.datasets.re10k import Re10KPairSample


class CalibratedMultiViewDataset(IterableDataset[Re10KPairSample]):
    def __init__(
        self,
        manifest: Path,
        *,
        output_h: int,
        output_w: int,
        ddp_rank: int = 0,
        ddp_world_size: int = 1,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.manifest = Path(manifest)
        self.output_h, self.output_w = int(output_h), int(output_w)
        self.ddp_rank, self.ddp_world_size = int(ddp_rank), int(ddp_world_size)
        self.seed, self.epoch = int(seed), 0
        self.scenes = self._read_manifest(self.manifest)
        if not self.scenes:
            raise RuntimeError(f"No calibrated multi-view scenes in {self.manifest}")

    @staticmethod
    def _read_manifest(path: Path) -> list[tuple[str, list[dict[str, Any]]]]:
        scenes: list[tuple[str, list[dict[str, Any]]]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            frames = [frame for frame in item.get("frames", []) if Path(str(frame.get("image", ""))).is_file()]
            if len(frames) >= 2:
                scenes.append((str(item.get("scene", path.stem)), frames))
        return scenes

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @staticmethod
    def _load_u8(path: Path) -> torch.Tensor:
        array = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8).copy()
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    def _frame(self, frame: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image = self._load_u8(Path(str(frame["image"])))
        height, width = int(image.shape[-2]), int(image.shape[-1])
        intrinsics = torch.tensor(frame["intrinsics"], dtype=torch.float32).reshape(3, 3)
        w2c = torch.tensor(frame["w2c"], dtype=torch.float32).reshape(4, 4)
        image = resize_rgb_u8_chw_high_quality(image, size=(self.output_h, self.output_w))
        intrinsics = resize_k3_align_corners_false(
            intrinsics, sx=float(self.output_w) / float(width), sy=float(self.output_h) / float(height)
        )
        return image, intrinsics, w2c

    def __iter__(self):
        worker = get_worker_info()
        worker_id, worker_count = (worker.id, worker.num_workers) if worker is not None else (0, 1)
        indices = list(range(self.ddp_rank, len(self.scenes), max(1, self.ddp_world_size)))[worker_id::worker_count]
        rng = random.Random(self.seed + self.epoch * 100003 + self.ddp_rank * 997 + worker_id)
        rng.shuffle(indices)
        for scene_index in indices:
            scene, frames = self.scenes[scene_index]
            src_index, tgt_index = rng.sample(range(len(frames)), 2)
            src, src_k, src_w2c = self._frame(frames[src_index])
            tgt, tgt_k, tgt_w2c = self._frame(frames[tgt_index])
            yield Re10KPairSample(
                src_rgb_u8=src, tgt_rgb_u8=tgt,
                src_w2c=src_w2c, tgt_w2c=tgt_w2c,
                src_intrinsics=src_k, tgt_intrinsics=tgt_k,
                src_idx=src_index, tgt_idx=tgt_index, scene=scene,
            )

from __future__ import annotations

"""Single-image portrait/outdoor appearance batches for UniSHARP pre-training.

These corpora have person/face boxes but no calibrated target camera or depth.
They therefore train the source-view reconstruction branch only: source and
target are the same cropped image and use an identity camera transform.  This
must be mixed with a calibrated multi-view corpus for novel-view geometry.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info

from unisharp.datasets.pair_sampling import resize_rgb_u8_chw_high_quality
from unisharp.datasets.re10k import Re10KPairSample


class AppearanceDataset(IterableDataset[Re10KPairSample]):
    def __init__(
        self,
        manifest: Path,
        *,
        output_h: int,
        output_w: int,
        crop_probability: float = 0.8,
        ddp_rank: int = 0,
        ddp_world_size: int = 1,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.manifest = Path(manifest)
        self.output_h, self.output_w = int(output_h), int(output_w)
        if self.output_h < 1 or self.output_w < 1:
            raise ValueError("AppearanceDataset requires a positive fixed output size.")
        self.crop_probability = float(crop_probability)
        self.ddp_rank, self.ddp_world_size = int(ddp_rank), int(ddp_world_size)
        self.seed, self.epoch = int(seed), 0
        self.records = self._read_manifest(self.manifest)
        if not self.records:
            raise RuntimeError(f"No readable image records in {self.manifest}")

    @staticmethod
    def _read_manifest(path: Path) -> list[tuple[Path, list[list[float]]]]:
        records: list[tuple[Path, list[list[float]]]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            boxes: list[list[float]] = []
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                image = Path(line)
            else:
                image = Path(str(item["image"]))
                raw = item.get("face_boxes_xywh", item.get("person_boxes_xywh", []))
                if isinstance(raw, list):
                    boxes = [[float(v) for v in box[:4]] for box in raw if isinstance(box, list) and len(box) >= 4]
            if image.is_file():
                records.append((image, boxes))
        return records

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @staticmethod
    def _load_u8(path: Path) -> torch.Tensor:
        array = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8).copy()
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    def _crop_box(self, image: torch.Tensor, boxes: list[list[float]], rng: random.Random) -> torch.Tensor:
        if not boxes or rng.random() > self.crop_probability:
            return image
        _, h, w = image.shape
        x, y, bw, bh = rng.choice(boxes)
        if bw <= 1.0 or bh <= 1.0:
            return image
        # Include context around the face/person, then match the training aspect ratio.
        side_w, side_h = max(bw * 1.8, 8.0), max(bh * 1.8, 8.0)
        target_ratio = float(self.output_w) / float(self.output_h)
        if side_w / side_h < target_ratio:
            side_w = side_h * target_ratio
        else:
            side_h = side_w / target_ratio
        cx, cy = x + 0.5 * bw, y + 0.5 * bh
        left = int(round(max(0.0, min(float(w) - side_w, cx - 0.5 * side_w))))
        top = int(round(max(0.0, min(float(h) - side_h, cy - 0.5 * side_h))))
        right = min(w, max(left + 2, int(round(left + side_w))))
        bottom = min(h, max(top + 2, int(round(top + side_h))))
        return image[:, top:bottom, left:right].contiguous()

    def __iter__(self):
        worker = get_worker_info()
        worker_id, worker_count = (worker.id, worker.num_workers) if worker is not None else (0, 1)
        indices = list(range(self.ddp_rank, len(self.records), max(1, self.ddp_world_size)))
        indices = indices[worker_id::worker_count]
        rng = random.Random(self.seed + self.epoch * 100003 + self.ddp_rank * 997 + worker_id)
        rng.shuffle(indices)
        for index in indices:
            path, boxes = self.records[index]
            image = self._crop_box(self._load_u8(path), boxes, rng)
            image = resize_rgb_u8_chw_high_quality(image, size=(self.output_h, self.output_w))
            focal = 0.9 * float(max(self.output_h, self.output_w))
            intrinsics = torch.tensor(
                [[focal, 0.0, 0.5 * (self.output_w - 1)], [0.0, focal, 0.5 * (self.output_h - 1)], [0.0, 0.0, 1.0]],
                dtype=torch.float32,
            )
            yield Re10KPairSample(
                src_rgb_u8=image,
                tgt_rgb_u8=image.clone(),
                src_w2c=torch.eye(4, dtype=torch.float32),
                tgt_w2c=torch.eye(4, dtype=torch.float32),
                src_intrinsics=intrinsics,
                tgt_intrinsics=intrinsics.clone(),
                src_idx=index,
                tgt_idx=index,
                scene=str(path),
            )

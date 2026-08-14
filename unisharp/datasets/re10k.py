from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from io import BytesIO
import logging
import os
from pathlib import Path
import random
import time

import torch
import torchvision.transforms as tf
from PIL import Image
from torch.utils.data import IterableDataset

from unisharp.datasets.pair_sampling import (
    project_overlap_ratio,
    resize_k3_align_corners_false,
    resize_rgb_u8_chw_high_quality,
    select_targets_for_source,
)
from unisharp import DEFAULT_MAX_DEPTH_M
from unisharp.utils.pixel_convention import normalized_intrinsics_to_integer_pixel_k
from unisharp.utils.unik3d_adapter import infer_unik3d_pinhole, load_unik3d_model


LOGGER = logging.getLogger(__name__)


def _torch_load_any(path: Path) -> object:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _pack_re10k_batch(batch: list["Re10KPairSample"]) -> "Re10KPairSample":
    def stack(xs):
        if isinstance(xs[0], torch.Tensor):
            ref_shape = tuple(xs[0].shape)
            for idx, x in enumerate(xs[1:], start=1):
                if tuple(x.shape) != ref_shape:
                    raise RuntimeError(
                        "RE10K collate got mixed tensor shapes: "
                        f"ref={ref_shape} mismatch_idx={idx} got={tuple(x.shape)}"
                    )
            return torch.stack(xs, dim=0)
        return xs

    def stack_optional_depth(xs):
        if all(torch.is_tensor(x) for x in xs):
            ref_shape = tuple(xs[0].shape)
            for idx, x in enumerate(xs[1:], start=1):
                if tuple(x.shape) != ref_shape:
                    raise RuntimeError(
                        "RE10K collate got mixed depth shapes: "
                        f"ref={ref_shape} mismatch_idx={idx} got={tuple(x.shape)}"
                    )
            return torch.stack(xs, dim=0)
        return None

    return Re10KPairSample(
        src_rgb_u8=stack([b.src_rgb_u8 for b in batch]),
        tgt_rgb_u8=stack([b.tgt_rgb_u8 for b in batch]),
        src_w2c=stack([b.src_w2c for b in batch]),
        tgt_w2c=stack([b.tgt_w2c for b in batch]),
        src_intrinsics=stack([b.src_intrinsics for b in batch]),
        tgt_intrinsics=stack([b.tgt_intrinsics for b in batch]),
        src_idx=[b.src_idx for b in batch],  # type: ignore[arg-type]
        tgt_idx=[b.tgt_idx for b in batch],  # type: ignore[arg-type]
        scene=[b.scene for b in batch],  # type: ignore[arg-type]
        src_depth_m=stack_optional_depth([b.src_depth_m for b in batch]),  # type: ignore[arg-type]
        tgt_depth_m=stack_optional_depth([b.tgt_depth_m for b in batch]),  # type: ignore[arg-type]
    )


def re10k_passthrough(batch: "Re10KPairSample") -> "Re10KPairSample":
    return batch


@dataclass(frozen=True)
class Re10KPairSample:
    src_rgb_u8: torch.Tensor
    tgt_rgb_u8: torch.Tensor
    src_w2c: torch.Tensor
    tgt_w2c: torch.Tensor
    src_intrinsics: torch.Tensor
    tgt_intrinsics: torch.Tensor
    src_idx: int
    tgt_idx: int
    scene: str
    src_depth_m: torch.Tensor | None = None
    tgt_depth_m: torch.Tensor | None = None


class Re10KDataset(IterableDataset):

    def __init__(
        self,
        root: Path,
        chunks_file: Path | None = None,
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
        shuffle_chunk: bool = True,
        shuffle_example: bool = True,
        ddp_rank: int = 0,
        ddp_world_size: int = 1,
        pseudo_depth_root: Path | None = None,
        pseudo_depth_autogen: bool = True,
        pseudo_depth_backbone: str = "vitl",
        pseudo_depth_device: str = "cpu",
        pseudo_lock_timeout_sec: float = 120.0,
        pseudo_lock_stale_sec: float = 1800.0,
        pseudo_wait_poll_sec: float = 0.25,
        batch_size_hint: int = 1,
        depth_max_m: float = DEFAULT_MAX_DEPTH_M,
        pseudo_far_depth_invalid_m: float = 30.0,
        seed: int = 0,
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
        self.shuffle_chunk = bool(shuffle_chunk)
        self.shuffle_example = bool(shuffle_example)
        self.ddp_rank = int(ddp_rank)
        self.ddp_world_size = int(ddp_world_size)
        self.to_tensor = tf.ToTensor()
        self.pseudo_depth_root = Path(pseudo_depth_root) if pseudo_depth_root is not None else None
        self.pseudo_depth_autogen = bool(pseudo_depth_autogen)
        self.pseudo_depth_backbone = str(pseudo_depth_backbone)
        self.pseudo_depth_device = str(pseudo_depth_device)
        self.pseudo_lock_timeout_sec = float(max(1.0, pseudo_lock_timeout_sec))
        self.pseudo_lock_stale_sec = float(max(30.0, pseudo_lock_stale_sec))
        self.pseudo_wait_poll_sec = float(max(0.05, pseudo_wait_poll_sec))
        self.batch_size_hint = int(max(1, batch_size_hint))
        self.depth_max_m = float(depth_max_m)
        self.pseudo_far_depth_invalid_m = float(pseudo_far_depth_invalid_m)
        self._pseudo_model: torch.nn.Module | None = None
        self.seed = int(seed)
        self.epoch = 0

        self.chunks_file = Path(chunks_file) if chunks_file is not None else None
        split_dir = self.root / self.split
        if self.chunks_file is not None:
            if not self.chunks_file.exists():
                raise FileNotFoundError(self.chunks_file)
            chunks: list[Path] = []
            for raw in self.chunks_file.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line:
                    continue
                p = Path(line)
                if not p.is_absolute():
                    p = split_dir / p
                if p.suffix == ".torch":
                    chunks.append(p)
            self.chunks = sorted(chunks)
        else:
            if not split_dir.exists():
                raise FileNotFoundError(split_dir)
            self.chunks = sorted([p for p in split_dir.iterdir() if p.suffix == ".torch"])
        if not self.chunks:
            source = self.chunks_file if self.chunks_file is not None else split_dir
            raise RuntimeError(f"No .torch chunks found for {source}")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

        if self.pseudo_depth_root is not None:
            (self.pseudo_depth_root / self.split).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _decode_image_u8(image_bytes_tensor: torch.Tensor) -> torch.Tensor:
        if image_bytes_tensor.dtype != torch.uint8:
            raise ValueError(f"Expected uint8 bytes tensor, got {image_bytes_tensor.dtype}")
        image = Image.open(BytesIO(image_bytes_tensor.numpy().tobytes())).convert("RGB")
        chw_float = tf.ToTensor()(image)
        return (chw_float * 255.0).round().to(torch.uint8)

    @staticmethod
    def _convert_pose_row_to_w2c(poses: torch.Tensor) -> torch.Tensor:
        t = poses.shape[0]
        w2c = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(t, 1, 1)
        w2c[:, :3] = poses[:, 6:].reshape(t, 3, 4).to(torch.float32)
        return w2c

    @staticmethod
    def _convert_intrinsics_to_pixel(poses: torch.Tensor, h: int, w: int) -> torch.Tensor:
        t = poses.shape[0]
        fx, fy, cx, cy = poses[:, 0], poses[:, 1], poses[:, 2], poses[:, 3]
        del t
        return normalized_intrinsics_to_integer_pixel_k(
            fx,
            fy,
            cx,
            cy,
            height=int(h),
            width=int(w),
        )

    @staticmethod
    def _sanitize_scene(scene: str) -> str:
        s = str(scene).strip()
        s = s.replace("\\", "__").replace("/", "__")
        return s if len(s) > 0 else "unknown_scene"

    def _pseudo_depth_path(self, scene: str, frame_idx: int) -> Path | None:
        if self.pseudo_depth_root is None:
            return None
        scene_key = self._sanitize_scene(scene)
        return self.pseudo_depth_root / self.split / scene_key / f"{int(frame_idx):05d}.pt"

    @staticmethod
    def _load_pseudo_depth(path: Path) -> tuple[torch.Tensor | None, str]:
        if not path.exists():
            return None, "unknown"
        try:
            payload = _torch_load_any(path)
            depth_kind = "distance"
            if isinstance(payload, dict):
                depth = payload.get("depth_m", None)
                depth_kind = str(payload.get("depth_kind", "distance")).strip().lower()
                if depth_kind not in ("distance", "zdepth"):
                    depth_kind = "distance"
            else:
                depth = payload
            if not torch.is_tensor(depth):
                return None, "unknown"
            if depth.ndim == 3 and depth.shape[0] == 1:
                depth = depth[0]
            if depth.ndim != 2:
                return None, "unknown"
            depth = depth.to(torch.float32)
            valid = torch.isfinite(depth) & (depth > 0.0)
            if int(valid.sum().item()) <= 0:
                return None, "unknown"
            return depth.unsqueeze(0), depth_kind
        except Exception:
            return None, "unknown"

    @staticmethod
    def _distance_to_z_depth(depth_1hw: torch.Tensor, intrinsics_k3: torch.Tensor) -> torch.Tensor:
        if depth_1hw.ndim != 3 or depth_1hw.shape[0] != 1:
            raise ValueError(f"Expected depth shape (1,H,W), got {tuple(depth_1hw.shape)}")
        d = depth_1hw.to(torch.float32)
        h = int(d.shape[-2])
        w = int(d.shape[-1])
        k = intrinsics_k3.to(dtype=torch.float32, device=d.device)
        fx = k[0, 0]
        fy = k[1, 1]
        cx = k[0, 2]
        cy = k[1, 2]
        ys = torch.arange(h, device=d.device, dtype=torch.float32)
        xs = torch.arange(w, device=d.device, dtype=torch.float32)
        vv, uu = torch.meshgrid(ys, xs, indexing="ij")
        x = (uu - cx) / fx
        y = (vv - cy) / fy
        ray_z = 1.0 / torch.sqrt(x * x + y * y + 1.0).clamp(min=1e-8)
        z = d[0] * ray_z
        return z.unsqueeze(0)

    @staticmethod
    def _sanitize_pseudo_depth(
        depth_1hw: torch.Tensor,
        *,
        max_depth_m: float = DEFAULT_MAX_DEPTH_M,
        far_depth_invalid_m: float = 30.0,
    ) -> torch.Tensor:
        d = depth_1hw.to(torch.float32)
        valid = torch.isfinite(d) & (d > 0.0)
        if int(valid.sum().item()) <= 0:
            return d
        out = d.clone()
        if float(far_depth_invalid_m) > 0.0:
            valid = valid & (out <= float(far_depth_invalid_m))
            out = torch.where(valid, out, torch.zeros_like(out))
        out[valid] = out[valid].clamp(max=float(max_depth_m))
        return out

    def _get_or_create_pseudo_model(self) -> torch.nn.Module:
        if self._pseudo_model is None:
            dev = torch.device(self.pseudo_depth_device)
            self._pseudo_model = load_unik3d_model(
                backbone=self.pseudo_depth_backbone,
                pretrained=True,
                device=dev,
            )
            self._pseudo_model.eval()
            LOGGER.info(
                "Re10K pseudo-depth model loaded (split=%s, device=%s, backbone=%s)",
                self.split,
                str(dev),
                self.pseudo_depth_backbone,
            )
        return self._pseudo_model

    def _save_pseudo_depth_atomic(
        self,
        path: Path,
        depth_2d: torch.Tensor,
        scene: str,
        frame_idx: int,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.parent / f".tmp_{os.getpid()}_{int(time.time() * 1e6)}_{random.randint(0, 10_000_000)}.pt"
        payload = {
            "depth_m": depth_2d.to(torch.float16),
            "depth_kind": "distance",
            "scene": str(scene),
            "frame_idx": int(frame_idx),
        }
        torch.save(payload, tmp)
        os.replace(tmp, path)

    def _acquire_lock_or_wait_for_file(self, target: Path) -> tuple[bool, bool]:
        lock_dir = Path(str(target) + ".lock")
        start = time.time()
        while True:
            if target.exists():
                return False, True
            try:
                lock_dir.mkdir(parents=False, exist_ok=False)
                meta = lock_dir / "owner.txt"
                meta.write_text(f"pid={os.getpid()} time={time.time():.3f}\n", encoding="utf-8")
                return True, False
            except FileExistsError:
                try:
                    mtime = lock_dir.stat().st_mtime
                    if (time.time() - float(mtime)) > self.pseudo_lock_stale_sec:
                        for p in lock_dir.iterdir():
                            try:
                                p.unlink()
                            except Exception:
                                pass
                        lock_dir.rmdir()
                        continue
                except Exception:
                    pass
                if (time.time() - start) >= self.pseudo_lock_timeout_sec:
                    return False, False
                time.sleep(self.pseudo_wait_poll_sec)
            except Exception:
                return False, False

    def _release_lock(self, target: Path) -> None:
        lock_dir = Path(str(target) + ".lock")
        if not lock_dir.exists():
            return
        try:
            for p in lock_dir.iterdir():
                try:
                    p.unlink()
                except Exception:
                    pass
            lock_dir.rmdir()
        except Exception:
            pass

    def _get_pseudo_depth_for_frame(
        self,
        *,
        scene: str,
        frame_idx: int,
        rgb_u8: torch.Tensor,
        intrinsics_k3: torch.Tensor,
    ) -> torch.Tensor | None:
        path = self._pseudo_depth_path(scene, frame_idx)
        if path is None:
            return None
        depth, depth_kind = self._load_pseudo_depth(path)
        if depth is not None:
            if depth_kind != "zdepth":
                try:
                    depth = self._distance_to_z_depth(
                        self._sanitize_pseudo_depth(
                            depth,
                            max_depth_m=self.depth_max_m,
                            far_depth_invalid_m=self.pseudo_far_depth_invalid_m,
                        ),
                        intrinsics_k3=intrinsics_k3,
                    )
                except Exception:
                    return None
            else:
                depth = self._sanitize_pseudo_depth(
                    depth,
                    max_depth_m=self.depth_max_m,
                    far_depth_invalid_m=self.pseudo_far_depth_invalid_m,
                )
            return depth
        if not self.pseudo_depth_autogen:
            return None

        acquired, ready = self._acquire_lock_or_wait_for_file(path)
        if ready:
            depth, depth_kind = self._load_pseudo_depth(path)
            if depth is None:
                return None
            if depth_kind != "zdepth":
                try:
                    depth = self._distance_to_z_depth(
                        self._sanitize_pseudo_depth(
                            depth,
                            max_depth_m=self.depth_max_m,
                            far_depth_invalid_m=self.pseudo_far_depth_invalid_m,
                        ),
                        intrinsics_k3=intrinsics_k3,
                    )
                except Exception:
                    return None
            else:
                depth = self._sanitize_pseudo_depth(
                    depth,
                    max_depth_m=self.depth_max_m,
                    far_depth_invalid_m=self.pseudo_far_depth_invalid_m,
                )
            return depth
        if not acquired:
            depth, depth_kind = self._load_pseudo_depth(path)
            if depth is None:
                return None
            if depth_kind != "zdepth":
                try:
                    depth = self._distance_to_z_depth(
                        self._sanitize_pseudo_depth(
                            depth,
                            max_depth_m=self.depth_max_m,
                            far_depth_invalid_m=self.pseudo_far_depth_invalid_m,
                        ),
                        intrinsics_k3=intrinsics_k3,
                    )
                except Exception:
                    return None
            else:
                depth = self._sanitize_pseudo_depth(
                    depth,
                    max_depth_m=self.depth_max_m,
                    far_depth_invalid_m=self.pseudo_far_depth_invalid_m,
                )
            return depth

        try:
            depth, depth_kind = self._load_pseudo_depth(path)
            if depth is not None:
                if depth_kind != "zdepth":
                    try:
                        depth = self._distance_to_z_depth(
                            self._sanitize_pseudo_depth(
                                depth,
                                max_depth_m=self.depth_max_m,
                                far_depth_invalid_m=self.pseudo_far_depth_invalid_m,
                            ),
                            intrinsics_k3=intrinsics_k3,
                        )
                    except Exception:
                        return None
                else:
                    depth = self._sanitize_pseudo_depth(
                        depth,
                        max_depth_m=self.depth_max_m,
                        far_depth_invalid_m=self.pseudo_far_depth_invalid_m,
                    )
                return depth
            model = self._get_or_create_pseudo_model()
            out = infer_unik3d_pinhole(
                model,
                rgb_u8=rgb_u8.unsqueeze(0),
                intrinsics=intrinsics_k3.unsqueeze(0),
            )
            dist = out.get("distance", None) if isinstance(out, dict) else None
            if not torch.is_tensor(dist) or dist.ndim != 4 or dist.shape[1] != 1:
                return None
            dist_1hw = self._sanitize_pseudo_depth(
                dist[0:1, 0:1].detach().to(torch.float32).cpu()[0],
                max_depth_m=self.depth_max_m,
                far_depth_invalid_m=self.pseudo_far_depth_invalid_m,
            )
            valid = torch.isfinite(dist_1hw) & (dist_1hw > 0.0)
            if int(valid.sum().item()) <= 0:
                return None
            self._save_pseudo_depth_atomic(
                path,
                depth_2d=dist_1hw[0],
                scene=scene,
                frame_idx=frame_idx,
            )
            return self._distance_to_z_depth(dist_1hw, intrinsics_k3=intrinsics_k3.cpu())
        except Exception as e:
            LOGGER.warning(
                "Pseudo-depth generation failed scene=%s frame=%d: %s",
                str(scene),
                int(frame_idx),
                str(e),
            )
            return None
        finally:
            self._release_lock(path)

    def _candidate_target_indices(
        self,
        src_idx: int,
        num_frames: int,
        w2c_all: torch.Tensor,
        intr_all: torch.Tensor,
        h: int,
        w: int,
    ) -> list[int]:
        if num_frames < 2:
            return []
        centers = torch.linalg.inv(w2c_all)[:, :3, 3].to(torch.float32)
        sample_h = int(self.pair_overlap_sample_h)
        sample_w = int(self.pair_overlap_sample_w)
        return select_targets_for_source(
            src_idx=int(src_idx),
            candidate_indices=list(range(num_frames)),
            centers=centers,
            min_index_gap=int(self.min_frame_gap),
            max_index_gap=int(self.max_frame_gap),
            pair_max_translation_m=float(self.pair_max_translation_m),
            pair_min_overlap=float(self.pair_min_overlap),
            overlap_score_fn=lambda si, tj: float(
                0.5
                * (
                    project_overlap_ratio(
                        src_w2c=w2c_all[si],
                        tgt_w2c=w2c_all[tj],
                        src_k=intr_all[si],
                        tgt_k=intr_all[tj],
                        h=h,
                        w=w,
                        sample_h=sample_h,
                        sample_w=sample_w,
                    )
                    + project_overlap_ratio(
                        src_w2c=w2c_all[tj],
                        tgt_w2c=w2c_all[si],
                        src_k=intr_all[tj],
                        tgt_k=intr_all[si],
                        h=h,
                        w=w,
                        sample_h=sample_h,
                        sample_w=sample_w,
                    )
                )
            ),
        )

    def __iter__(self):
        chunks = list(self.chunks)
        order_rng = random.Random(self.seed + self.epoch)
        if self.shuffle_chunk and self.split == "train":
            order_rng.shuffle(chunks)
        pending_by_hw: dict[tuple[int, int], deque[Re10KPairSample]] = defaultdict(deque)

        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        total_shards = max(1, self.ddp_world_size * num_workers)
        shard_id = self.ddp_rank * num_workers + worker_id
        chunks = [chunk for i, chunk in enumerate(chunks) if i % total_shards == shard_id]

        for chunk_order_idx, chunk_path in enumerate(chunks):
            chunk = _torch_load_any(chunk_path)
            if not isinstance(chunk, list):
                continue
            examples = list(chunk)
            chunk_rng = random.Random(self.seed + self.epoch * 1000003 + chunk_order_idx)
            if self.shuffle_example and self.split == "train":
                chunk_rng.shuffle(examples)

            for example in examples:
                if not isinstance(example, dict):
                    continue
                if "cameras" not in example or "images" not in example:
                    continue
                poses = example["cameras"]
                images = example["images"]
                scene = str(example.get("key", "unknown"))
                if not torch.is_tensor(poses) or not isinstance(images, list):
                    continue
                if poses.ndim != 2 or poses.shape[1] != 18:
                    continue
                if len(images) != int(poses.shape[0]):
                    continue

                try:
                    src_probe = self._decode_image_u8(images[0])
                except Exception:
                    continue
                h, w = int(src_probe.shape[1]), int(src_probe.shape[2])
                w2c_all = self._convert_pose_row_to_w2c(poses)
                intr_all = self._convert_intrinsics_to_pixel(poses, h=h, w=w)
                src_indices = list(range(len(images)))
                if self.shuffle_example and self.split == "train":
                    chunk_rng.shuffle(src_indices)
                for src_idx in src_indices:
                    tgt_candidates = self._candidate_target_indices(
                        int(src_idx),
                        len(images),
                        w2c_all=w2c_all,
                        intr_all=intr_all,
                        h=h,
                        w=w,
                    )
                    if not tgt_candidates:
                        continue
                    tgt_idx = chunk_rng.choice(tgt_candidates)

                    try:
                        src_img = self._decode_image_u8(images[src_idx])
                        tgt_img = self._decode_image_u8(images[tgt_idx])
                    except Exception:
                        continue
                    if src_img.shape != tgt_img.shape:
                        continue
                    src_intr = intr_all[src_idx].clone()
                    tgt_intr = intr_all[tgt_idx].clone()
                    src_depth = self._get_pseudo_depth_for_frame(
                        scene=scene,
                        frame_idx=int(src_idx),
                        rgb_u8=src_img,
                        intrinsics_k3=intr_all[src_idx].to(torch.float32),
                    )
                    tgt_depth = self._get_pseudo_depth_for_frame(
                        scene=scene,
                        frame_idx=int(tgt_idx),
                        rgb_u8=tgt_img,
                        intrinsics_k3=intr_all[tgt_idx].to(torch.float32),
                    )
                    if self.pseudo_depth_root is not None and (
                        (not torch.is_tensor(src_depth)) or (not torch.is_tensor(tgt_depth))
                    ):
                        continue
                    if self.output_h is not None and self.output_w is not None:
                        oh, ow = int(src_img.shape[1]), int(src_img.shape[2])
                        if oh > 0 and ow > 0 and (oh != self.output_h or ow != self.output_w):
                            sx = float(self.output_w) / float(ow)
                            sy = float(self.output_h) / float(oh)
                            src_img = resize_rgb_u8_chw_high_quality(src_img, size=(self.output_h, self.output_w))
                            tgt_img = resize_rgb_u8_chw_high_quality(tgt_img, size=(self.output_h, self.output_w))
                            src_intr = resize_k3_align_corners_false(src_intr, sx=sx, sy=sy)
                            tgt_intr = resize_k3_align_corners_false(tgt_intr, sx=sx, sy=sy)
                            if torch.is_tensor(src_depth):
                                src_depth = (
                                    torch.nn.functional.interpolate(
                                        src_depth[None],
                                        size=(self.output_h, self.output_w),
                                        mode="bilinear",
                                        align_corners=False,
                                    )
                                    .squeeze(0)
                                    .to(torch.float32)
                                )
                            if torch.is_tensor(tgt_depth):
                                tgt_depth = (
                                    torch.nn.functional.interpolate(
                                        tgt_depth[None],
                                        size=(self.output_h, self.output_w),
                                        mode="bilinear",
                                        align_corners=False,
                                    )
                                    .squeeze(0)
                                    .to(torch.float32)
                                )

                    sample = Re10KPairSample(
                        src_rgb_u8=src_img,
                        tgt_rgb_u8=tgt_img,
                        src_w2c=w2c_all[src_idx],
                        tgt_w2c=w2c_all[tgt_idx],
                        src_intrinsics=src_intr,
                        tgt_intrinsics=tgt_intr,
                        src_idx=int(src_idx),
                        tgt_idx=int(tgt_idx),
                        scene=scene,
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
                        yield _pack_re10k_batch(packed)

        dropped = sum(len(bucket) for bucket in pending_by_hw.values())
        if dropped > 0 and self.split == "train" and self.batch_size_hint > 1:
            LOGGER.debug(
                "Dropped %d RE10K leftover samples that could not form a same-resolution batch of size %d.",
                int(dropped),
                int(self.batch_size_hint),
            )


def re10k_collate(batch: list[Re10KPairSample]) -> Re10KPairSample:
    return _pack_re10k_batch(batch)


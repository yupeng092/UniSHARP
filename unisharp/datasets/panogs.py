from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from typing import cast
import tarfile

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from unisharp import DEFAULT_MAX_DEPTH_M


MAX_DEPTH_M = DEFAULT_MAX_DEPTH_M

_PAIR_RECIPE_FIXED: tuple[str, bool] = ("c2w", True)

_PAIR_CONVENTIONS: tuple[str, ...] = ("c2w",)


def _torch_load_any(path: Path) -> object:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except (KeyError, tarfile.ReadError, EOFError, OSError, RuntimeError) as e:
        raise RuntimeError(f"torch.load failed (possibly incomplete/corrupted): {path}") from e


@dataclass(frozen=True)
class PanOGSSample:

    src_erp_rgb_u8: torch.Tensor
    tgt_erp_rgb_u8: torch.Tensor
    src_erp_depth_m: torch.Tensor
    tgt_erp_depth_m: torch.Tensor

    src_cube_rgb_u8: torch.Tensor
    tgt_cube_rgb_u8: torch.Tensor
    src_cube_depth_m: torch.Tensor
    tgt_cube_depth_m: torch.Tensor

    src_R: torch.Tensor
    src_t: torch.Tensor
    tgt_R: torch.Tensor
    tgt_t: torch.Tensor

    src_idx: int
    tgt_idx: int
    scene: str


def _load_erp_rgb_u8(path: Path) -> torch.Tensor:
    img = np.array(Image.open(path))
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected RGB image at {path}, got shape={img.shape}")
    return torch.from_numpy(img.astype(np.uint8)).permute(2, 0, 1).contiguous()


def _load_depth_png(path: Path) -> torch.Tensor:
    dep = np.array(Image.open(path))
    return torch.from_numpy(dep)


def _depth_to_meters(depth: torch.Tensor, max_depth_m: float = DEFAULT_MAX_DEPTH_M) -> torch.Tensor:
    depth_f = depth.to(torch.float32)
    maxv = float(depth_f.max().item()) if depth_f.numel() else 0.0
    if maxv > 200.0:
        depth_f = depth_f / 1000.0
    depth_f[~torch.isfinite(depth_f)] = 0.0
    return depth_f.clamp(min=0.0, max=float(max_depth_m))


class PanOGSDataset(Dataset[PanOGSSample]):

    def __init__(
        self,
        root: Path,
        index_manifest_path: Path | None = None,
        src_tgt_max_index_gap: int = 25,
        use_cubemap_supervision: bool = True,
        pair_sampling: bool = True,
        pair_max_translation_m: float = 0.5,
        pair_min_depth_overlap: float = 0.6,
        pair_overlap_face_w: int = 64,
        pair_overlap_margin: float = 1.05,
        pair_max_tries: int = 48,
        depth_max_m: float = DEFAULT_MAX_DEPTH_M,
    ) -> None:
        self.root = root
        self.src_tgt_max_index_gap = int(src_tgt_max_index_gap)
        self.use_cubemap_supervision = use_cubemap_supervision
        self.pair_sampling = bool(pair_sampling)
        self.pair_max_translation_m = float(pair_max_translation_m)
        self.pair_min_depth_overlap = float(pair_min_depth_overlap)
        self.pair_overlap_face_w = int(pair_overlap_face_w)
        self.pair_overlap_margin = float(pair_overlap_margin)
        self.pair_max_tries = int(pair_max_tries)
        self.depth_max_m = float(depth_max_m)
        self.index_manifest_path = Path(index_manifest_path) if index_manifest_path is not None else None

        self._pair_valid_tgts: dict[tuple[str, int], list[int]] = {}
        self._pair_overlap_cache: dict[tuple[str, int, int], float] = {}

        if not root.exists():
            raise FileNotFoundError(root)

        self.scenes = sorted([p for p in root.iterdir() if p.is_dir()])
        if not self.scenes:
            raise RuntimeError(f"No scene folders found in {root}")

        self._pose_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._meta_paths: dict[str, Path] = {}
        self._num_frames: dict[str, int] = {}
        self._available_frames: dict[str, list[int]] = {}

        if self.index_manifest_path is not None:
            if not self.index_manifest_path.exists():
                raise FileNotFoundError(self.index_manifest_path)
            valid_scenes: list[Path] = []
            for raw in self.index_manifest_path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line:
                    continue
                parts = line.split("|")
                scene_name = parts[0].strip()
                if not scene_name:
                    continue
                scene_dir = root / scene_name
                meta_path = scene_dir / "meta.pt"
                if not meta_path.exists():
                    continue
                if len(parts) >= 2:
                    try:
                        n_pose = int(parts[1])
                    except ValueError:
                        n_pose = 0
                else:
                    n_pose = 0
                if n_pose <= 0:
                    continue
                self._meta_paths[scene_name] = meta_path
                self._num_frames[scene_name] = n_pose
                self._available_frames[scene_name] = list(range(n_pose))
                valid_scenes.append(scene_dir)
            self.scenes = valid_scenes


        if not self._available_frames:

            valid_scenes = []
            for scene_i, scene_dir in enumerate(self.scenes):
                meta_path = scene_dir / "meta.pt"
                if not meta_path.exists():
                    continue

                ex = _torch_load_any(meta_path)
                cams = ex.get("cameras", None)
                if not isinstance(cams, torch.Tensor):
                    raise ValueError(f"meta.pt missing 'cameras' tensor in {scene_dir}")
                if cams.ndim != 3 or tuple(cams.shape[1:]) != (4, 4):
                    raise ValueError(f"Bad meta.pt cameras shape {tuple(cams.shape)} in {scene_dir}")
                n_pose = int(cams.shape[0])

                frames = list(range(n_pose))

                name = scene_dir.name
                self._meta_paths[name] = meta_path
                self._num_frames[name] = n_pose
                self._available_frames[name] = frames
                valid_scenes.append(scene_dir)

            self.scenes = valid_scenes

    def _get_pose(self, scene: str) -> tuple[np.ndarray, np.ndarray]:
        cached = self._pose_cache.get(scene)
        if cached is not None:
            return cached

        meta_path = self._meta_paths.get(scene)
        if meta_path is None:
            raise FileNotFoundError(f"meta.pt not indexed for scene={scene} under {self.root}")
        ex = _torch_load_any(meta_path)
        cams = ex.get("cameras", None)
        if not isinstance(cams, torch.Tensor):
            raise ValueError(f"meta.pt missing 'cameras' tensor for scene={scene}")
        cams = cams.to(torch.float32)
        if cams.ndim != 3 or tuple(cams.shape[1:]) != (4, 4):
            raise ValueError(f"Bad meta.pt cameras shape {tuple(cams.shape)} for scene={scene}")
        R = cams[:, :3, :3].cpu().numpy()
        t = cams[:, :3, 3].cpu().numpy()
        out = (R, t)
        self._pose_cache[scene] = out
        return out

    def __len__(self) -> int:
        return len(self._index)

    def _sample_target(self, scene: str, src_idx: int) -> int:
        frames = self._available_frames[scene]
        if len(frames) <= 1:
            return src_idx
        effective_gap = self.src_tgt_max_index_gap
        candidates = [i for i in frames if i != src_idx and abs(i - src_idx) <= effective_gap]
        if not candidates:
            return src_idx
        j = int(torch.randint(low=0, high=len(candidates), size=(1,)).item())
        return int(candidates[j])

    def _candidate_targets_by_translation(self, scene: str, src_idx: int) -> list[int]:
        frames = self._available_frames[scene]
        if len(frames) <= 1:
            return []
        R_np, t_np = self._get_pose(scene)
        if not (0 <= src_idx < len(t_np) and 0 <= src_idx < len(R_np)):
            return []
        th = float(self.pair_max_translation_m)

        def _cam_center_from(R: np.ndarray, t: np.ndarray, conv: str) -> np.ndarray:
            if conv in ("c2w", "w2c_t_camcenter"):
                return t
            if conv == "w2c":
                return -(R.transpose(0, 2, 1) @ t[..., None])[..., 0]
            if conv == "c2w_t_w2c":
                return -(R @ t[..., None])[..., 0]
            raise ValueError(conv)

        def _min_dist(idxs: np.ndarray) -> np.ndarray:
            R_sub = R_np[idxs].astype(np.float32)
            t_sub = t_np[idxs].astype(np.float32)
            R_src = R_np[int(src_idx) : int(src_idx) + 1].astype(np.float32)
            t_src = t_np[int(src_idx) : int(src_idx) + 1].astype(np.float32)
            d_min = None
            for conv in _PAIR_CONVENTIONS:
                C_src = _cam_center_from(R_src, t_src, conv)[0]
                C_sub = _cam_center_from(R_sub, t_sub, conv)
                d = np.linalg.norm(C_sub - C_src[None, :], axis=1)
                d_min = d if (d_min is None) else np.minimum(d_min, d)
            assert d_min is not None
            return d_min

        effective_gap = self.src_tgt_max_index_gap
        cand0 = np.array([i for i in frames if i != src_idx and abs(i - src_idx) <= effective_gap], dtype=np.int64)
        if cand0.size > 0:
            d0 = _min_dist(cand0)
            ok0 = cand0[d0 < th]
            if ok0.size > 0:
                return [int(x) for x in ok0.tolist()]
        return []

    def _resize_cube_depth(self, depth: torch.Tensor, face_w: int) -> torch.Tensor:
        if depth.ndim != 4 or depth.shape[0] != 6 or depth.shape[-1] != 1:
            raise ValueError(f"Expected cube depth shape (6,H,W,1), got {tuple(depth.shape)}")
        H = int(depth.shape[1])
        W = int(depth.shape[2])
        if H == face_w and W == face_w:
            return depth.to(dtype=torch.float32)
        import torch.nn.functional as F

        x = depth.permute(0, 3, 1, 2).to(dtype=torch.float32)
        x = F.interpolate(x, size=(face_w, face_w), mode="bilinear", align_corners=False)
        return x.permute(0, 2, 3, 1).contiguous()

    @staticmethod
    def _cubemap_z_depth_to_distance(depth: torch.Tensor) -> torch.Tensor:
        if depth.ndim != 4 or depth.shape[0] != 6 or depth.shape[-1] != 1:
            raise ValueError(f"Expected cube depth shape (6,H,W,1), got {tuple(depth.shape)}")
        from unisharp.utils.pano import get_pinhole_intrinsics_4x4

        h = int(depth.shape[1])
        w = int(depth.shape[2])
        if h != w:
            raise ValueError(f"Expected square cubemap faces, got {(h, w)}")
        depth_61hw = depth.permute(0, 3, 1, 2).to(dtype=torch.float32).contiguous()
        intr = get_pinhole_intrinsics_4x4(w).to(device=depth_61hw.device, dtype=depth_61hw.dtype)
        ys = torch.arange(h, device=depth_61hw.device, dtype=depth_61hw.dtype)
        xs = torch.arange(w, device=depth_61hw.device, dtype=depth_61hw.dtype)
        vv, uu = torch.meshgrid(ys, xs, indexing="ij")
        x = (uu - intr[0, 2]) / intr[0, 0].clamp(min=1e-8)
        y = (vv - intr[1, 2]) / intr[1, 1].clamp(min=1e-8)
        ray_z = 1.0 / torch.sqrt(x * x + y * y + 1.0).clamp(min=1e-8)
        dist = depth_61hw / ray_z.view(1, 1, h, w).clamp(min=1e-8)
        valid = torch.isfinite(dist) & (dist > 0.0)
        dist = torch.where(valid, dist, torch.zeros_like(dist))
        return dist.permute(0, 2, 3, 1).contiguous()

    def _pair_depth_overlap_score(
        self,
        *,
        src_R: torch.Tensor,
        src_t: torch.Tensor,
        tgt_R: torch.Tensor,
        tgt_t: torch.Tensor,
        src_cube_depth_m: torch.Tensor,
        tgt_cube_depth_m: torch.Tensor,
    ) -> float:
        from unisharp.utils.camera_projection import build_extrinsics_w2c, view_frustum_mask_cubemap_union  # noqa: WPS433

        device = torch.device("cpu")
        src_R = src_R.to(device=device, dtype=torch.float32)
        src_t = src_t.to(device=device, dtype=torch.float32)
        tgt_R = tgt_R.to(device=device, dtype=torch.float32)
        tgt_t = tgt_t.to(device=device, dtype=torch.float32)

        face_w = int(self.pair_overlap_face_w)
        margin = float(self.pair_overlap_margin)
        src_d = self._cubemap_z_depth_to_distance(self._resize_cube_depth(src_cube_depth_m.to(device=device), face_w=face_w))
        tgt_d = self._cubemap_z_depth_to_distance(self._resize_cube_depth(tgt_cube_depth_m.to(device=device), face_w=face_w))

        def _score_one(recipe: tuple[str, bool]) -> float:
            pose_conv, flip_yz = recipe
            extr_src = build_extrinsics_w2c(src_R, src_t, pose_conv)
            extr_tgt = build_extrinsics_w2c(tgt_R, tgt_t, pose_conv)

            with torch.autocast(device_type="cpu", enabled=False):
                c2w_src = torch.linalg.inv(extr_src)
                c2w_tgt = torch.linalg.inv(extr_tgt)
                if bool(flip_yz):
                    D = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=torch.float32, device=device))
                    c2w_src = c2w_src @ D
                    c2w_tgt = c2w_tgt @ D
                ref_inv = torch.linalg.inv(c2w_src)
                c2w_src = ref_inv @ c2w_src
                c2w_tgt = ref_inv @ c2w_tgt
                extr_src_n = torch.linalg.inv(c2w_src)
                extr_tgt_n = torch.linalg.inv(c2w_tgt)

            m_tgt_in_src = view_frustum_mask_cubemap_union(
                depth_novel=tgt_d,
                extr_novel_w2c=extr_tgt_n,
                extr_source_w2c=extr_src_n,
                face_w=face_w,
                margin=margin,
            )
            m_src_in_tgt = view_frustum_mask_cubemap_union(
                depth_novel=src_d,
                extr_novel_w2c=extr_src_n,
                extr_source_w2c=extr_tgt_n,
                face_w=face_w,
                margin=margin,
            )
            tgt_valid = torch.isfinite(tgt_d[..., 0]) & (tgt_d[..., 0] > 0.0)
            src_valid = torch.isfinite(src_d[..., 0]) & (src_d[..., 0] > 0.0)
            denom_t = float(tgt_valid.sum().item())
            denom_s = float(src_valid.sum().item())
            if denom_t < 10 or denom_s < 10:
                return 0.0
            a = float((m_tgt_in_src & tgt_valid).sum().item()) / denom_t
            b = float((m_src_in_tgt & src_valid).sum().item()) / denom_s
            return 0.5 * (a + b)

        return _score_one(_PAIR_RECIPE_FIXED)

    def __getitem__(self, idx: int) -> PanOGSSample:
        src_erp: torch.Tensor | None = None
        tgt_erp: torch.Tensor | None = None
        src_dep: torch.Tensor | None = None
        tgt_dep: torch.Tensor | None = None
        src_cube: torch.Tensor | None = None
        tgt_cube: torch.Tensor | None = None
        src_cdep: torch.Tensor | None = None
        tgt_cdep: torch.Tensor | None = None
        last_err: Exception | None = None

        max_outer = 16
        for outer in range(max_outer):
            scene, src_idx = self._index[int(idx) % len(self._index)]
            scene_dir = self.root / scene
            tgt_idx = self._sample_target(scene, src_idx)

            max_retries = 8
            ok = False
            for _ in range(max_retries):
                try:
                    if src_erp is None:
                        src_erp = _load_erp_rgb_u8(scene_dir / "pano" / f"{src_idx:05d}.png")
                        src_dep = _depth_to_meters(
                            _load_depth_png(scene_dir / "pano_depth" / f"{src_idx:05d}.png"),
                            max_depth_m=self.depth_max_m,
                        )
                        if self.use_cubemap_supervision:
                            src_cube_any = _torch_load_any(scene_dir / "cubemaps" / f"{src_idx:05d}.torch")
                            src_cdep_any = _torch_load_any(scene_dir / "cubemaps_depth" / f"{src_idx:05d}.torch")
                            if not all(isinstance(x, torch.Tensor) for x in [src_cube_any, src_cdep_any]):
                                raise RuntimeError("Bad .torch payload for src (expected Tensor).")
                            src_cube = cast(torch.Tensor, src_cube_any)
                            src_cdep = cast(torch.Tensor, src_cdep_any).to(torch.float32).clamp(min=0.0, max=self.depth_max_m)
                        else:
                            src_cube = torch.zeros((6, 256, 256, 3), dtype=torch.uint8)
                            src_cdep = torch.zeros((6, 256, 256, 1), dtype=torch.float32)

                    candidates: list[int] = []
                    if self.pair_sampling and self.use_cubemap_supervision:
                        key = (scene, int(src_idx))
                        cached = self._pair_valid_tgts.get(key)
                        if cached:
                            candidates = list(cached)
                        else:
                            candidates = self._candidate_targets_by_translation(scene, int(src_idx))
                    if not candidates:
                        candidates = [int(tgt_idx)]

                    tried: set[int] = set()
                    found = False
                    max_try = (
                        1
                        if (not self.pair_sampling or not self.use_cubemap_supervision)
                        else max(1, self.pair_max_tries)
                    )
                    for _try in range(max_try):
                        pool = [
                            c
                            for c in candidates
                            if int(c) not in tried and int(c) != int(src_idx)
                        ]
                        if not pool:
                            break
                        j = int(torch.randint(0, len(pool), (1,)).item())
                        tgt_idx = int(pool[j])
                        tried.add(int(tgt_idx))

                        if self.use_cubemap_supervision:
                            tgt_cdep_any = _torch_load_any(scene_dir / "cubemaps_depth" / f"{tgt_idx:05d}.torch")
                            if not isinstance(tgt_cdep_any, torch.Tensor):
                                raise RuntimeError("Bad .torch payload for tgt depth (expected Tensor).")
                            tgt_cdep = cast(torch.Tensor, tgt_cdep_any).to(torch.float32).clamp(min=0.0, max=self.depth_max_m)
                        else:
                            tgt_cdep = torch.zeros((6, 256, 256, 1), dtype=torch.float32)

                        if self.pair_sampling and self.use_cubemap_supervision:
                            k = (scene, int(src_idx), int(tgt_idx))
                            score = self._pair_overlap_cache.get(k)
                            if score is None:
                                R_np, t_np = self._get_pose(scene)
                                src_R = torch.from_numpy(R_np[int(src_idx)])
                                src_t = torch.from_numpy(t_np[int(src_idx)])
                                tgt_R = torch.from_numpy(R_np[int(tgt_idx)])
                                tgt_t = torch.from_numpy(t_np[int(tgt_idx)])
                                score = self._pair_depth_overlap_score(
                                    src_R=src_R,
                                    src_t=src_t,
                                    tgt_R=tgt_R,
                                    tgt_t=tgt_t,
                                    src_cube_depth_m=cast(torch.Tensor, src_cdep),
                                    tgt_cube_depth_m=cast(torch.Tensor, tgt_cdep),
                                )
                                self._pair_overlap_cache[k] = float(score)
                            if float(score) < float(self.pair_min_depth_overlap):
                                continue
                            kk = (scene, int(src_idx))
                            self._pair_valid_tgts.setdefault(kk, []).append(int(tgt_idx))

                        tgt_erp = _load_erp_rgb_u8(scene_dir / "pano" / f"{tgt_idx:05d}.png")
                        tgt_dep = _depth_to_meters(
                            _load_depth_png(scene_dir / "pano_depth" / f"{tgt_idx:05d}.png"),
                            max_depth_m=self.depth_max_m,
                        )
                        if self.use_cubemap_supervision:
                            tgt_cube_any = _torch_load_any(scene_dir / "cubemaps" / f"{tgt_idx:05d}.torch")
                            if not isinstance(tgt_cube_any, torch.Tensor):
                                raise RuntimeError("Bad .torch payload for tgt RGB cubemap (expected Tensor).")
                            tgt_cube = cast(torch.Tensor, tgt_cube_any)
                        else:
                            tgt_cube = torch.zeros((6, 256, 256, 3), dtype=torch.uint8)

                        found = True
                        break

                    if not found:
                        raise RuntimeError(
                            f"No valid tgt found for scene={scene} src={src_idx} within constraints "
                            f"(trans<{self.pair_max_translation_m}m, overlap>{self.pair_min_depth_overlap})."
                        )

                    ok = True
                    break
                except (FileNotFoundError, RuntimeError, EOFError, KeyError, tarfile.ReadError, OSError) as e:
                    last_err = e
                    frames = self._available_frames.get(scene, [])
                    if not frames:
                        break
                    src_idx = int(frames[int(torch.randint(0, len(frames), (1,)).item())])
                    tgt_idx = self._sample_target(scene, src_idx)
                    src_erp = None
                    src_dep = None
                    src_cube = None
                    src_cdep = None

            if ok:
                break

            idx = int(idx) + 9973 + outer * 13
        else:
            raise RuntimeError(f"PanOGS __getitem__ failed after retries. last_err={last_err}")

        assert src_erp is not None and tgt_erp is not None
        assert src_dep is not None and tgt_dep is not None
        assert src_cube is not None and tgt_cube is not None
        assert src_cdep is not None and tgt_cdep is not None

        src_dep = src_dep.to(torch.float32).unsqueeze(0)
        tgt_dep = tgt_dep.to(torch.float32).unsqueeze(0)

        R_np, t_np = self._get_pose(scene)
        src_R = torch.from_numpy(R_np[src_idx])
        src_t = torch.from_numpy(t_np[src_idx])
        tgt_R = torch.from_numpy(R_np[tgt_idx])
        tgt_t = torch.from_numpy(t_np[tgt_idx])

        return PanOGSSample(
            src_erp_rgb_u8=src_erp,
            tgt_erp_rgb_u8=tgt_erp,
            src_erp_depth_m=src_dep,
            tgt_erp_depth_m=tgt_dep,
            src_cube_rgb_u8=src_cube,
            tgt_cube_rgb_u8=tgt_cube,
            src_cube_depth_m=src_cdep,
            tgt_cube_depth_m=tgt_cdep,
            src_R=src_R,
            src_t=src_t,
            tgt_R=tgt_R,
            tgt_t=tgt_t,
            src_idx=src_idx,
            tgt_idx=tgt_idx,
            scene=scene,
        )


def panogs_collate(batch: list[PanOGSSample]) -> PanOGSSample:
    def stack(xs):
        if isinstance(xs[0], torch.Tensor):
            return torch.stack(xs, dim=0)
        return xs

    return PanOGSSample(
        src_erp_rgb_u8=stack([b.src_erp_rgb_u8 for b in batch]),
        tgt_erp_rgb_u8=stack([b.tgt_erp_rgb_u8 for b in batch]),
        src_erp_depth_m=stack([b.src_erp_depth_m for b in batch]),
        tgt_erp_depth_m=stack([b.tgt_erp_depth_m for b in batch]),
        src_cube_rgb_u8=stack([b.src_cube_rgb_u8 for b in batch]),
        tgt_cube_rgb_u8=stack([b.tgt_cube_rgb_u8 for b in batch]),
        src_cube_depth_m=stack([b.src_cube_depth_m for b in batch]),
        tgt_cube_depth_m=stack([b.tgt_cube_depth_m for b in batch]),
        src_R=stack([b.src_R for b in batch]),
        src_t=stack([b.src_t for b in batch]),
        tgt_R=stack([b.tgt_R for b in batch]),
        tgt_t=stack([b.tgt_t for b in batch]),
        src_idx=[b.src_idx for b in batch],  # type: ignore[arg-type]
        tgt_idx=[b.tgt_idx for b in batch],  # type: ignore[arg-type]
        scene=[b.scene for b in batch],  # type: ignore[arg-type]
    )


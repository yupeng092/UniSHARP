#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import logging
import math
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from unisharp.datasets.panogs import panogs_collate  # noqa: E402
from unisharp.datasets.scannetpp_fisheye import ScannetppFisheyeDataset  # noqa: E402
from unisharp.datasets.sim_panorama import _EquirecToCube, SimPanoramaDataset  # noqa: E402
from unisharp.datasets.wildrgbd import WildRGBDDataset  # noqa: E402
from unisharp.losses import UnisharpLoss, UnisharpLossWeights  # noqa: E402
from unisharp.models.unisharp_feature import UnisharpFeatureConfig, UnisharpFeatureModel  # noqa: E402
from unisharp.utils.color_space import linearRGB2sRGB  # noqa: E402
from unisharp import DEFAULT_MAX_DEPTH_M  # noqa: E402
from unisharp.utils.io import save_image  # noqa: E402
from unisharp.utils.metrics import (  # noqa: E402
    MetricsCalculator,
    compute_masked_rgb_metrics,
    default_metric_mask_cache_dir,
    metric_mask_from_pinhole_batch,
)
from unisharp.utils.vis import colorize_alpha, colorize_scalar_map  # noqa: E402
from unisharp.validation.io_common import (  # noqa: E402
    decode_rgb_u8 as _decode_rgb_u8,
    distance_to_z_depth_pinhole as _distance_to_z_depth_pinhole,
    colmap_image_dir as _colmap_image_dir,
    colmap_scene_roots as _colmap_scene_roots,
    load_colmap_entries as _load_colmap_entries,
    load_hm3d_pose as _load_hm3d_pose,
    load_scaled_colmap_entries as _load_scaled_colmap_entries,
    load_png_depth_m as _load_png_depth_m,
    load_png_rgb_u8 as _load_png_rgb_u8,
    load_validation_pseudo_depth as _load_validation_pseudo_depth,
    load_validation_pseudo_distance as _load_validation_pseudo_distance,
    nerf_c2w_to_opencv_c2w as _nerf_c2w_to_opencv_c2w,
    normalize_depth_kind as _normalize_depth_kind,
    read_manifest_lines as _read_manifest_lines,
    resolve_replica_test_root as _resolve_replica_test_root,
    resize_k3_align_corners_false as _resize_k3_align_corners_false,
    torch_load_any as _torch_load_any,
    wild_validation_roots as _wild_validation_roots,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)

ValidationTag = str | list[str]
ValidationItem = tuple[str, Any, ValidationTag, str]

METRIC_FIELDS = ["psnr", "ssim", "lpips"]


def _configure_torchhub_cache() -> Path:
    torchhub_dir = REPO_ROOT / "checkpoints" / "torchhub"
    torchhub_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(torchhub_dir)
    torch.hub.set_dir(str(torchhub_dir))
    return torchhub_dir


def _training_config_for_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
    except Exception:
        ckpt = None
    if isinstance(ckpt, dict):
        cfg = ckpt.get("config", None)
        if isinstance(cfg, dict):
            payload.update(cfg)
    config_path = Path(checkpoint_path).parent / "config.json"
    if not config_path.exists():
        return payload
    try:
        json_payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return payload
    if isinstance(json_payload, dict):
        payload.update(json_payload)
    return payload


def _fill_arg_from_config(args: argparse.Namespace, attr: str, config: dict[str, Any], key: str, default: float) -> None:
    if getattr(args, attr, None) is not None:
        return
    value = config.get(key, default)
    try:
        setattr(args, attr, float(value))
    except Exception:
        setattr(args, attr, float(default))


def _fill_int_arg_from_config(args: argparse.Namespace, attr: str, config: dict[str, Any], key: str, default: int) -> None:
    if getattr(args, attr, None) is not None:
        return
    value = config.get(key, default)
    try:
        setattr(args, attr, int(value))
    except Exception:
        setattr(args, attr, int(default))


def _apply_training_depth_config_defaults(args: argparse.Namespace) -> None:
    config = _training_config_for_checkpoint(Path(args.checkpoint))
    _fill_arg_from_config(args, "max_depth_m", config, "max_depth_m", DEFAULT_MAX_DEPTH_M)
    _fill_arg_from_config(args, "sim_far_depth_invalid_m", config, "sim_far_depth_invalid_m", 30.0)
    _fill_arg_from_config(args, "sim_far_depth_invalid_max_frac", config, "sim_far_depth_invalid_max_frac", 1.0)
    _fill_arg_from_config(args, "re10k_pseudo_far_depth_invalid_m", config, "re10k_pseudo_far_depth_invalid_m", 30.0)
    _fill_arg_from_config(args, "scanetpp_fisheye_far_depth_invalid_m", config, "scanetpp_fisheye_far_depth_invalid_m", 30.0)
    _fill_arg_from_config(args, "low_pass_filter_eps", config, "render_low_pass_filter_eps", 1e-2)


def _append_metrics_row(csv_path: Path, row: dict[str, float]) -> None:
    fieldnames = list(METRIC_FIELDS)
    row_out = {k: row.get(k, float("nan")) for k in fieldnames}
    if csv_path.exists():
        try:
            with csv_path.open("r", newline="") as f:
                reader = csv.reader(f)
                existing_header = next(reader, [])
            if existing_header:
                if fieldnames != existing_header:
                    with csv_path.open("r", newline="") as f:
                        old_rows = list(csv.DictReader(f))
                    with csv_path.open("w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        for r in old_rows:
                            writer.writerow({k: r.get(k, float("nan")) for k in fieldnames})
        except Exception:
            pass
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row_out)


def _append_sample_metrics_row(csv_path: Path, group_key: str, tag: str, row: dict[str, float]) -> None:
    fieldnames = ["group", "tag", *METRIC_FIELDS]
    row_out = {"group": group_key, "tag": tag, **{k: row.get(k, float("nan")) for k in METRIC_FIELDS}}
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row_out)


def _feature_config_from_checkpoint(checkpoint_path: Path, ckpt: dict[str, Any]) -> UnisharpFeatureConfig:
    cfg = UnisharpFeatureConfig()
    merged: dict[str, Any] = {}
    cfg_payload = ckpt.get("config", {})
    if isinstance(cfg_payload, dict):
        merged.update(cfg_payload)
    for key in cfg.__dict__.keys():
        if key in ckpt:
            merged[key] = ckpt[key]
    config_path = Path(checkpoint_path).parent / "config.json"
    if config_path.exists():
        try:
            sidecar = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            sidecar = None
        if isinstance(sidecar, dict):
            merged.update({k: v for k, v in sidecar.items() if k in cfg.__dict__})
    for k in cfg.__dict__.keys():
        if k in merged:
            setattr(cfg, k, merged[k])
    return cfg


def _load_model(checkpoint_path: Path, device: torch.device) -> tuple[UnisharpFeatureModel, int]:
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError(f"Expected feature-only checkpoint dict, got {type(ckpt)} from {checkpoint_path}")

    cfg = _feature_config_from_checkpoint(checkpoint_path, ckpt)
    model = UnisharpFeatureModel(cfg).to(device)
    model.load_from_checkpoint(str(checkpoint_path), strict=True)
    model.eval()
    return model, int(ckpt.get("step", 0))


def _build_trainer(model: UnisharpFeatureModel, device: torch.device, args: argparse.Namespace) -> Any:
    from unisharp.cli.unified_trainer import UnifiedTrainer
    zero_w = UnisharpLossWeights(
        lambda_color=0.0,
        lambda_alpha=0.0,
        lambda_percep=0.0,
        lambda_depth=0.0,
        lambda_tv=0.0,
        lambda_grad=0.0,
        lambda_grad_img=0.0,
        lambda_delta=0.0,
        lambda_splat=0.0,
    )
    loss_fn = UnisharpLoss(zero_w).to(device)
    max_depth_m = float(getattr(args, "max_depth_m", getattr(model.config, "max_distance_m", DEFAULT_MAX_DEPTH_M)))
    loss_fn.SUPERVISION_MAX_DEPTH_M = max_depth_m
    from unisharp.utils.gsplat import GSplatRenderer

    renderer = GSplatRenderer(
        color_space="sRGB",
        background_color="black",
        low_pass_filter_eps=float(getattr(args, "low_pass_filter_eps", 1e-2)),
    ).to(device)
    return UnifiedTrainer(
        model=model,
        renderer=renderer,
        loss_fn=loss_fn,
        device=device,
        enable_tgt_unik3d_vis=False,
        max_depth_m=max_depth_m,
        sim_far_depth_invalid_m=float(getattr(args, "sim_far_depth_invalid_m", 30.0)),
        re10k_pseudo_far_depth_invalid_m=float(getattr(args, "re10k_pseudo_far_depth_invalid_m", 30.0)),
        scanetpp_fisheye_far_depth_invalid_m=float(getattr(args, "scanetpp_fisheye_far_depth_invalid_m", 30.0)),
    )


def _make_pinhole_batch(
    *,
    src_img: torch.Tensor,
    tgt_img: torch.Tensor,
    src_w2c: torch.Tensor,
    tgt_w2c: torch.Tensor,
    src_k: torch.Tensor,
    tgt_k: torch.Tensor,
    scene: str,
    src_idx: int | list[int],
    tgt_idx: int | list[int],
    src_depth: torch.Tensor | None = None,
    tgt_depth: torch.Tensor | None = None,
    src_img_orig: torch.Tensor | None = None,
    tgt_img_orig: torch.Tensor | None = None,
    src_k_orig: torch.Tensor | None = None,
    tgt_k_orig: torch.Tensor | None = None,
    src_depth_orig: torch.Tensor | None = None,
    tgt_depth_orig: torch.Tensor | None = None,
) -> SimpleNamespace:
    batch_size = int(tgt_img.shape[0]) if torch.is_tensor(tgt_img) and tgt_img.ndim == 4 else 1
    scene_values = [scene] * batch_size if isinstance(scene, str) else list(scene)
    src_idx_values = [int(src_idx)] * batch_size if isinstance(src_idx, int) else [int(x) for x in src_idx]
    tgt_idx_values = [int(tgt_idx)] if isinstance(tgt_idx, int) else [int(x) for x in tgt_idx]
    return SimpleNamespace(
        src_rgb_u8=src_img,
        tgt_rgb_u8=tgt_img,
        src_depth_m=src_depth,
        tgt_depth_m=tgt_depth,
        src_rgb_u8_orig=(src_img if src_img_orig is None else src_img_orig),
        tgt_rgb_u8_orig=(tgt_img if tgt_img_orig is None else tgt_img_orig),
        src_depth_m_orig=(src_depth if src_depth_orig is None else src_depth_orig),
        tgt_depth_m_orig=(tgt_depth if tgt_depth_orig is None else tgt_depth_orig),
        src_w2c=src_w2c,
        tgt_w2c=tgt_w2c,
        src_intrinsics=src_k,
        tgt_intrinsics=tgt_k,
        src_intrinsics_orig=(src_k if src_k_orig is None else src_k_orig),
        tgt_intrinsics_orig=(tgt_k if tgt_k_orig is None else tgt_k_orig),
        scene=scene_values,
        src_idx=torch.tensor(src_idx_values, dtype=torch.long),
        tgt_idx=torch.tensor(tgt_idx_values, dtype=torch.long),
        share_src_forward=True,
        collect_all_vis=True,
    )


def _make_scanetpp_fisheye_batch(
    *,
    scene: str,
    src_pos: int,
    tgt_positions: list[int],
    src_frame: dict[str, Any],
    tgt_frames: list[dict[str, Any]],
    src_loaded: dict[str, torch.Tensor],
    tgt_loaded: list[dict[str, torch.Tensor]],
) -> SimpleNamespace:
    n = int(len(tgt_positions))
    src_rgb = src_loaded["rgb_u8"].unsqueeze(0).repeat(n, 1, 1, 1)
    src_depth = src_loaded["depth_m"].unsqueeze(0).repeat(n, 1, 1, 1)
    src_mask = src_loaded["valid_mask"].unsqueeze(0).repeat(n, 1, 1, 1)
    src_w2c = src_frame["w2c"].to(torch.float32).unsqueeze(0).repeat(n, 1, 1)
    src_cam = src_loaded["camera_params"].to(torch.float32).unsqueeze(0).repeat(n, 1)
    return SimpleNamespace(
        src_rgb_u8=src_rgb,
        tgt_rgb_u8=torch.stack([item["rgb_u8"] for item in tgt_loaded], dim=0),
        src_depth_m=src_depth,
        tgt_depth_m=torch.stack([item["depth_m"] for item in tgt_loaded], dim=0),
        src_valid_mask=src_mask,
        tgt_valid_mask=torch.stack([item["valid_mask"] for item in tgt_loaded], dim=0),
        src_w2c=src_w2c,
        tgt_w2c=torch.stack([frame["w2c"].to(torch.float32) for frame in tgt_frames], dim=0),
        src_camera_params=src_cam,
        tgt_camera_params=torch.stack([item["camera_params"].to(torch.float32) for item in tgt_loaded], dim=0),
        src_idx=torch.full((n,), int(src_pos), dtype=torch.long),
        tgt_idx=torch.tensor([int(x) for x in tgt_positions], dtype=torch.long),
        scene=[str(scene)] * n,
        camera_model="fisheye624",
        share_src_forward=True,
        collect_all_vis=True,
    )



@dataclass
class _PinholeTargetAdapter:
    idx: int
    img: torch.Tensor
    w2c: torch.Tensor
    k: torch.Tensor
    depth: torch.Tensor | None = None


@dataclass
class _PinholeGroupAdapter:
    scene: str
    group_key: str
    src_idx: int
    src_img: torch.Tensor
    src_w2c: torch.Tensor
    src_k: torch.Tensor
    tgt_indices: list[int]
    load_target: Callable[[int], _PinholeTargetAdapter | None]
    src_depth: torch.Tensor | None = None


def _iter_manifest_parts(
    args: argparse.Namespace,
    *,
    expected_parts: int,
) -> Iterator[tuple[int, list[str]]]:
    manifest_in = _read_manifest_lines(getattr(args, "manifest_file", None), max_lines=_manifest_max_groups(args))
    for group_idx, raw in enumerate(manifest_in):
        parts = raw.split("|")
        if len(parts) == int(expected_parts):
            yield group_idx, parts


def _yield_pinhole_group_batches(
    dataset: str,
    adapter: _PinholeGroupAdapter,
    args: argparse.Namespace,
) -> Iterator[ValidationItem]:
    batch_size = max(1, int(getattr(args, "validation_batch_size", 1)))
    pending: list[_PinholeTargetAdapter] = []

    def _flush(targets: list[_PinholeTargetAdapter]) -> Iterator[ValidationItem]:
        if not targets:
            return
        n = len(targets)
        src_img_orig = adapter.src_img.clone()
        if n > 1:
            src_img_orig = src_img_orig.repeat(n, 1, 1, 1)
        tgt_img_orig = torch.cat([t.img.clone() for t in targets], dim=0)
        src_k_orig = adapter.src_k.clone()
        if n > 1:
            src_k_orig = src_k_orig.repeat(n, 1, 1)
        tgt_k_orig = torch.cat([t.k.clone() for t in targets], dim=0)
        src_w2c = adapter.src_w2c
        if n > 1:
            src_w2c = src_w2c.repeat(n, 1, 1)
        tgt_w2c = torch.cat([t.w2c for t in targets], dim=0)
        src_depth_orig = None if adapter.src_depth is None else adapter.src_depth.clone()
        if src_depth_orig is not None and n > 1:
            src_depth_orig = src_depth_orig.repeat(n, 1, 1, 1)
        tgt_depth_values = [t.depth for t in targets]
        tgt_depth_orig = None
        if all(torch.is_tensor(d) for d in tgt_depth_values):
            tgt_depth_orig = torch.cat([d for d in tgt_depth_values if torch.is_tensor(d)], dim=0)
        batch = _make_pinhole_batch(
            src_img=src_img_orig,
            tgt_img=tgt_img_orig,
            src_w2c=src_w2c,
            tgt_w2c=tgt_w2c,
            src_k=src_k_orig,
            tgt_k=tgt_k_orig,
            scene=adapter.scene,
            src_idx=[adapter.src_idx] * n,
            tgt_idx=[int(t.idx) for t in targets],
            src_depth=src_depth_orig,
            tgt_depth=tgt_depth_orig,
            src_img_orig=src_img_orig,
            tgt_img_orig=tgt_img_orig,
            src_k_orig=src_k_orig,
            tgt_k_orig=tgt_k_orig,
            src_depth_orig=src_depth_orig,
            tgt_depth_orig=tgt_depth_orig,
        )
        tags = [f"{adapter.group_key}_t{int(t.idx):05d}" for t in targets]
        yield (dataset, batch, tags[0] if len(tags) == 1 else tags, adapter.group_key)

    for tgt_idx in adapter.tgt_indices:
        tgt = adapter.load_target(int(tgt_idx))
        if tgt is None:
            continue
        pending.append(tgt)
        if len(pending) >= batch_size:
            yield from _flush(pending)
            pending = []
    if pending:
        yield from _flush(pending)


def _yield_panogs_group_batches(
    dataset: str,
    group_key: str,
    samples: list[Any],
    tags: list[str],
    args: argparse.Namespace,
) -> Iterator[ValidationItem]:
    batch_size = max(1, int(getattr(args, "validation_batch_size", 1)))
    if len(samples) != len(tags):
        raise ValueError(f"Expected samples/tags length match, got {len(samples)} vs {len(tags)}")
    for start in range(0, len(samples), batch_size):
        end = min(len(samples), start + batch_size)
        batch_tags = tags[start:end]
        batch = panogs_collate(samples[start:end])
        object.__setattr__(batch, "collect_all_vis", True)
        yield (
            dataset,
            batch,
            batch_tags[0] if len(batch_tags) == 1 else batch_tags,
            group_key,
        )


def _mask_connected_to_border(mask_2d: torch.Tensor) -> torch.Tensor:
    h, w = int(mask_2d.shape[0]), int(mask_2d.shape[1])
    if h <= 0 or w <= 0:
        return torch.zeros_like(mask_2d, dtype=torch.bool)
    border = torch.zeros_like(mask_2d, dtype=torch.bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    frontier = mask_2d & border
    visited = frontier.clone()
    kernel = torch.tensor(
        [[[[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]]]],
        device=mask_2d.device,
        dtype=torch.float32,
    )
    for _ in range(h + w):
        if not bool(frontier.any()):
            break
        neigh = F.conv2d(frontier[None, None].to(torch.float32), kernel, padding=1)[0, 0] > 0.0
        new_frontier = neigh & mask_2d & (~visited)
        visited = visited | new_frontier
        frontier = new_frontier
    return visited


def _compute_metrics_from_vis(
    vis: dict[str, Any],
    metrics_calc: MetricsCalculator,
) -> dict[str, float]:
    tgt_gt = vis["tgt_gt"].detach().to(torch.float32).clamp(0, 1)
    tgt_alpha = vis["tgt_alpha"].detach().to(torch.float32).clamp(0.0, 1.0)
    tgt_pred = linearRGB2sRGB(
        (vis["tgt_pred"].detach().to(torch.float32) / tgt_alpha.clamp(min=1e-4)).clamp(0.0, 1.0)
    ).clamp(0, 1)

    geom_mask = vis.get("tgt_metric_mask", None)
    if torch.is_tensor(geom_mask):
        geom_mask = geom_mask.detach().to(device=tgt_pred.device, dtype=torch.float32)
        if geom_mask.ndim == 3:
            geom_mask = geom_mask.unsqueeze(1)
        if tuple(geom_mask.shape[-2:]) != tuple(tgt_pred.shape[-2:]):
            geom_mask = F.interpolate(geom_mask, size=tgt_pred.shape[-2:], mode="nearest")
        tgt_geom = compute_masked_rgb_metrics(
            pred=tgt_pred,
            gt=tgt_gt,
            mask=geom_mask,
            metrics_calc=metrics_calc,
        )
    else:
        tgt_geom = metrics_calc.compute_rgb_metrics(tgt_pred, tgt_gt)

    return {
        "psnr": float(tgt_geom["psnr"]),
        "ssim": float(tgt_geom["ssim"]),
        "lpips": float(tgt_geom["lpips"]),
    }


def _save_vis_from_payload(vis: dict[str, Any], vis_dir: Path, tag: str, step: int) -> None:
    from unisharp.utils.unified_vis import save_pair_visualization

    save_pair_visualization(
        vis_dir / f"step_{int(step):07d}_{tag}.png",
        src_gt=vis["src_gt"],
        src_pred=vis["src_pred"],
        src_alpha=vis["src_alpha"],
        tgt_gt=vis["tgt_gt"],
        tgt_pred=vis["tgt_pred"],
        tgt_alpha=vis["tgt_alpha"],
        src_gt_depth=vis.get("src_gt_depth", None),
        tgt_gt_depth=vis.get("tgt_gt_depth", None),
        src_pred_depth=vis.get("src_pred_depth", None),
        tgt_pred_depth=vis.get("tgt_pred_depth", None),
        src_unik3d_depth=vis.get("src_unik3d_depth", None),
        tgt_unik3d_depth=vis.get("tgt_unik3d_depth", None),
        dataset_name=str(vis.get("dataset_name", "unknown")),
        scene=str(vis.get("scene", "unknown")),
        step=int(step),
        src_idx=int(vis.get("src_idx", -1)),
        tgt_idx=int(vis.get("tgt_idx", -1)),
        src_pose_w2c=vis.get("src_pose_w2c", None),
        tgt_pose_w2c=vis.get("tgt_pose_w2c", None),
        src_cube_gt_u8=vis.get("src_cube_gt_u8", None),
        src_cube_pred_linear=vis.get("src_cube_pred_linear", None),
        src_cube_alpha=vis.get("src_cube_alpha", None),
        tgt_cube_gt_u8=vis.get("tgt_cube_gt_u8", None),
        tgt_cube_pred_linear=vis.get("tgt_cube_pred_linear", None),
        tgt_cube_alpha=vis.get("tgt_cube_alpha", None),
    )


def _save_group_pair_pngs(group_dir: Path, group_items: list[dict[str, Any]]) -> None:
    visual_items = [item for item in group_items if isinstance(item.get("vis", None), dict)]
    if not visual_items:
        return
    group_dir.mkdir(parents=True, exist_ok=True)

    def _save_mask(mask: torch.Tensor | None, path: Path) -> None:
        if not torch.is_tensor(mask):
            return
        mask_rgb = mask.detach().to(torch.float32).clamp(0.0, 1.0)
        if mask_rgb.ndim == 4:
            mask_rgb = mask_rgb[0]
        if mask_rgb.ndim == 3 and int(mask_rgb.shape[0]) == 1:
            mask_rgb = mask_rgb.repeat(3, 1, 1)
        if mask_rgb.ndim == 2:
            mask_rgb = mask_rgb[None].repeat(3, 1, 1)
        if mask_rgb.ndim == 3:
            save_image(_to_u8_hwc(mask_rgb), path)

    def _save_alpha(alpha: torch.Tensor | None, path: Path) -> None:
        if not torch.is_tensor(alpha):
            return
        a = alpha.detach().to(torch.float32).clamp(0.0, 1.0)
        if a.ndim == 4:
            a = a[0]
        if a.ndim == 3 and int(a.shape[0]) == 1:
            a = a.repeat(3, 1, 1)
        if a.ndim == 2:
            a = a[None].repeat(3, 1, 1)
        if a.ndim == 3:
            save_image(_to_u8_hwc(a), path)

    def _noextrap_mask_from_vis(vis: dict[str, Any], which: str) -> torch.Tensor | None:
        pred = vis.get(f"{which}_pred", None)
        alpha = vis.get(f"{which}_alpha", None)
        if not (torch.is_tensor(pred) and torch.is_tensor(alpha)):
            return None
        pred_pm = linearRGB2sRGB(pred.detach().to(torch.float32).clamp(min=0.0)).clamp(0.0, 1.0)
        alpha = alpha.detach().to(torch.float32).clamp(0.0, 1.0)
        masks: list[torch.Tensor] = []
        for bi in range(int(pred_pm.shape[0])):
            black = pred_pm[bi : bi + 1].max(dim=1, keepdim=True).values <= float(2.0 / 255.0)
            low_alpha = alpha[bi : bi + 1] <= float(0.02)
            extrap_border = _mask_connected_to_border((black & low_alpha)[0, 0])
            masks.append((~extrap_border)[None, None].to(torch.float32))
        return torch.cat(masks, dim=0)

    src_row = _build_perspective_row(visual_items[0]["vis"], "src")
    save_image(src_row[0], group_dir / "src_gt.png")
    save_image(src_row[1], group_dir / "src_pred.png")
    _save_alpha(visual_items[0]["vis"].get("src_alpha", None), group_dir / "src_alpha.png")
    _save_mask(_noextrap_mask_from_vis(visual_items[0]["vis"], "src"), group_dir / "src_noextrap_mask.png")
    _save_mask(visual_items[0]["vis"].get("src_metric_mask", None), group_dir / "src_mask.png")
    for idx, item in enumerate(visual_items):
        tgt_row = _build_perspective_row(item["vis"], "tgt")
        save_image(tgt_row[0], group_dir / f"tgt_{idx:03d}_gt.png")
        save_image(tgt_row[1], group_dir / f"tgt_{idx:03d}_pred.png")
        _save_alpha(item["vis"].get("tgt_alpha", None), group_dir / f"tgt_{idx:03d}_alpha.png")
        _save_mask(_noextrap_mask_from_vis(item["vis"], "tgt"), group_dir / f"tgt_{idx:03d}_noextrap_mask.png")
        _save_mask(item["vis"].get("tgt_training_mask", None), group_dir / f"tgt_{idx:03d}_training_mask.png")
        _save_mask(item["vis"].get("tgt_metric_mask", None), group_dir / f"tgt_{idx:03d}_mask.png")


def _aggregate_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    agg: dict[str, float] = {}
    if not rows:
        return agg
    keys = sorted(set().union(*[set(r.keys()) for r in rows]))
    for k in keys:
        arr = np.array([r.get(k, np.nan) for r in rows], dtype=np.float64)
        agg[k] = _safe_nanmean(arr)
    agg["num_samples"] = float(len(rows))
    return agg


def _safe_nanmean(values: Any) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    if not np.isfinite(arr).any():
        return float("nan")
    return float(np.nanmean(arr))


def _to_u8_hwc(img_chw: torch.Tensor) -> np.ndarray:
    if img_chw.dtype == torch.uint8:
        return img_chw.permute(1, 2, 0).detach().cpu().numpy()
    x = img_chw.detach().to(torch.float32).clamp(0.0, 1.0)
    return (x * 255.0).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()


def _concat_grid(rows: list[list[np.ndarray]], pad: int = 6, pad_value: int = 0) -> np.ndarray:
    row_imgs: list[np.ndarray] = []
    for r in rows:
        padded: list[np.ndarray] = []
        for i, im in enumerate(r):
            padded.append(im)
            if i != len(r) - 1 and pad > 0:
                padded.append(np.full((im.shape[0], pad, 3), pad_value, dtype=np.uint8))
        row_imgs.append(np.concatenate(padded, axis=1))
    merged: list[np.ndarray] = []
    for i, im in enumerate(row_imgs):
        merged.append(im)
        if i != len(row_imgs) - 1 and pad > 0:
            merged.append(np.full((pad, im.shape[1], 3), pad_value, dtype=np.uint8))
    return np.concatenate(merged, axis=0)


def _resize_panel_np(panel: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    if panel.shape[0] == out_h and panel.shape[1] == out_w:
        return panel
    return np.asarray(Image.fromarray(panel).resize((int(out_w), int(out_h)), resample=Image.BILINEAR))


def _normalize_rows_for_grid(rows: list[list[np.ndarray]]) -> list[list[np.ndarray]]:
    if not rows or not rows[0]:
        return rows
    ref_h, ref_w = rows[0][0].shape[:2]
    return [[_resize_panel_np(panel, ref_h, ref_w) for panel in row] for row in rows]


def _save_gif(frames: list[np.ndarray], out_file: Path, duration_ms: int = 250) -> None:
    if not frames:
        return
    out_file.parent.mkdir(parents=True, exist_ok=True)
    pil_frames = [Image.fromarray(frame) for frame in frames]
    pil_frames[0].save(
        out_file,
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(duration_ms),
        loop=0,
        disposal=2,
    )


def _depth_range(depth: torch.Tensor | None, fallback: tuple[float, float] = (0.0, 10.0)) -> tuple[float, float]:
    if not torch.is_tensor(depth):
        return fallback
    valid = depth[torch.isfinite(depth) & (depth > 0.0)]
    if int(valid.numel()) < 8:
        return fallback
    valid = valid.to(torch.float32).flatten()
    if int(valid.numel()) > 262144:
        step = max(1, int(valid.numel()) // 262144)
        valid = valid[::step]
    vmin = float(torch.quantile(valid, 0.01).item())
    vmax = float(torch.quantile(valid, 0.99).item())
    vmin = max(0.0, vmin)
    vmax = max(vmin + 1e-3, vmax)
    return (vmin, vmax)


def _depth_panel(depth: torch.Tensor | None, val_min: float, val_max: float, blank: np.ndarray) -> np.ndarray:
    if not torch.is_tensor(depth):
        return blank
    d = depth.detach().to(torch.float32)
    valid = torch.isfinite(d) & (d > 0.0)
    if int(valid.sum().item()) < 8:
        return blank
    valid_vals = d[valid].flatten()
    if int(valid_vals.numel()) > 262144:
        step = max(1, int(valid_vals.numel()) // 262144)
        valid_vals = valid_vals[::step]
    fill = float(torch.quantile(valid_vals, 0.5).item())
    d_safe = torch.where(valid, d, torch.full_like(d, fill)).clamp(min=val_min, max=val_max)
    panel = colorize_scalar_map(d_safe[0, 0], val_min=val_min, val_max=val_max, color_map="turbo")
    out = _to_u8_hwc(panel)
    out[~valid[0, 0].detach().cpu().numpy()] = 0
    return out


def _build_perspective_row(vis: dict[str, Any], which: str) -> list[np.ndarray]:
    if which not in ("src", "tgt"):
        raise ValueError(f"which must be src/tgt, got {which}")
    gt = vis[f"{which}_gt"].detach().to(torch.float32).clamp(0.0, 1.0)
    pred = vis[f"{which}_pred"].detach().to(torch.float32)
    alpha = vis[f"{which}_alpha"].detach().to(torch.float32).clamp(0.0, 1.0)
    pred_vis = linearRGB2sRGB((pred / alpha.clamp(min=1e-4)).clamp(0.0, 1.0)).clamp(0.0, 1.0)
    err = (pred_vis - gt).abs().mean(dim=1, keepdim=True)
    err_vals = err.flatten()
    if int(err_vals.numel()) > 262144:
        step = max(1, int(err_vals.numel()) // 262144)
        err_vals = err_vals[::step]
    vmax = float(max(1e-3, min(float(torch.quantile(err_vals, 0.99).item()), 0.5)))
    err_u8 = _to_u8_hwc(colorize_scalar_map(err[0, 0], val_min=0.0, val_max=vmax, color_map="turbo"))
    alpha_u8 = _to_u8_hwc(colorize_alpha(alpha)[0])
    blank = np.zeros_like(_to_u8_hwc(gt[0]))
    gt_depth = vis.get(f"{which}_gt_depth", None)
    pred_depth = vis.get(f"{which}_pred_depth", None)
    dmin, dmax = _depth_range(gt_depth)
    if not torch.is_tensor(gt_depth):
        dmin, dmax = _depth_range(pred_depth)
    ref_d = _depth_panel(gt_depth, dmin, dmax, blank)
    pred_d = _depth_panel(pred_depth, dmin, dmax, blank)
    return [_to_u8_hwc(gt[0]), _to_u8_hwc(pred_vis[0]), err_u8, alpha_u8, ref_d, pred_d]


def _build_perspective_gif_frame(vis: dict[str, Any], which: str) -> np.ndarray:
    pred = vis[f"{which}_pred"].detach().to(torch.float32)
    alpha = vis[f"{which}_alpha"].detach().to(torch.float32).clamp(0.0, 1.0)
    pred_vis = linearRGB2sRGB((pred / alpha.clamp(min=1e-4)).clamp(0.0, 1.0)).clamp(0.0, 1.0)
    return _to_u8_hwc(pred_vis[0])


def _cube_faces_u8(cube_img: torch.Tensor, face_count: int = 6) -> list[np.ndarray]:
    x = cube_img
    if x.ndim == 5 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 4:
        return []
    faces = []
    if x.shape[0] == face_count and x.shape[1] == 3:
        for i in range(face_count):
            faces.append(_to_u8_hwc(x[i]))
    elif x.shape[0] == face_count and x.shape[-1] == 3:
        for i in range(face_count):
            faces.append(_to_u8_hwc(x[i].permute(2, 0, 1).contiguous()))
    return faces


def _build_hm3d_front_gif_frame(vis: dict[str, Any], which: str) -> np.ndarray:
    cube_pred = vis.get(f"{which}_cube_pred_linear", None)
    cube_alpha = vis.get(f"{which}_cube_alpha", None)
    if torch.is_tensor(cube_pred) and torch.is_tensor(cube_alpha):
        pred = linearRGB2sRGB(
            (cube_pred.detach().to(torch.float32) / cube_alpha.detach().to(torch.float32).clamp(min=1e-4)).clamp(0.0, 1.0)
        ).clamp(0.0, 1.0)
        faces = _cube_faces_u8(pred)
        if len(faces) == 6:
            return faces[3]
    return _build_perspective_gif_frame(vis, which)


def _save_group_gif(
    *,
    dataset: str,
    group_dir: Path,
    group_key: str,
    step: int,
    group_items: list[dict[str, Any]],
) -> None:
    visual_items = [item for item in group_items if isinstance(item.get("vis", None), dict)]
    if not visual_items:
        return
    if dataset in {"hm3d", "replica"}:
        frames = [_build_hm3d_front_gif_frame(visual_items[0]["vis"], "src")]
        frames.extend(_build_hm3d_front_gif_frame(item["vis"], "tgt") for item in visual_items[:10])
    else:
        frames = [_build_perspective_gif_frame(visual_items[0]["vis"], "src")]
        frames.extend(_build_perspective_gif_frame(item["vis"], "tgt") for item in visual_items[:10])
    _save_gif(frames, group_dir / f"step_{int(step):07d}_{group_key}.gif")


def _save_perspective_group_grid(
    *,
    group_dir: Path,
    group_key: str,
    step: int,
    group_items: list[dict[str, Any]],
) -> None:
    visual_items = [item for item in group_items if isinstance(item.get("vis", None), dict)]
    if not visual_items:
        return
    group_dir.mkdir(parents=True, exist_ok=True)
    first_vis = visual_items[0]["vis"]
    rows: list[list[np.ndarray]] = [_build_perspective_row(first_vis, "src")]
    for item in visual_items[:10]:
        rows.append(_build_perspective_row(item["vis"], "tgt"))
    while len(rows) < 11:
        rows.append(list(rows[-1]))
    rows = _normalize_rows_for_grid(rows)
    grid = _concat_grid(rows=rows, pad=6, pad_value=0)
    out_file = group_dir / f"step_{int(step):07d}_{group_key}_erp_11x6.png"
    save_image(grid, out_file)



def _manifest_max_groups(args: argparse.Namespace) -> int:
    return max(0, int(getattr(args, "manifest_max_groups", 0)))


def _validation_pseudo_root(args: argparse.Namespace) -> Path | None:
    root = getattr(args, "validation_pseudo_depth_root", None)
    return Path(root) if root is not None else Path("/media/team_data/ML4_team/datasets/sharp/validation_unik3d_pseudo_depth")


def _re10k_pseudo_scene_key(scene: Any) -> str:
    key = str(scene).strip().replace("\\", "__").replace("/", "__")
    return key if key else "unknown_scene"


def _re10k_training_pseudo_depth_path(args: argparse.Namespace, scene: Any, frame_idx: Any) -> Path | None:
    root = getattr(args, "re10k_pseudo_depth_root", None)
    if root is None:
        return None
    root = Path(root)
    split = str(getattr(args, "split", "test"))
    base = root if root.name == split else root / split
    try:
        frame_key = f"{int(frame_idx):05d}"
    except Exception:
        frame_key = str(frame_idx)
    return base / _re10k_pseudo_scene_key(scene) / f"{frame_key}.pt"


def _load_re10k_training_pseudo_depth(
    args: argparse.Namespace,
    *,
    scene: Any,
    frame_idx: Any,
    intrinsics_k3: torch.Tensor,
) -> torch.Tensor | None:
    path = _re10k_training_pseudo_depth_path(args, scene, frame_idx)
    if path is None or not path.exists():
        return None
    try:
        payload = _torch_load_any(path)
        depth_kind = "distance"
        if isinstance(payload, dict):
            depth = payload.get("z_depth_m", None)
            if torch.is_tensor(depth):
                depth_kind = "zdepth"
            else:
                depth = payload.get("distance_m", None)
                if torch.is_tensor(depth):
                    depth_kind = "distance"
                else:
                    depth = payload.get("depth_m", None)
                    depth_kind = _normalize_depth_kind(payload.get("depth_kind", "distance"), default="distance")
        else:
            depth = payload
            depth_kind = "distance"
        if isinstance(depth, np.ndarray):
            depth = torch.from_numpy(depth)
        if not torch.is_tensor(depth):
            return None
        if depth.ndim == 2:
            depth = depth.unsqueeze(0)
        if depth.ndim != 3 or int(depth.shape[0]) != 1:
            return None
        depth = depth.to(torch.float32)
        max_depth_m = float(getattr(args, "max_depth_m", DEFAULT_MAX_DEPTH_M))
        far_invalid_m = float(getattr(args, "re10k_pseudo_far_depth_invalid_m", 30.0))
        valid = torch.isfinite(depth) & (depth > 0.0)
        if far_invalid_m > 0.0:
            valid = valid & (depth <= far_invalid_m)
        depth = torch.where(valid, depth, torch.zeros_like(depth))
        if int(valid.sum().item()) <= 0:
            return None
        depth[valid] = depth[valid].clamp(max=max_depth_m)
        if _normalize_depth_kind(depth_kind, default="distance") != "zdepth":
            depth = _distance_to_z_depth_pinhole(depth, intrinsics_k3=intrinsics_k3)
        return depth
    except Exception:
        return None


def _load_val_pseudo_depth(
    args: argparse.Namespace,
    *,
    dataset: str,
    scene: Any,
    frame_idx: Any,
    intrinsics_k3: torch.Tensor,
) -> torch.Tensor | None:
    if str(dataset) == "re10k":
        depth = _load_re10k_training_pseudo_depth(
            args,
            scene=scene,
            frame_idx=frame_idx,
            intrinsics_k3=intrinsics_k3,
        )
        if torch.is_tensor(depth):
            return depth
    return _load_validation_pseudo_depth(
        _validation_pseudo_root(args),
        dataset=dataset,
        scene=scene,
        frame_idx=frame_idx,
        intrinsics_k3=intrinsics_k3,
    )


def _load_val_pseudo_depth_b1hw(
    args: argparse.Namespace,
    *,
    dataset: str,
    scene: Any,
    frame_idx: Any,
    intrinsics_k3: torch.Tensor,
) -> torch.Tensor | None:
    depth = _load_val_pseudo_depth(
        args,
        dataset=dataset,
        scene=scene,
        frame_idx=frame_idx,
        intrinsics_k3=intrinsics_k3,
    )
    return depth.unsqueeze(0) if torch.is_tensor(depth) else None


def _load_val_pseudo_distance_b1hw(
    args: argparse.Namespace,
    *,
    dataset: str,
    scene: Any,
    frame_idx: Any,
) -> torch.Tensor | None:
    depth = _load_validation_pseudo_distance(
        _validation_pseudo_root(args),
        dataset=dataset,
        scene=scene,
        frame_idx=frame_idx,
    )
    return depth.unsqueeze(0) if torch.is_tensor(depth) else None


def _iter_re10k_manifest_items(args: argparse.Namespace) -> Iterator[ValidationItem]:
    for group_idx, parts in _iter_manifest_parts(args, expected_parts=4):
        chunk_path = Path(parts[0])
        scene = str(parts[1])
        src_idx = int(parts[2])
        tgt_indices = [int(x) for x in parts[3].split(",") if x.strip()]
        payload = _torch_load_any(chunk_path)
        if not isinstance(payload, list):
            continue
        example = next((ex for ex in payload if isinstance(ex, dict) and str(ex.get("key", chunk_path.stem)) == scene), None)
        if not isinstance(example, dict):
            continue
        poses = example.get("cameras", None)
        images = example.get("images", None)
        if not torch.is_tensor(poses) or not isinstance(images, list) or poses.ndim != 2 or poses.shape[1] != 18:
            continue
        if not (0 <= src_idx < len(images)):
            continue
        src_probe = _decode_rgb_u8(images[0])
        h0, w0 = int(src_probe.shape[1]), int(src_probe.shape[2])
        intr_all = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(int(poses.shape[0]), 1, 1)
        intr_all[:, 0, 0] = poses[:, 0] * float(w0)
        intr_all[:, 1, 1] = poses[:, 1] * float(h0)
        intr_all[:, 0, 2] = poses[:, 2] * float(w0) - 0.5
        intr_all[:, 1, 2] = poses[:, 3] * float(h0) - 0.5
        w2c_all = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(int(poses.shape[0]), 1, 1)
        w2c_all[:, :3] = poses[:, 6:].reshape(-1, 3, 4).to(torch.float32)
        group_key = f"{scene}_g{group_idx:05d}"
        adapter = _PinholeGroupAdapter(
            scene=scene,
            group_key=group_key,
            src_idx=int(src_idx),
            src_img=_decode_rgb_u8(images[src_idx]).unsqueeze(0),
            src_depth=_load_val_pseudo_depth_b1hw(
                args, dataset="re10k", scene=scene, frame_idx=src_idx, intrinsics_k3=intr_all[src_idx]
            ),
            src_w2c=w2c_all[src_idx].unsqueeze(0),
            src_k=intr_all[src_idx].unsqueeze(0).clone(),
            tgt_indices=tgt_indices,
            load_target=lambda tgt_idx, images=images, intr_all=intr_all, w2c_all=w2c_all: None
            if not (0 <= int(tgt_idx) < len(images))
            else _PinholeTargetAdapter(
                idx=int(tgt_idx),
                img=_decode_rgb_u8(images[int(tgt_idx)]).unsqueeze(0),
                w2c=w2c_all[int(tgt_idx)].unsqueeze(0),
                k=intr_all[int(tgt_idx)].unsqueeze(0).clone(),
                depth=_load_val_pseudo_depth_b1hw(
                    args, dataset="re10k", scene=scene, frame_idx=int(tgt_idx), intrinsics_k3=intr_all[int(tgt_idx)]
                ),
            ),
        )
        yield from _yield_pinhole_group_batches("re10k", adapter, args)


def _iter_wildrgbd_manifest_items(args: argparse.Namespace) -> Iterator[ValidationItem]:
    root_map: dict[str, Path] = {}
    for root in _wild_validation_roots(Path(args.data_root)):
        scene_parent = root / "scenes"
        for scene_dir in sorted([p for p in scene_parent.iterdir() if p.is_dir()]) if scene_parent.exists() else []:
            root_map[f"{root.name}/{scene_dir.name}"] = scene_dir
    for group_idx, parts in _iter_manifest_parts(args, expected_parts=3):
        scene_name = str(parts[0])
        src_idx = int(parts[1])
        tgt_indices = [int(x) for x in parts[2].split(",") if x.strip()]
        scene_dir = root_map.get(scene_name)
        if scene_dir is None:
            continue
        pose_ids_np, w2c_map, intr = WildRGBDDataset._load_scene_pose_and_k(scene_dir)
        pose_ids = {int(x) for x in pose_ids_np.tolist()}
        rgb_ids = WildRGBDDataset._collect_frame_ids(scene_dir / "rgb")
        dep_ids = WildRGBDDataset._collect_frame_ids(scene_dir / "depth")
        valid_ids = pose_ids & rgb_ids & dep_ids
        if int(src_idx) not in valid_ids:
            continue
        ds_loader = WildRGBDDataset(root=scene_dir.parent.parent, split="scenes", scene_list_file=None)
        group_key = f"{scene_name}_g{group_idx:05d}"
        adapter = _PinholeGroupAdapter(
            scene=scene_name,
            group_key=group_key,
            src_idx=src_idx,
            src_img=WildRGBDDataset._load_rgb_u8(WildRGBDDataset._resolve_img_path(scene_dir / "rgb", src_idx)).unsqueeze(0),
            src_depth=ds_loader._load_depth_m(WildRGBDDataset._resolve_img_path(scene_dir / "depth", src_idx)).unsqueeze(0),
            src_w2c=torch.from_numpy(w2c_map[src_idx]).to(torch.float32).unsqueeze(0),
            src_k=intr.to(torch.float32).unsqueeze(0).clone(),
            tgt_indices=tgt_indices,
            load_target=lambda tgt_idx, scene_dir=scene_dir, valid_ids=valid_ids, ds_loader=ds_loader, intr=intr, w2c_map=w2c_map: None
            if int(tgt_idx) not in valid_ids
            else _PinholeTargetAdapter(
                idx=int(tgt_idx),
                img=WildRGBDDataset._load_rgb_u8(
                    WildRGBDDataset._resolve_img_path(scene_dir / "rgb", int(tgt_idx))
                ).unsqueeze(0),
                depth=ds_loader._load_depth_m(
                    WildRGBDDataset._resolve_img_path(scene_dir / "depth", int(tgt_idx))
                ).unsqueeze(0),
                w2c=torch.from_numpy(w2c_map[int(tgt_idx)]).to(torch.float32).unsqueeze(0),
                k=intr.to(torch.float32).unsqueeze(0).clone(),
            ),
        )
        yield from _yield_pinhole_group_batches("wildrgbd", adapter, args)


def _iter_hm3d_manifest_items(args: argparse.Namespace) -> Iterator[ValidationItem]:
    root = Path(args.data_root)
    manifest_in = _read_manifest_lines(getattr(args, "manifest_file", None), max_lines=_manifest_max_groups(args))
    for group_idx, raw in enumerate(manifest_in):
        parts = raw.split("|")
        if len(parts) != 3:
            continue
        scene_name = str(parts[0])
        src_idx = int(parts[1])
        tgt_indices = [int(x) for x in parts[2].split(",") if x.strip()]
        scene_dir = root / scene_name
        pano_dir = scene_dir / "pano"
        depth_dir = scene_dir / "pano_depth"
        cube_dir = scene_dir / "cubemaps"
        cube_depth_dir = scene_dir / "cubemaps_depth"
        if not (pano_dir.exists() and depth_dir.exists() and cube_dir.exists() and cube_depth_dir.exists()):
            continue
        R_np, t_np = _load_hm3d_pose(scene_dir)
        group_key = f"{scene_dir.name}_g{group_idx:05d}"
        src_rgb = _load_png_rgb_u8(pano_dir / f"{src_idx:05d}.png")
        src_dep = _load_png_depth_m(depth_dir / f"{src_idx:05d}.png")
        src_cube = _torch_load_any(cube_dir / f"{src_idx:05d}.torch")
        src_cdep = _torch_load_any(cube_depth_dir / f"{src_idx:05d}.torch")
        src_R = torch.from_numpy(R_np[src_idx])
        src_t = torch.from_numpy(t_np[src_idx])
        batch_size = max(1, int(getattr(args, "validation_batch_size", 1)))
        samples: list[Any] = []
        tags: list[str] = []
        for tgt_idx in tgt_indices:
            tgt_rgb = _load_png_rgb_u8(pano_dir / f"{tgt_idx:05d}.png")
            tgt_dep = _load_png_depth_m(depth_dir / f"{tgt_idx:05d}.png")
            tgt_cube = _torch_load_any(cube_dir / f"{tgt_idx:05d}.torch")
            tgt_cdep = _torch_load_any(cube_depth_dir / f"{tgt_idx:05d}.torch")
            sample = SimpleNamespace(
                src_erp_rgb_u8=src_rgb,
                tgt_erp_rgb_u8=tgt_rgb,
                src_erp_depth_m=src_dep,
                tgt_erp_depth_m=tgt_dep,
                src_cube_rgb_u8=src_cube,
                tgt_cube_rgb_u8=tgt_cube,
                src_cube_depth_m=src_cdep,
                tgt_cube_depth_m=tgt_cdep,
                src_R=src_R,
                src_t=src_t,
                tgt_R=torch.from_numpy(R_np[tgt_idx]),
                tgt_t=torch.from_numpy(t_np[tgt_idx]),
                src_idx=src_idx,
                tgt_idx=tgt_idx,
                scene=scene_dir.name,
            )
            samples.append(sample)
            tags.append(f"{group_key}_t{tgt_idx:05d}")
            if len(samples) >= batch_size:
                yield from _yield_panogs_group_batches("hm3d", group_key, samples, tags, args)
                samples = []
                tags = []
        yield from _yield_panogs_group_batches("hm3d", group_key, samples, tags, args)


def _iter_replica_manifest_items(args: argparse.Namespace) -> Iterator[ValidationItem]:
    root = _resolve_replica_test_root(Path(args.data_root))
    manifest_in = _read_manifest_lines(getattr(args, "manifest_file", None), max_lines=_manifest_max_groups(args))
    for group_idx, raw in enumerate(manifest_in):
        parts = raw.split("|")
        if len(parts) != 3:
            continue
        scene_name = str(parts[0])
        src_idx = int(parts[1])
        tgt_indices = [int(x) for x in parts[2].split(",") if x.strip()]
        scene_dir = root / scene_name
        pano_dir = scene_dir / "pano"
        depth_dir = scene_dir / "pano_depth"
        cube_dir = scene_dir / "cubemaps"
        cube_depth_dir = scene_dir / "cubemaps_depth"
        if not (pano_dir.exists() and depth_dir.exists() and cube_dir.exists() and cube_depth_dir.exists()):
            continue
        R_np, t_np = _load_hm3d_pose(scene_dir)
        group_key = f"replica_{scene_dir.name}_g{group_idx:05d}"
        src_rgb = _load_png_rgb_u8(pano_dir / f"{src_idx:05d}.png")
        src_dep = _load_png_depth_m(depth_dir / f"{src_idx:05d}.png")
        src_cube = _torch_load_any(cube_dir / f"{src_idx:05d}.torch")
        src_cdep = _torch_load_any(cube_depth_dir / f"{src_idx:05d}.torch")
        src_R = torch.from_numpy(R_np[src_idx])
        src_t = torch.from_numpy(t_np[src_idx])
        batch_size = max(1, int(getattr(args, "validation_batch_size", 1)))
        samples: list[Any] = []
        tags: list[str] = []
        for tgt_idx in tgt_indices:
            tgt_rgb = _load_png_rgb_u8(pano_dir / f"{tgt_idx:05d}.png")
            tgt_dep = _load_png_depth_m(depth_dir / f"{tgt_idx:05d}.png")
            tgt_cube = _torch_load_any(cube_dir / f"{tgt_idx:05d}.torch")
            tgt_cdep = _torch_load_any(cube_depth_dir / f"{tgt_idx:05d}.torch")
            sample = SimpleNamespace(
                src_erp_rgb_u8=src_rgb,
                tgt_erp_rgb_u8=tgt_rgb,
                src_erp_depth_m=src_dep,
                tgt_erp_depth_m=tgt_dep,
                src_cube_rgb_u8=src_cube,
                tgt_cube_rgb_u8=tgt_cube,
                src_cube_depth_m=src_cdep,
                tgt_cube_depth_m=tgt_cdep,
                src_R=src_R,
                src_t=src_t,
                tgt_R=torch.from_numpy(R_np[tgt_idx]),
                tgt_t=torch.from_numpy(t_np[tgt_idx]),
                src_idx=src_idx,
                tgt_idx=tgt_idx,
                scene=scene_dir.name,
            )
            samples.append(sample)
            tags.append(f"{group_key}_t{tgt_idx:05d}")
            if len(samples) >= batch_size:
                yield from _yield_panogs_group_batches("replica", group_key, samples, tags, args)
                samples = []
                tags = []
        yield from _yield_panogs_group_batches("replica", group_key, samples, tags, args)


def _iter_sim_manifest_items(args: argparse.Namespace) -> Iterator[ValidationItem]:
    root = Path(args.data_root)
    pose_root = Path(getattr(args, "sim_pose_root", root / "30cm"))
    manifest_in = _read_manifest_lines(getattr(args, "manifest_file", None), max_lines=_manifest_max_groups(args))
    dataset = SimPanoramaDataset(
        root=root,
        pose_root=pose_root,
        scene_names=["AI_vol3_03"],
        scene_list_file=None,
        max_index_gap=10,
        pair_max_translation_m=0.5,
        pair_min_depth_overlap=0.0,
        chunk_size=30,
        shuffle_scene=False,
        depth_max_m=float(getattr(args, "max_depth_m", DEFAULT_MAX_DEPTH_M)),
        far_depth_invalid_m=float(getattr(args, "sim_far_depth_invalid_m", 30.0)),
        far_depth_invalid_max_frac=float(getattr(args, "sim_far_depth_invalid_max_frac", 1.0)),
        seed=int(args.seed),
    )
    def _load_scene(scene_name: str) -> tuple[dict[int, Any], _EquirecToCube] | None:
        try:
            frames = dataset._load_or_build_scene_frames(scene_name)
            if not frames:
                return None
            first_rgb = dataset._load_rgb(frames[0].rgb_path)
            equ_h, equ_w = int(first_rgb.shape[1]), int(first_rgb.shape[2])
            converter = _EquirecToCube(equ_h=equ_h, equ_w=equ_w, face_w=max(1, equ_h // 2))
            frame_map = {int(frame.frame_idx): frame for frame in frames}
        except Exception as exc:
            LOGGER.warning("Skip SIM scene=%s: %s", scene_name, str(exc))
            return None
        return frame_map, converter

    def _load_frame(frame: Any, converter: _EquirecToCube) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rgb = dataset._load_rgb(frame.rgb_path)
        depth = dataset._load_depth(frame.depth_path)
        cube_rgb, cube_depth = converter.run(rgb, depth)
        return rgb, depth, cube_rgb, cube_depth

    for group_idx, raw in enumerate(manifest_in):
        parts = raw.split("|")
        if len(parts) != 3:
            continue
        scene_name = str(parts[0])
        src_idx = int(parts[1])
        tgt_indices = [int(x) for x in parts[2].split(",") if x.strip()]
        loaded_scene = _load_scene(scene_name)
        if loaded_scene is None:
            continue
        frame_map, converter = loaded_scene
        src_frame = frame_map.get(src_idx)
        if src_frame is None:
            continue
        try:
            src_rgb, src_dep, src_cube, src_cdep = _load_frame(src_frame, converter)
        except Exception as exc:
            LOGGER.warning("Skip SIM src scene=%s src=%d: %s", scene_name, int(src_idx), str(exc))
            continue
        group_key = f"sim_{scene_name}_g{group_idx:05d}"
        batch_size = max(1, int(getattr(args, "validation_batch_size", 1)))
        samples: list[Any] = []
        tags: list[str] = []
        for tgt_idx in tgt_indices:
            tgt_frame = frame_map.get(int(tgt_idx))
            if tgt_frame is None:
                continue
            try:
                tgt_rgb, tgt_dep, tgt_cube, tgt_cdep = _load_frame(tgt_frame, converter)
            except Exception:
                continue
            sample = SimpleNamespace(
                src_erp_rgb_u8=src_rgb,
                tgt_erp_rgb_u8=tgt_rgb,
                src_erp_depth_m=src_dep,
                tgt_erp_depth_m=tgt_dep,
                src_cube_rgb_u8=src_cube,
                tgt_cube_rgb_u8=tgt_cube,
                src_cube_depth_m=src_cdep,
                tgt_cube_depth_m=tgt_cdep,
                src_R=torch.eye(3, dtype=torch.float32),
                src_t=src_frame.position_xyz.clone(),
                tgt_R=torch.eye(3, dtype=torch.float32),
                tgt_t=tgt_frame.position_xyz.clone(),
                src_idx=src_idx,
                tgt_idx=int(tgt_idx),
                scene=scene_name,
            )
            samples.append(sample)
            tags.append(f"{group_key}_t{int(tgt_idx):05d}")
            if len(samples) >= batch_size:
                yield from _yield_panogs_group_batches("sim", group_key, samples, tags, args)
                samples = []
                tags = []
        yield from _yield_panogs_group_batches("sim", group_key, samples, tags, args)
        getattr(dataset, "_scene_frames_cache", {}).pop(scene_name, None)
        getattr(dataset, "_scene_pair_cache", {}).pop(scene_name, None)


def _iter_scannetpp_manifest_items(args: argparse.Namespace) -> Iterator[ValidationItem]:
    for group_idx, parts in _iter_manifest_parts(args, expected_parts=4):
        tf = Path(parts[0])
        sample_key = str(parts[1])
        src_idx = int(parts[2])
        tgt_indices = [int(x) for x in parts[3].split(",") if x.strip()]
        payload = _torch_load_any(tf)
        sample_raw = payload[0] if isinstance(payload, list) else payload
        if str(sample_raw.get("key", tf.stem)) != sample_key:
            continue
        cameras = sample_raw["cameras"].to(torch.float32)
        images = sample_raw["images"]
        if not isinstance(images, list) or int(cameras.shape[0]) < 11:
            continue
        w2c_all = []
        intr_all = []
        for i in range(int(cameras.shape[0])):
            cam = cameras[i]
            fx_n, fy_n, cx_n, cy_n, w0, h0 = cam[:6]
            w2c = torch.eye(4, dtype=torch.float32)
            w2c[:3, :] = cam[6:].reshape(3, 4)
            k = torch.eye(3, dtype=torch.float32)
            k[0, 0] = fx_n * w0
            k[1, 1] = fy_n * h0
            k[0, 2] = cx_n * w0
            k[1, 2] = cy_n * h0
            w2c_all.append(w2c)
            intr_all.append(k)
        w2c_t = torch.stack(w2c_all, dim=0)
        intr_t = torch.stack(intr_all, dim=0)
        group_key = f"{sample_key}_g{group_idx:05d}"
        adapter = _PinholeGroupAdapter(
            scene=sample_key,
            group_key=group_key,
            src_idx=int(src_idx),
            src_img=_decode_rgb_u8(images[src_idx]).unsqueeze(0),
            src_depth=_load_val_pseudo_depth_b1hw(
                args, dataset="scannetpp", scene=sample_key, frame_idx=src_idx, intrinsics_k3=intr_t[src_idx]
            ),
            src_w2c=w2c_t[src_idx].unsqueeze(0),
            src_k=intr_t[src_idx].unsqueeze(0),
            tgt_indices=tgt_indices,
            load_target=lambda tgt_idx, images=images, intr_t=intr_t, w2c_t=w2c_t: _PinholeTargetAdapter(
                idx=int(tgt_idx),
                img=_decode_rgb_u8(images[int(tgt_idx)]).unsqueeze(0),
                w2c=w2c_t[int(tgt_idx)].unsqueeze(0),
                k=intr_t[int(tgt_idx)].unsqueeze(0),
                depth=_load_val_pseudo_depth_b1hw(
                    args,
                    dataset="scannetpp",
                    scene=sample_key,
                    frame_idx=int(tgt_idx),
                    intrinsics_k3=intr_t[int(tgt_idx)],
                ),
            ),
        )
        yield from _yield_pinhole_group_batches("scannetpp", adapter, args)


def _iter_scanetpp_fisheye_manifest_items(args: argparse.Namespace) -> Iterator[ValidationItem]:
    root = Path(args.data_root)
    loader = ScannetppFisheyeDataset(
        root=root,
        scene_list_file=None,
        min_frame_gap=1,
        max_frame_gap=10,
        pair_max_translation_m=float(args.pair_max_translation_m),
        shuffle_scene=False,
        shuffle_frame=False,
        skip_bad=True,
        batch_size_hint=1,
        depth_max_m=float(getattr(args, "max_depth_m", DEFAULT_MAX_DEPTH_M)),
        far_depth_invalid_m=float(getattr(args, "scanetpp_fisheye_far_depth_invalid_m", 30.0)),
        seed=int(args.seed),
    )
    batch_size = max(1, int(getattr(args, "validation_batch_size", 1)))
    for group_idx, parts in _iter_manifest_parts(args, expected_parts=4):
        scene_id = str(parts[0])
        scene_dir = Path(parts[1])
        if not scene_dir.is_absolute():
            scene_dir = root / scene_dir
        src_pos = int(parts[2])
        tgt_positions = [int(x) for x in parts[3].split(",") if x.strip()]
        try:
            camera_params, frames = loader._load_scene_frames(scene_id, scene_dir)
        except Exception as exc:
            LOGGER.warning("Skip ScanNet++ fisheye scene=%s: %s", scene_id, str(exc))
            continue
        if not (0 <= src_pos < len(frames)):
            continue
        try:
            src_loaded = loader._load_frame_tensor(frames[src_pos], camera_params)
        except Exception as exc:
            LOGGER.warning("Skip ScanNet++ fisheye src scene=%s src=%d: %s", scene_id, int(src_pos), str(exc))
            continue
        pending_pos: list[int] = []
        pending_frames: list[dict[str, Any]] = []
        pending_loaded: list[dict[str, torch.Tensor]] = []

        def _flush() -> Iterator[tuple[str, Any, str | list[str], str]]:
            if not pending_pos:
                return
            group_key = f"scanetpp_fisheye_{scene_id}_g{group_idx:05d}"
            batch = _make_scanetpp_fisheye_batch(
                scene=scene_id,
                src_pos=src_pos,
                tgt_positions=list(pending_pos),
                src_frame=frames[src_pos],
                tgt_frames=list(pending_frames),
                src_loaded=src_loaded,
                tgt_loaded=list(pending_loaded),
            )
            tags = [f"{group_key}_t{int(t):05d}" for t in pending_pos]
            yield ("scanetpp_fisheye", batch, tags[0] if len(tags) == 1 else tags, group_key)

        for tgt_pos in tgt_positions:
            if not (0 <= int(tgt_pos) < len(frames)):
                continue
            try:
                tgt_loaded = loader._load_frame_tensor(frames[int(tgt_pos)], camera_params)
            except Exception:
                continue
            pending_pos.append(int(tgt_pos))
            pending_frames.append(frames[int(tgt_pos)])
            pending_loaded.append(tgt_loaded)
            if len(pending_pos) >= batch_size:
                yield from _flush()
                pending_pos = []
                pending_frames = []
                pending_loaded = []
        if pending_pos:
            yield from _flush()


def _load_smx_sim_fisheye_scene(scene_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    meta = json.loads((scene_dir / "transforms.json").read_text(encoding="utf-8"))
    raw_frames = list(meta.get("frames", []))
    frames: list[dict[str, Any]] = []
    for local_idx, frame in enumerate(raw_frames):
        rel = Path(str(frame.get("file_path", "")))
        image_path = scene_dir / rel
        if not image_path.exists():
            image_path = Path(str(frame.get("source_image", "")))
        source_image = Path(str(frame.get("source_image", "")))
        if not image_path.exists() or frame.get("transform_matrix") is None:
            continue
        c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
        frames.append(
            {
                "image_name": image_path.name,
                "image_path": image_path,
                "source_image": source_image,
                "w2c": torch.linalg.inv(c2w),
                "idx": int(frame.get("source_image_index", local_idx)),
                "pos": int(local_idx),
                "yaw_pitch_roll_deg": list(frame.get("yaw_pitch_roll_deg", [0.0, 0.0, 0.0])),
            }
        )
    return meta, frames


def _smx_sim_fisheye_valid_mask(rgb_u8: torch.Tensor, meta: dict[str, Any]) -> torch.Tensor:
    h, w = int(rgb_u8.shape[-2]), int(rgb_u8.shape[-1])
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    cx = float(meta.get("cx", w * 0.5))
    cy = float(meta.get("cy", h * 0.5))
    radius = float(meta.get("valid_radius_px", min(h, w) * 0.5))
    circle = ((xx.to(torch.float32) - cx) ** 2 + (yy.to(torch.float32) - cy) ** 2) <= radius * radius
    nonblack = rgb_u8.to(torch.float32).sum(dim=0) > 1.0
    return (circle & nonblack).to(torch.float32).unsqueeze(0)


def _smx_sim_rotation_yaw_pitch_roll(yaw_deg: float, pitch_deg: float, roll_deg: float) -> torch.Tensor:
    yaw = math.radians(float(yaw_deg))
    pitch = math.radians(float(pitch_deg))
    roll = math.radians(float(roll_deg))
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    r_yaw = torch.tensor([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=torch.float32)
    r_pitch = torch.tensor([[1.0, 0.0, 0.0], [0.0, cp, sp], [0.0, -sp, cp]], dtype=torch.float32)
    r_roll = torch.tensor([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
    return (r_yaw @ r_pitch @ r_roll).to(torch.float32)


def _smx_sim_fisheye_grid(
    *,
    meta: dict[str, Any],
    frame: dict[str, Any],
    erp_h: int,
    erp_w: int,
    fish_h: int,
    fish_w: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    yy, xx = torch.meshgrid(
        torch.arange(fish_h, dtype=torch.float32) + 0.5,
        torch.arange(fish_w, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    fx = float(meta.get("fl_x", fish_w / max(math.radians(float(meta.get("fov_deg", 130.0))), 1e-6)))
    fy = float(meta.get("fl_y", fish_h / max(math.radians(float(meta.get("fov_deg", 130.0))), 1e-6)))
    cx = float(meta.get("cx", fish_w * 0.5))
    cy = float(meta.get("cy", fish_h * 0.5))
    fov_rad = math.radians(float(meta.get("fov_deg", 130.0)))
    half_fov = 0.5 * fov_rad

    dx = (xx - cx) / max(fx, 1e-6)
    dy = (yy - cy) / max(fy, 1e-6)
    theta = torch.sqrt(dx * dx + dy * dy)
    valid = theta <= float(half_fov)

    scale = torch.zeros_like(theta)
    nonzero = theta > 1e-8
    scale[nonzero] = torch.sin(theta[nonzero]) / theta[nonzero]
    rays = torch.stack([dx * scale, -dy * scale, torch.cos(theta)], dim=-1)
    rays[~nonzero] = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)

    ypr = list(frame.get("yaw_pitch_roll_deg", [0.0, 0.0, 0.0]))
    rot = _smx_sim_rotation_yaw_pitch_roll(float(ypr[0]), float(ypr[1]), float(ypr[2]))
    rays = rays @ rot.T
    rays = rays / rays.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    lon = torch.atan2(rays[..., 0], rays[..., 2])
    lat = torch.atan2(rays[..., 1], torch.sqrt(rays[..., 0] ** 2 + rays[..., 2] ** 2))
    map_x = (lon / (2.0 * math.pi) + 0.5) * float(erp_w) - 0.5
    map_y = (0.5 - lat / math.pi) * float(erp_h) - 0.5
    map_x = torch.remainder(map_x + 0.5, float(erp_w)) - 0.5
    x_norm = 2.0 * (map_x + 0.5) / float(erp_w) - 1.0
    y_norm = 2.0 * (map_y + 0.5) / float(erp_h) - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1).to(torch.float32)
    return grid, valid.to(torch.float32).unsqueeze(0)


def _smx_sim_project_erp_tensor_to_fisheye(
    tensor: torch.Tensor,
    grid: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    mode: str = "bilinear",
) -> torch.Tensor:
    x = tensor.detach().to(torch.float32)
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.ndim != 4:
        raise ValueError(f"Expected 3D/4D tensor for SMX projection, got shape={tuple(x.shape)}")
    device = x.device
    grid_b = grid.to(device=device, dtype=torch.float32).unsqueeze(0).expand(int(x.shape[0]), -1, -1, -1)
    mask_b = valid_mask.to(device=device, dtype=x.dtype)
    if mask_b.ndim == 3:
        mask_b = mask_b.unsqueeze(0)
    if mask_b.ndim != 4:
        raise ValueError(f"Expected 3D/4D SMX valid mask, got shape={tuple(mask_b.shape)}")
    mask_b = mask_b.expand(int(x.shape[0]), -1, -1, -1)
    out = F.grid_sample(x, grid_b, mode=mode, padding_mode="zeros", align_corners=False)
    return out * mask_b


def _project_smx_sim_fisheye_vis(vis: dict[str, Any], batch: Any, sample_idx: int) -> dict[str, Any]:
    if not hasattr(batch, "smx_fisheye_src_grid"):
        return vis
    i = int(sample_idx)
    src_grid = batch.smx_fisheye_src_grid[i]
    tgt_grid = batch.smx_fisheye_tgt_grid[i]
    src_mask = batch.smx_fisheye_src_valid_mask[i : i + 1]
    tgt_mask = batch.smx_fisheye_tgt_valid_mask[i : i + 1]

    def _batch_value(name: str, idx: int, device: torch.device) -> torch.Tensor | None:
        value = getattr(batch, name, None)
        if not torch.is_tensor(value) or idx >= int(value.shape[0]):
            return None
        return value[idx : idx + 1].to(device=device)

    device = vis["tgt_pred"].device if torch.is_tensor(vis.get("tgt_pred", None)) else torch.device("cpu")
    src_gt = _batch_value("smx_fisheye_src_rgb_u8", i, device)
    tgt_gt = _batch_value("smx_fisheye_tgt_rgb_u8", i, device)
    if src_gt is not None:
        vis["src_gt"] = (src_gt.to(torch.float32) / 255.0).clamp(0.0, 1.0)
    if tgt_gt is not None:
        vis["tgt_gt"] = (tgt_gt.to(torch.float32) / 255.0).clamp(0.0, 1.0)

    for prefix, grid, mask in (("src", src_grid, src_mask), ("tgt", tgt_grid, tgt_mask)):
        pred_key = f"{prefix}_pred"
        alpha_key = f"{prefix}_alpha"
        if torch.is_tensor(vis.get(pred_key, None)):
            vis[pred_key] = _smx_sim_project_erp_tensor_to_fisheye(vis[pred_key], grid, mask, mode="bilinear")
        if torch.is_tensor(vis.get(alpha_key, None)):
            vis[alpha_key] = _smx_sim_project_erp_tensor_to_fisheye(vis[alpha_key], grid, mask, mode="bilinear")
        for depth_key in (f"{prefix}_gt_depth", f"{prefix}_pred_depth", f"{prefix}_unik3d_depth"):
            if torch.is_tensor(vis.get(depth_key, None)):
                vis[depth_key] = _smx_sim_project_erp_tensor_to_fisheye(vis[depth_key], grid, mask, mode="bilinear")

    vis["src_metric_mask"] = src_mask.to(device=device, dtype=torch.float32)
    vis["tgt_metric_mask"] = tgt_mask.to(device=device, dtype=torch.float32)
    vis["dataset_name"] = "smx_sim_fisheye"
    vis["projection_pipeline"] = "source_fisheye_to_source_pano_infer_target_pano_to_target_fisheye"
    return vis


def _yield_smx_sim_fisheye_pano_batches(
    *,
    group_key: str,
    samples: list[Any],
    tags: list[str],
    args: argparse.Namespace,
) -> Iterator[ValidationItem]:
    batch_size = max(1, int(getattr(args, "validation_batch_size", 1)))
    if len(samples) != len(tags):
        raise ValueError(f"Expected samples/tags length match, got {len(samples)} vs {len(tags)}")
    for start in range(0, len(samples), batch_size):
        end = min(len(samples), start + batch_size)
        chunk = samples[start:end]
        batch_tags = tags[start:end]
        batch = panogs_collate(chunk)
        object.__setattr__(batch, "collect_all_vis", True)
        object.__setattr__(batch, "disable_depth_gt", True)
        for attr in (
            "smx_fisheye_src_rgb_u8",
            "smx_fisheye_tgt_rgb_u8",
            "smx_fisheye_src_valid_mask",
            "smx_fisheye_tgt_valid_mask",
            "smx_fisheye_src_grid",
            "smx_fisheye_tgt_grid",
        ):
            object.__setattr__(batch, attr, torch.stack([getattr(s, attr) for s in chunk], dim=0))
        yield ("smx_sim_fisheye", batch, batch_tags[0] if len(batch_tags) == 1 else batch_tags, group_key)


def _smx_frame_position_from_w2c(frame: dict[str, Any], meta: dict[str, Any]) -> torch.Tensor:
    w2c = frame["w2c"]
    if torch.is_tensor(w2c):
        w2c_t = w2c.detach().clone().to(torch.float32)
    else:
        w2c_t = torch.as_tensor(w2c, dtype=torch.float32)
    raw_xyz = torch.linalg.inv(w2c_t)[:3, 3].clone()
    raw_scale = float(meta.get("position_scale", 1.0))
    if abs(raw_scale) > 1e-8:
        raw_xyz = raw_xyz / raw_scale
    return torch.stack([raw_xyz[1], -raw_xyz[2], raw_xyz[0]], dim=0).to(torch.float32) * 0.01


def _iter_smx_sim_fisheye_manifest_items(args: argparse.Namespace) -> Iterator[ValidationItem]:
    root = Path(args.data_root)
    batch_size = max(1, int(getattr(args, "validation_batch_size", 1)))
    for group_idx, parts in _iter_manifest_parts(args, expected_parts=4):
        scene_id = str(parts[0])
        scene_dir = Path(parts[1])
        if not scene_dir.is_absolute():
            scene_dir = root / scene_dir
        src_pos = int(parts[2])
        tgt_positions = [int(x) for x in parts[3].split(",") if x.strip()]
        try:
            meta, frames = _load_smx_sim_fisheye_scene(scene_dir)
        except Exception as exc:
            LOGGER.warning("Skip SMX SIM fisheye scene=%s: %s", scene_id, str(exc))
            continue
        if not (0 <= src_pos < len(frames)):
            continue

        def _load_frame(frame: dict[str, Any]) -> dict[str, torch.Tensor]:
            source_image = Path(str(frame.get("source_image", "")))
            if not source_image.exists():
                raise FileNotFoundError(source_image)
            erp_rgb = _load_png_rgb_u8(source_image)
            converter = _EquirecToCube(
                equ_h=int(erp_rgb.shape[-2]),
                equ_w=int(erp_rgb.shape[-1]),
                face_w=max(1, int(erp_rgb.shape[-2]) // 2),
            )
            cube_rgb = converter.run_rgb(erp_rgb)
            erp_depth = torch.zeros((1, int(erp_rgb.shape[-2]), int(erp_rgb.shape[-1])), dtype=torch.float32)
            cube_depth = torch.zeros((6, int(converter.face_w), int(converter.face_w), 1), dtype=torch.float32)
            fish_rgb = _load_png_rgb_u8(Path(frame["image_path"]))
            grid, valid = _smx_sim_fisheye_grid(
                meta=meta,
                frame=frame,
                erp_h=int(erp_rgb.shape[-2]),
                erp_w=int(erp_rgb.shape[-1]),
                fish_h=int(fish_rgb.shape[-2]),
                fish_w=int(fish_rgb.shape[-1]),
            )
            valid = (valid * _smx_sim_fisheye_valid_mask(fish_rgb, meta)).clamp(0.0, 1.0)
            return {
                "erp_rgb_u8": erp_rgb,
                "erp_depth_m": erp_depth,
                "cube_rgb_u8": cube_rgb,
                "cube_depth_m": cube_depth,
                "fish_rgb_u8": fish_rgb,
                "fish_valid_mask": valid,
                "fish_grid": grid,
            }

        try:
            src_loaded = _load_frame(frames[src_pos])
        except Exception as exc:
            LOGGER.warning("Skip SMX SIM fisheye src scene=%s src=%d: %s", scene_id, int(src_pos), str(exc))
            continue
        pending_pos: list[int] = []
        pending_samples: list[Any] = []
        pending_tags: list[str] = []

        def _flush() -> Iterator[tuple[str, Any, str | list[str], str]]:
            if not pending_pos:
                return
            group_key = f"smx_sim_fisheye_{scene_id}_g{group_idx:05d}"
            yield from _yield_smx_sim_fisheye_pano_batches(
                group_key=group_key,
                samples=list(pending_samples),
                tags=list(pending_tags),
                args=args,
            )

        for tgt_pos in tgt_positions:
            if not (0 <= int(tgt_pos) < len(frames)):
                continue
            try:
                tgt_loaded = _load_frame(frames[int(tgt_pos)])
            except Exception:
                continue
            group_key = f"smx_sim_fisheye_{scene_id}_g{group_idx:05d}"
            sample = SimpleNamespace(
                src_erp_rgb_u8=src_loaded["erp_rgb_u8"],
                tgt_erp_rgb_u8=tgt_loaded["erp_rgb_u8"],
                src_erp_depth_m=src_loaded["erp_depth_m"],
                tgt_erp_depth_m=tgt_loaded["erp_depth_m"],
                src_cube_rgb_u8=src_loaded["cube_rgb_u8"],
                tgt_cube_rgb_u8=tgt_loaded["cube_rgb_u8"],
                src_cube_depth_m=src_loaded["cube_depth_m"],
                tgt_cube_depth_m=tgt_loaded["cube_depth_m"],
                src_R=torch.eye(3, dtype=torch.float32),
                src_t=_smx_frame_position_from_w2c(frames[src_pos], meta),
                tgt_R=torch.eye(3, dtype=torch.float32),
                tgt_t=_smx_frame_position_from_w2c(frames[int(tgt_pos)], meta),
                src_idx=int(frames[src_pos].get("idx", src_pos)),
                tgt_idx=int(frames[int(tgt_pos)].get("idx", tgt_pos)),
                scene=scene_id,
                smx_fisheye_src_rgb_u8=src_loaded["fish_rgb_u8"],
                smx_fisheye_tgt_rgb_u8=tgt_loaded["fish_rgb_u8"],
                smx_fisheye_src_valid_mask=src_loaded["fish_valid_mask"],
                smx_fisheye_tgt_valid_mask=tgt_loaded["fish_valid_mask"],
                smx_fisheye_src_grid=src_loaded["fish_grid"],
                smx_fisheye_tgt_grid=tgt_loaded["fish_grid"],
            )
            pending_pos.append(int(tgt_pos))
            pending_samples.append(sample)
            pending_tags.append(f"{group_key}_t{int(tgt_pos):05d}")
            if len(pending_pos) >= batch_size:
                yield from _flush()
                pending_pos = []
                pending_samples = []
                pending_tags = []
        if pending_pos:
            yield from _flush()


def _iter_tat_manifest_items(args: argparse.Namespace) -> Iterator[ValidationItem]:
    root = Path(args.data_root)
    scene_roots = _colmap_scene_roots(root)
    scene_root_map = {scene_root.name: scene_root for scene_root in scene_roots}
    for group_idx, parts in _iter_manifest_parts(args, expected_parts=3):
        scene_name = str(parts[0])
        scene_root = scene_root_map.get(scene_name)
        if scene_root is None:
            continue
        image_dir = _colmap_image_dir(scene_root)
        image_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
        image_map = {p.name: p for p in image_paths}
        colmap_entries = _load_scaled_colmap_entries(scene_root)
        if not colmap_entries:
            continue
        image_paths = [p for p in image_paths if p.name in colmap_entries]
        image_map = {p.name: p for p in image_paths}
        src_name = str(parts[1])
        tgt_names = [x for x in parts[2].split(",") if x.strip()]
        if src_name not in image_map:
            continue
        group_key = f"tat_{scene_name}_g{group_idx:05d}"
        src_img = _load_png_rgb_u8(image_map[src_name]).unsqueeze(0)
        src_meta = colmap_entries[src_name]
        src_k = src_meta["k"].unsqueeze(0).clone()
        src_w2c = src_meta["w2c"].unsqueeze(0).clone()
        ref_h = int(src_meta["height"])
        ref_w = int(src_meta["width"])
        if (int(src_img.shape[-2]) != ref_h) or (int(src_img.shape[-1]) != ref_w):
            sx0 = float(int(src_img.shape[-1])) / float(ref_w)
            sy0 = float(int(src_img.shape[-2])) / float(ref_h)
            src_k = _resize_k3_align_corners_false(src_k, sx=sx0, sy=sy0)
        name_to_idx = {p.name: i for i, p in enumerate(image_paths)}
        src_idx = int(name_to_idx[src_name])

        def _load_tat_target(tgt_idx: int) -> _PinholeTargetAdapter | None:
            tgt_name = image_paths[int(tgt_idx)].name
            if tgt_name not in image_map or tgt_name not in colmap_entries:
                return None
            tgt_img = _load_png_rgb_u8(image_map[tgt_name]).unsqueeze(0)
            tgt_meta = colmap_entries[tgt_name]
            tgt_k = tgt_meta["k"].unsqueeze(0).clone()
            if (int(tgt_img.shape[-2]) != int(tgt_meta["height"])) or (int(tgt_img.shape[-1]) != int(tgt_meta["width"])):
                sx0 = float(int(tgt_img.shape[-1])) / float(int(tgt_meta["width"]))
                sy0 = float(int(tgt_img.shape[-2])) / float(int(tgt_meta["height"]))
                tgt_k = _resize_k3_align_corners_false(tgt_k, sx=sx0, sy=sy0)
            return _PinholeTargetAdapter(
                idx=int(tgt_idx),
                img=tgt_img,
                w2c=tgt_meta["w2c"].unsqueeze(0).clone(),
                k=tgt_k,
                depth=_load_val_pseudo_depth_b1hw(
                    args, dataset="tat", scene=scene_name, frame_idx=int(tgt_idx), intrinsics_k3=tgt_k[0]
                ),
            )

        adapter = _PinholeGroupAdapter(
            scene=scene_name,
            group_key=group_key,
            src_idx=src_idx,
            src_img=src_img,
            src_depth=_load_val_pseudo_depth_b1hw(
                args, dataset="tat", scene=scene_name, frame_idx=src_idx, intrinsics_k3=src_k[0]
            ),
            src_w2c=src_w2c,
            src_k=src_k,
            tgt_indices=[int(name_to_idx[tgt_name]) for tgt_name in tgt_names if tgt_name in name_to_idx],
            load_target=_load_tat_target,
        )
        yield from _yield_pinhole_group_batches("tat", adapter, args)


def _dl3dv_frame_id_from_name(name: str) -> int:
    return int(Path(name).stem.split("_")[-1])


def _load_dl3dv_scene(scene_dir: Path) -> tuple[dict[int, Path], dict[int, torch.Tensor], dict[int, torch.Tensor]] | None:
    transforms_path = scene_dir / "transforms.json"
    image_dir = scene_dir / "images_4"
    if not (transforms_path.exists() and image_dir.exists()):
        return None
    meta = json.loads(transforms_path.read_text(encoding="utf-8"))
    image_paths = {int(_dl3dv_frame_id_from_name(p.name)): p for p in image_dir.glob("*.png")}
    if not image_paths:
        return None
    orig_w = int(meta["w"])
    orig_h = int(meta["h"])
    k = torch.eye(3, dtype=torch.float32)
    k[0, 0] = float(meta["fl_x"])
    k[1, 1] = float(meta["fl_y"])
    k[0, 2] = float(meta["cx"])
    k[1, 2] = float(meta["cy"])
    example_path = next(iter(image_paths.values()))
    with Image.open(example_path) as img:
        cur_w, cur_h = int(img.size[0]), int(img.size[1])
    k_cur = k.clone()
    if cur_h != orig_h or cur_w != orig_w:
        k_cur = _resize_k3_align_corners_false(
            k_cur.unsqueeze(0),
            sx=float(cur_w) / float(orig_w),
            sy=float(cur_h) / float(orig_h),
        )[0]
    w2c_map: dict[int, torch.Tensor] = {}
    intr_map: dict[int, torch.Tensor] = {}
    for frame in meta.get("frames", []):
        rel_path = str(frame.get("file_path", ""))
        try:
            frame_id = int(_dl3dv_frame_id_from_name(Path(rel_path).name))
        except Exception:
            continue
        if frame_id not in image_paths:
            continue
        c2w = _nerf_c2w_to_opencv_c2w(frame["transform_matrix"])
        w2c_map[frame_id] = torch.linalg.inv(c2w)
        intr_map[frame_id] = k_cur.clone()
    return image_paths, w2c_map, intr_map


def _resolve_dl3dv_scene_dir(args: argparse.Namespace, scene_name: str, scene_dir_raw: str) -> Path | None:
    scene_dir = Path(scene_dir_raw)
    if scene_dir.exists():
        return scene_dir
    root = Path(args.data_root)
    parts = scene_name.split("/", 1)
    if len(parts) == 2:
        candidate = root / parts[0] / parts[1] / parts[1]
        if candidate.exists():
            return candidate
        candidate = root / parts[0] / parts[1]
        if candidate.exists():
            return candidate
    return None


def _iter_dl3dv_manifest_items(args: argparse.Namespace) -> Iterator[ValidationItem]:
    for group_idx, parts in _iter_manifest_parts(args, expected_parts=4):
        scene_name = str(parts[0])
        scene_dir = _resolve_dl3dv_scene_dir(args, scene_name=scene_name, scene_dir_raw=parts[1])
        if scene_dir is None:
            continue
        src_idx = int(parts[2])
        tgt_indices = [int(x) for x in parts[3].split(",") if x.strip()]
        loaded = _load_dl3dv_scene(scene_dir)
        if loaded is None:
            continue
        image_paths, w2c_map, intr_map = loaded
        if src_idx not in image_paths or src_idx not in w2c_map:
            continue
        group_key = f"dl3dv_{scene_name.replace('/', '_')}_g{group_idx:05d}"
        adapter = _PinholeGroupAdapter(
            scene=scene_name,
            group_key=group_key,
            src_idx=src_idx,
            src_img=_load_png_rgb_u8(image_paths[src_idx]).unsqueeze(0),
            src_depth=_load_val_pseudo_depth_b1hw(
                args, dataset="dl3dv", scene=scene_name, frame_idx=src_idx, intrinsics_k3=intr_map[src_idx]
            ),
            src_w2c=w2c_map[src_idx].unsqueeze(0),
            src_k=intr_map[src_idx].unsqueeze(0).clone(),
            tgt_indices=tgt_indices,
            load_target=lambda tgt_idx, image_paths=image_paths, w2c_map=w2c_map, intr_map=intr_map: None
            if int(tgt_idx) not in image_paths or int(tgt_idx) not in w2c_map
            else _PinholeTargetAdapter(
                idx=int(tgt_idx),
                img=_load_png_rgb_u8(image_paths[int(tgt_idx)]).unsqueeze(0),
                w2c=w2c_map[int(tgt_idx)].unsqueeze(0),
                k=intr_map[int(tgt_idx)].unsqueeze(0).clone(),
                depth=_load_val_pseudo_depth_b1hw(
                    args,
                    dataset="dl3dv",
                    scene=scene_name,
                    frame_idx=int(tgt_idx),
                    intrinsics_k3=intr_map[int(tgt_idx)],
                ),
            ),
        )
        yield from _yield_pinhole_group_batches("dl3dv", adapter, args)


def _iter_dataset_items(args: argparse.Namespace) -> Iterable[ValidationItem]:
    if getattr(args, "manifest_file", None) is None:
        raise ValueError("Validation requires --manifest-file. Build manifests first with scripts/build_validation_manifests.py.")
    dataset = str(args.dataset)
    if dataset == "re10k":
        return _iter_re10k_manifest_items(args)
    if dataset == "dl3dv":
        return _iter_dl3dv_manifest_items(args)
    if dataset == "replica":
        return _iter_replica_manifest_items(args)
    if dataset == "sim":
        return _iter_sim_manifest_items(args)
    if dataset == "wildrgbd":
        return _iter_wildrgbd_manifest_items(args)
    if dataset == "scannetpp":
        return _iter_scannetpp_manifest_items(args)
    if dataset == "scanetpp_fisheye":
        return _iter_scanetpp_fisheye_manifest_items(args)
    if dataset == "smx_sim_fisheye":
        return _iter_smx_sim_fisheye_manifest_items(args)
    if dataset == "tat":
        return _iter_tat_manifest_items(args)
    return _iter_hm3d_manifest_items(args)


def _finalize_validation_group(
    *,
    dataset: str,
    step: int,
    vis_dir: Path,
    group_key: str,
    group_items: list[dict[str, Any]],
) -> dict[str, float] | None:
    if not group_items:
        return None
    group_row = {
        "psnr": _safe_nanmean([float(e["row"]["psnr"]) for e in group_items]),
        "ssim": _safe_nanmean([float(e["row"]["ssim"]) for e in group_items]),
        "lpips": _safe_nanmean([float(e["row"]["lpips"]) for e in group_items]),
    }
    group_dir = vis_dir / group_key
    _save_group_pair_pngs(group_dir, group_items)
    _save_perspective_group_grid(
        group_dir=group_dir,
        group_key=group_key,
        step=int(step),
        group_items=group_items,
    )
    _save_group_gif(
        dataset=dataset,
        group_dir=group_dir,
        group_key=group_key,
        step=int(step),
        group_items=group_items,
    )
    visual_items = [item for item in group_items if isinstance(item.get("vis", None), dict)]
    if dataset in {"hm3d", "replica"}:
        for j, item in enumerate(visual_items[:10], start=1):
            _save_vis_from_payload(
                item["vis"],
                vis_dir=group_dir,
                tag=f"{group_key}_t{j:02d}",
                step=int(step),
            )
    return group_row

def run_validation(args: argparse.Namespace) -> None:
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    dev = torch.device(args.device)
    model, step = _load_model(Path(args.checkpoint), dev)
    trainer = _build_trainer(model, dev, args)
    metrics_calc = MetricsCalculator(device=dev, compute_lpips=not bool(getattr(args, "fast_metrics", False)))

    dataset = str(args.dataset)
    items = _iter_dataset_items(args)

    if getattr(args, "out_dir", None) is not None:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(args.checkpoint).parent / f"validation_{dataset}"
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = out_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    sample_csv = out_dir / f"validation_sample_metrics_{dataset}.csv"


    group_rows: list[dict[str, float]] = []
    failure_rows: list[dict[str, Any]] = []
    current_group_key: str | None = None
    current_group_items: list[dict[str, Any]] = []
    num_rows = 0

    LOGGER.info("Validation start: dataset=%s checkpoint=%s", dataset, str(args.checkpoint))
    pbar = tqdm(items, desc=f"validate_{dataset}", leave=False, disable=True)
    for i, (dataset_name, batch, tag, group_key) in enumerate(pbar):
        if current_group_key is not None and group_key != current_group_key:
            group_row = _finalize_validation_group(
                dataset=dataset,
                step=int(step),
                vis_dir=vis_dir,
                group_key=current_group_key,
                group_items=current_group_items,
            )
            if group_row is not None:
                group_rows.append(group_row)
                num_rows += len(current_group_items)
                pbar.set_postfix(groups=len(group_rows), targets=num_rows, refresh=False)
            current_group_items = []
        current_group_key = group_key
        try:
            with torch.no_grad():
                result = trainer.process_batch(
                    batch,
                    dataset_name=dataset_name,
                    step=int(step),
                    need_vis=True,
                )
            vis_payloads = result.get("vis_payloads", None)
            if isinstance(vis_payloads, list) and vis_payloads:
                vis_list = [v for v in vis_payloads if isinstance(v, dict)]
            else:
                vis = result.get("vis_payload", None)
                vis_list = [vis] if isinstance(vis, dict) else []
            if not vis_list:
                continue
            if str(dataset_name) == "smx_sim_fisheye":
                vis_list = [_project_smx_sim_fisheye_vis(vis, batch, j) for j, vis in enumerate(vis_list)]
            tags = tag if isinstance(tag, list) else [tag]
            metric_mask = metric_mask_from_pinhole_batch(
                batch,
                dataset=str(dataset_name),
                cache_dir=Path(args.metric_mask_cache_dir) if args.metric_mask_cache_dir is not None else None,
                device=dev,
            )
            for j, vis in enumerate(vis_list):
                if torch.is_tensor(vis.get("tgt_metric_mask", None)):
                    vis["tgt_training_mask"] = vis["tgt_metric_mask"].detach()
                if torch.is_tensor(metric_mask):
                    vis_b = int(vis["tgt_gt"].shape[0]) if torch.is_tensor(vis.get("tgt_gt", None)) else 1
                    if len(vis_list) == 1 and vis_b == int(metric_mask.shape[0]):
                        vis["tgt_metric_mask"] = metric_mask.detach()
                    elif j < int(metric_mask.shape[0]):
                        vis["tgt_metric_mask"] = metric_mask[j : j + 1].detach()
                row = _compute_metrics_from_vis(
                    vis,
                    metrics_calc=metrics_calc,
                )
                item_tag = str(tags[j]) if j < len(tags) else str(tag)
                _append_sample_metrics_row(sample_csv, str(group_key), item_tag, row)
                item = {
                    "dataset_name": dataset_name,
                    "tag": item_tag,
                    "row": row,
                }
                if len(current_group_items) < 10:
                    item["vis"] = vis
                current_group_items.append(item)
            pbar.set_postfix(
                groups=len(group_rows),
                targets=num_rows + len(current_group_items),
                refresh=False,
            )
        except Exception as e:
            LOGGER.warning("Skip %s sample idx=%d: %s", dataset, int(i), str(e))
            failure_rows.append(
                {
                    "step": int(step),
                    "sample_idx": int(i),
                    "dataset": str(dataset),
                    "tag": str(tag),
                    "group_key": str(group_key),
                    "error": str(e),
                }
            )
            if "cuda" in str(e).lower():
                raise RuntimeError(
                    f"CUDA error during {dataset} validation at sample idx={int(i)}; "
                    "the CUDA context may be corrupted, so this validation round must fail."
                ) from e

    if failure_rows:
        fail_csv = out_dir / f"validation_failures_{dataset}_step_{int(step):07d}.csv"
        with fail_csv.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["step", "sample_idx", "dataset", "tag", "group_key", "error"],
            )
            writer.writeheader()
            writer.writerows(failure_rows)

    if current_group_key is not None:
        group_row = _finalize_validation_group(
            dataset=dataset,
            step=int(step),
            vis_dir=vis_dir,
            group_key=current_group_key,
            group_items=current_group_items,
        )
        if group_row is not None:
            group_rows.append(group_row)
            num_rows += len(current_group_items)

    if not group_rows:
        if dataset in ("scannetpp", "scanetpp_fisheye", "tat"):
            LOGGER.warning("No validation samples processed for dataset=%s; skip this round.", dataset)
            return
        raise RuntimeError(f"No validation samples processed for dataset={dataset}")

    agg = _aggregate_rows(group_rows)
    agg["step"] = float(step)

    csv_main = out_dir / f"validation_metrics_{dataset}.csv"
    _append_metrics_row(csv_main, agg)

    LOGGER.info(
        "Validation done: dataset=%s groups=%d samples=%d psnr=%.3f ssim=%.4f lpips=%.4f",
        dataset,
        int(len(group_rows)),
        int(num_rows),
        float(agg.get("psnr", float("nan"))),
        float(agg.get("ssim", float("nan"))),
        float(agg.get("lpips", float("nan"))),
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified UniSharp validation")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=[
            "re10k",
            "dl3dv",
            "hm3d",
            "replica",
            "sim",
            "wildrgbd",
            "scannetpp",
            "scanetpp_fisheye",
            "smx_sim_fisheye",
            "tat",
        ],
    )
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--max-index-gap", type=int, default=10)
    p.add_argument("--pair-max-translation-m", type=float, default=0.5)
    p.add_argument("--pair-min-overlap", type=float, default=0.6)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--manifest-file", type=Path, default=None)
    p.add_argument("--manifest-max-groups", type=int, default=0)
    p.add_argument("--validation-batch-size", type=int, default=10)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--fast-metrics", action="store_true", help="Skip LPIPS during validation; keep PSNR/SSIM/depth metrics.")
    p.add_argument("--metric-mask-cache-dir", type=Path, default=default_metric_mask_cache_dir())
    p.add_argument("--max-depth-m", type=float, default=None)
    p.add_argument("--sim-far-depth-invalid-m", type=float, default=None)
    p.add_argument("--sim-far-depth-invalid-max-frac", type=float, default=None)
    p.add_argument("--re10k-pseudo-far-depth-invalid-m", type=float, default=None)
    p.add_argument("--scanetpp-fisheye-far-depth-invalid-m", type=float, default=None)
    p.add_argument("--low-pass-filter-eps", type=float, default=None)
    p.add_argument(
        "--validation-pseudo-depth-root",
        type=Path,
        default=Path("/media/team_data/ML4_team/datasets/sharp/validation_unik3d_pseudo_depth"),
    )
    p.add_argument("--sim-pose-root", type=Path, default=Path("/media/team_data/ML4_team/datasets/smx_sim/30cm"))
    p.add_argument(
        "--re10k-pseudo-depth-root",
        type=Path,
        default=Path("/media/team_data/ML4_team/datasets/nopose/re10k_unik3d_pseudo_depth/test"),
    )
    return p


def main() -> None:
    _configure_torchhub_cache()
    args = build_parser().parse_args()
    _apply_training_depth_config_defaults(args)
    run_validation(args)


if __name__ == "__main__":
    main()

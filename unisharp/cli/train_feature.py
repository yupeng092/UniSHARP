from __future__ import annotations

import csv
import json
import logging
import os
import random
import sys
import time
from dataclasses import fields, is_dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import click
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from unisharp.datasets.re10k import Re10KDataset, re10k_collate, re10k_passthrough
from unisharp.datasets.wildrgbd import WildRGBDDataset, wildrgbd_collate
from unisharp.datasets.dl3dv import DL3DVDataset
from unisharp.datasets.scannetpp_fisheye import ScannetppFisheyeDataset, scannetpp_fisheye_passthrough
from unisharp.datasets.sim_panorama import SimPanoramaDataset
from unisharp.datasets.panogs import PanOGSDataset, panogs_collate
from unisharp.losses import UnisharpLoss, UnisharpLossWeights
from unisharp.models.unisharp_feature import UnisharpFeatureModel, UnisharpFeatureConfig
from unisharp.utils import logging as logging_utils
from unisharp import DEFAULT_MAX_DEPTH_M
from unisharp.utils.gsplat import GSplatRenderer
from unisharp.utils.io import save_image
from unisharp.utils.rayfit_camera import scale_pinhole_intrinsics
from unisharp.utils.unified_vis import save_pair_visualization

from .mixed_sampler import LazyDataLoaderIterator, MixedDatasetSampler  # type: ignore[import]
from .train_utils import warmup_cosine_lr  # type: ignore[import]

LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]


def _default_dataset_manifest_file(name: str) -> Path:
    parent_path = REPO_ROOT.parent / "dataset_manifests" / name
    if parent_path.exists():
        return parent_path
    return REPO_ROOT / "dataset_manifests" / name


DEFAULT_WILDRGBD_ROOTS_FILE = _default_dataset_manifest_file("wildrgbd_roots.txt")


def _multiple_aligned_hw(hw: tuple[int, int], multiple: int) -> tuple[int, int]:
    h, w = int(hw[0]), int(hw[1])
    m = int(multiple)
    if m <= 1:
        return h, w
    out_h = max(m, (h // m) * m)
    out_w = max(m, (w // m) * m)
    return min(out_h, h), min(out_w, w)


def _erp_multiple_aligned_hw(hw: tuple[int, int], multiple: int) -> tuple[int, int]:
    h, w = int(hw[0]), int(hw[1])
    m = int(multiple)
    if m <= 1:
        return h, w
    max_h_from_h = h // m
    max_h_from_w = w // (2 * m)
    h_units = min(max_h_from_h, max_h_from_w)
    if h_units <= 0:
        return h, w
    out_h = h_units * m
    return out_h, 2 * out_h


def _resize_chw_tensor(x: torch.Tensor, dst_hw: tuple[int, int], *, kind: str) -> torch.Tensor:
    if not torch.is_tensor(x) or x.ndim < 3:
        return x
    src_hw = (int(x.shape[-2]), int(x.shape[-1]))
    if src_hw == tuple(int(v) for v in dst_hw):
        return x
    orig_dtype = x.dtype
    flat = x.reshape(-1, int(x.shape[-3]), src_hw[0], src_hw[1]).to(dtype=torch.float32)
    if kind == "image":
        y = F.interpolate(flat, size=dst_hw, mode="bilinear", align_corners=False)
        y = y.round().clamp(0.0, 255.0).to(dtype=orig_dtype) if orig_dtype == torch.uint8 else y.to(dtype=orig_dtype)
    elif kind == "ray":
        y = F.interpolate(flat, size=dst_hw, mode="bilinear", align_corners=False)
        y = y / torch.linalg.vector_norm(y, dim=1, keepdim=True).clamp(min=1e-6)
        y = y.to(dtype=orig_dtype)
    else:
        y = F.interpolate(flat, size=dst_hw, mode="nearest").to(dtype=orig_dtype)
    return y.reshape(*x.shape[:-2], int(dst_hw[0]), int(dst_hw[1])).contiguous()


def _resize_cube_tensor(x: torch.Tensor, dst_hw: tuple[int, int], *, kind: str) -> torch.Tensor:
    if not torch.is_tensor(x) or x.ndim < 4:
        return x
    src_hw = (int(x.shape[-3]), int(x.shape[-2]))
    if src_hw == tuple(int(v) for v in dst_hw):
        return x
    orig_dtype = x.dtype
    channels = int(x.shape[-1])
    flat = x.reshape(-1, src_hw[0], src_hw[1], channels).permute(0, 3, 1, 2).to(dtype=torch.float32)
    if kind == "image":
        y = F.interpolate(flat, size=dst_hw, mode="bilinear", align_corners=False)
        y = y.round().clamp(0.0, 255.0).to(dtype=orig_dtype) if orig_dtype == torch.uint8 else y.to(dtype=orig_dtype)
    else:
        y = F.interpolate(flat, size=dst_hw, mode="nearest").to(dtype=orig_dtype)
    y = y.permute(0, 2, 3, 1)
    return y.reshape(*x.shape[:-3], int(dst_hw[0]), int(dst_hw[1]), channels).contiguous()


def _training_batch_src_hw(batch: Any) -> tuple[int, int] | None:
    for name in ("src_rgb_u8", "src_erp_rgb_u8"):
        value = getattr(batch, name, None)
        if torch.is_tensor(value) and value.ndim >= 3:
            return int(value.shape[-2]), int(value.shape[-1])
    return None


def _scale_fisheye624_params_any(params: torch.Tensor, *, src_hw: tuple[int, int], dst_hw: tuple[int, int]) -> torch.Tensor:
    if tuple(int(x) for x in src_hw) == tuple(int(x) for x in dst_hw):
        return params
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


def _resize_training_batch_to_multiple(batch: Any, multiple: int) -> Any:
    if int(multiple) <= 1 or not is_dataclass(batch):
        return batch
    src_hw = _training_batch_src_hw(batch)
    if src_hw is None:
        return batch

    def _view_hw(prefix: str) -> tuple[int, int] | None:
        for rgb_name in (f"{prefix}_rgb_u8", f"{prefix}_erp_rgb_u8"):
            rgb = getattr(batch, rgb_name, None)
            if torch.is_tensor(rgb) and rgb.ndim >= 3:
                return int(rgb.shape[-2]), int(rgb.shape[-1])
        return None

    def _aligned_view_hw(prefix: str, hw: tuple[int, int]) -> tuple[int, int]:
        is_view_erp = torch.is_tensor(getattr(batch, f"{prefix}_erp_rgb_u8", None))
        return (
            _erp_multiple_aligned_hw(hw, int(multiple))
            if bool(is_view_erp)
            else _multiple_aligned_hw(hw, int(multiple))
        )

    def _field_dst_hw(name: str, value: torch.Tensor) -> tuple[int, int]:
        prefix = "tgt" if name.startswith("tgt_") else "src"
        view_hw = _view_hw(prefix)
        if view_hw is not None:
            return _aligned_view_hw(prefix, view_hw)
        hw = (int(value.shape[-2]), int(value.shape[-1]))
        return _erp_multiple_aligned_hw(hw, int(multiple)) if "_erp_" in name else _multiple_aligned_hw(hw, int(multiple))

    updates: dict[str, Any] = {}
    for field in fields(batch):
        name = field.name
        value = getattr(batch, name)
        if not torch.is_tensor(value):
            continue
        if name.endswith("_rgb_u8") and value.ndim >= 3:
            if "_cube_" in name:
                cube_hw = _multiple_aligned_hw((int(value.shape[-3]), int(value.shape[-2])), int(multiple))
                updates[name] = _resize_cube_tensor(value, cube_hw, kind="image")
            else:
                updates[name] = _resize_chw_tensor(value, _field_dst_hw(name, value), kind="image")
        elif name.endswith("_depth_m") and value.ndim >= 3:
            if "_cube_" in name:
                cube_hw = _multiple_aligned_hw((int(value.shape[-3]), int(value.shape[-2])), int(multiple))
                updates[name] = _resize_cube_tensor(value, cube_hw, kind="depth")
            else:
                updates[name] = _resize_chw_tensor(value, _field_dst_hw(name, value), kind="depth")
        elif name.endswith("_valid_mask") and value.ndim >= 3:
            updates[name] = _resize_chw_tensor(value, _field_dst_hw(name, value), kind="depth")
        elif name.endswith("_rays") and value.ndim >= 3:
            updates[name] = _resize_chw_tensor(value, _field_dst_hw(name, value), kind="ray")

    for intr_name in ("src_intrinsics", "tgt_intrinsics"):
        intr = getattr(batch, intr_name, None)
        if torch.is_tensor(intr):
            prefix = "tgt" if intr_name.startswith("tgt_") else "src"
            view_hw = _view_hw(prefix)
            if view_hw is not None:
                updates[intr_name] = scale_pinhole_intrinsics(
                    intr,
                    src_hw=view_hw,
                    dst_hw=_aligned_view_hw(prefix, view_hw),
                )
    for params_name in ("src_camera_params", "tgt_camera_params"):
        params = getattr(batch, params_name, None)
        if torch.is_tensor(params):
            prefix = "tgt" if params_name.startswith("tgt_") else "src"
            view_hw = _view_hw(prefix)
            if view_hw is not None:
                updates[params_name] = _scale_fisheye624_params_any(
                    params,
                    src_hw=view_hw,
                    dst_hw=_aligned_view_hw(prefix, view_hw),
                )

    return replace(batch, **updates) if updates else batch


def _build_optimizer_param_groups(
    raw_model: UnisharpFeatureModel,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    base_params: list[torch.nn.Parameter] = []
    unik3d_encoder_params: list[torch.nn.Parameter] = []
    unik3d_decoder_params: list[torch.nn.Parameter] = []
    for name, param in raw_model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("feature_extractor.unik3d.pixel_encoder."):
            unik3d_encoder_params.append(param)
        elif name.startswith("second_layer_depth_head."):
            unik3d_decoder_params.append(param)
        elif name.startswith("feature_extractor.unik3d."):
            unik3d_decoder_params.append(param)
        else:
            base_params.append(param)
    return base_params, unik3d_encoder_params, unik3d_decoder_params


def _count_numel(params: list[torch.nn.Parameter]) -> int:
    return int(sum(int(p.numel()) for p in params))


def _configure_torchhub_cache() -> Path:
    torchhub_dir = REPO_ROOT / "checkpoints" / "torchhub"
    torchhub_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(torchhub_dir)
    torch.hub.set_dir(str(torchhub_dir))
    return torchhub_dir


def _ddp_is_enabled() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _ddp_setup(device: str, ddp_timeout_hours: float = 8.0) -> tuple[torch.device, int, int, bool]:
    if not _ddp_is_enabled():
        dev = torch.device(device)
        return dev, 0, 1, True

    if device != "cuda":
        raise RuntimeError("DDP currently supports CUDA only.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available.")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    timeout_hours = max(float(ddp_timeout_hours), 0.25)
    if rank == 0:
        print(
            "[ddp_setup] init_process_group backend=nccl "
            f"world_size={world_size} NCCL_NET={os.environ.get('NCCL_NET', '<unset>')} "
            f"NCCL_IB_DISABLE={os.environ.get('NCCL_IB_DISABLE', '<unset>')}",
            flush=True,
        )
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=timeout_hours))
    if rank == 0:
        print("[ddp_setup] init_process_group done", flush=True)
    dev = torch.device("cuda", local_rank)
    return dev, rank, world_size, (rank == 0)


def _ddp_broadcast_path(p: Path, is_main: bool) -> Path:
    if not _ddp_is_enabled():
        return p
    obj_list: list[str] = [str(p) if is_main else ""]
    dist.broadcast_object_list(obj_list, src=0)
    return Path(obj_list[0])


def _ddp_broadcast_str(value: str, is_main: bool) -> str:
    if not _ddp_is_enabled():
        return value
    obj_list: list[str] = [str(value) if is_main else ""]
    dist.broadcast_object_list(obj_list, src=0)
    return str(obj_list[0])


def _ddp_any_bool(flag: bool, device: torch.device) -> bool:
    if not _ddp_is_enabled():
        return bool(flag)
    x = torch.tensor(1 if flag else 0, device=device, dtype=torch.int32)
    dist.all_reduce(x, op=dist.ReduceOp.MAX)
    return bool(int(x.item()) != 0)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _is_oom_exception(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    oom_markers = (
        "out of memory",
        "cuda error: out of memory",
        "cublas_status_alloc_failed",
        "cudnn_status_alloc_failed",
        "defaultcpuallocator",
    )
    return any(marker in msg for marker in oom_markers)


def _ddp_barrier(device: torch.device) -> None:
    if not _ddp_is_enabled():
        return
    if device.type == "cuda" and device.index is not None:
        dist.barrier(device_ids=[device.index])
    else:
        dist.barrier()


def _maybe_set_dataset_epoch(dataset: Any, epoch: int) -> None:
    set_epoch = getattr(dataset, "set_epoch", None)
    if callable(set_epoch):
        set_epoch(int(epoch))


def _ddp_mean(x: torch.Tensor) -> torch.Tensor:
    if not _ddp_is_enabled():
        return x
    y = x.detach().clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    y = y / float(dist.get_world_size())
    return y


def _save_train_vis(
    out_dir: Path,
    step: int,
    src_gt: torch.Tensor,
    src_pred: torch.Tensor,
    src_alpha: torch.Tensor,
    tgt_gt: torch.Tensor,
    tgt_pred: torch.Tensor,
    tgt_alpha: torch.Tensor,
    src_gt_depth: torch.Tensor | None = None,
    tgt_gt_depth: torch.Tensor | None = None,
    src_pred_depth: torch.Tensor | None = None,
    tgt_pred_depth: torch.Tensor | None = None,
    src_unik3d_depth: torch.Tensor | None = None,
    tgt_unik3d_depth: torch.Tensor | None = None,
    dataset_name: str | None = None,
    scene: str | None = None,
    src_idx: int | None = None,
    tgt_idx: int | None = None,
    src_pose_w2c: torch.Tensor | None = None,
    tgt_pose_w2c: torch.Tensor | None = None,
    src_metric_mask: torch.Tensor | None = None,
    tgt_metric_mask: torch.Tensor | None = None,
    src_cube_gt_u8: torch.Tensor | None = None,
    src_cube_pred_linear: torch.Tensor | None = None,
    src_cube_alpha: torch.Tensor | None = None,
    tgt_cube_gt_u8: torch.Tensor | None = None,
    tgt_cube_pred_linear: torch.Tensor | None = None,
    tgt_cube_alpha: torch.Tensor | None = None,
) -> None:
    vis_dir = out_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Saving train visualization: %s", str(vis_dir / f"step_{int(step):07d}.png"))
    save_pair_visualization(
        vis_dir / f"step_{int(step):07d}.png",
        src_gt=src_gt,
        src_pred=src_pred,
        src_alpha=src_alpha,
        tgt_gt=tgt_gt,
        tgt_pred=tgt_pred,
        tgt_alpha=tgt_alpha,
        src_gt_depth=src_gt_depth,
        tgt_gt_depth=tgt_gt_depth,
        src_pred_depth=src_pred_depth,
        tgt_pred_depth=tgt_pred_depth,
        src_unik3d_depth=src_unik3d_depth,
        tgt_unik3d_depth=tgt_unik3d_depth,
        dataset_name=dataset_name,
        scene=scene,
        step=int(step),
        src_idx=src_idx,
        tgt_idx=tgt_idx,
        src_pose_w2c=src_pose_w2c,
        tgt_pose_w2c=tgt_pose_w2c,
        src_cube_gt_u8=src_cube_gt_u8,
        src_cube_pred_linear=src_cube_pred_linear,
        src_cube_alpha=src_cube_alpha,
        tgt_cube_gt_u8=tgt_cube_gt_u8,
        tgt_cube_pred_linear=tgt_cube_pred_linear,
        tgt_cube_alpha=tgt_cube_alpha,
    )


def _read_nonempty_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _resolve_manifest_file(manifest_dir: Path | None, filename: str) -> Path | None:
    if manifest_dir is None:
        return None
    path = Path(manifest_dir) / filename
    return path if path.exists() else None


@click.command()
@click.option("--data-root-re10k", type=click.Path(path_type=Path, exists=True), default=None)
@click.option("--data-root-hm3d", type=click.Path(path_type=Path, exists=True), default=Path("/media/team_data/ML4_team/datasets/panogs"))
@click.option("--data-root-sim", type=click.Path(path_type=Path, exists=True), default=Path("/media/team_data/ML4_team/datasets/smx_sim"))
@click.option("--sim-pose-root", type=click.Path(path_type=Path, exists=True), default=Path("/media/team_data/ML4_team/datasets/smx_sim/30cm"))
@click.option("--data-root-wildrgbd", type=click.Path(path_type=Path, exists=True), default=None)
@click.option("--wild-roots-file", type=click.Path(path_type=Path, exists=True, dir_okay=False), default=DEFAULT_WILDRGBD_ROOTS_FILE)
@click.option("--data-root-dl3dv", type=click.Path(path_type=Path, exists=True), default=Path("/media/team_data/ML4_team/datasets/sharp/DL3DV-ALL-960P"))
@click.option("--data-root-dl3dv-depth", type=click.Path(path_type=Path, exists=True), default=Path("/media/team_data/ML4_team/datasets/sharp/DL3DV-ALL-960P_da3_outputs"))
@click.option("--data-root-scanetpp", type=click.Path(path_type=Path, exists=True), default=Path("/media/team_data/ML4_team/datasets/scan"))
@click.option("--dataset-manifest-dir", type=click.Path(path_type=Path, file_okay=False), default=None)
@click.option("--out-root", type=click.Path(path_type=Path, file_okay=False), required=True)
@click.option("--run-name", type=str, default=None)
@click.option("--steps", type=int, default=1000000)
@click.option("--batch-size", type=int, default=2)
@click.option("--num-workers", type=int, default=1)
@click.option("--warmup", type=int, default=75000)
@click.option("--lr0", type=float, default=1.2e-4)
@click.option("--lr1", type=float, default=1.6e-5)
@click.option("--unik3d-lr0", type=float, default=2.5e-5, help="UniK3D decoder/head peak LR.")
@click.option("--unik3d-lr1", type=float, default=2.5e-6, help="UniK3D decoder/head final LR.")
@click.option("--unik3d-encoder-lr0", type=float, default=1.5e-6, help="UniK3D pixel_encoder peak LR.")
@click.option("--unik3d-encoder-lr1", type=float, default=1.5e-7, help="UniK3D pixel_encoder final LR.")
@click.option("--grad-clip-norm", type=float, default=1.0, show_default=True)
@click.option("--max-step-grad-norm", type=float, default=100000.0, show_default=True, help="Skip optimizer step when pre-clip grad norm exceeds this value. 0 disables.")
@click.option("--max-depth-m", type=float, default=DEFAULT_MAX_DEPTH_M, show_default=True)
@click.option("--sim-far-depth-invalid-m", type=float, default=30.0, show_default=True)
@click.option("--sim-far-depth-invalid-max-frac", type=float, default=1.0, show_default=True)
@click.option("--sim-max-long-edge", type=int, default=512, show_default=True, help="Resize SIM ERP frames before cubemap conversion. 0 keeps native resolution.")
@click.option("--train-resize-multiple", type=int, default=256, show_default=True, help="Before model forward, downsize training inputs to the largest H/W divisible by this value. 0 disables.")
@click.option("--pinhole-train-size", type=int, default=0, show_default=True, help="Resize pinhole training datasets to NxN before model forward. 0 keeps dataset native resolution.")
@click.option("--scanetpp-fisheye-far-depth-invalid-m", type=float, default=30.0, show_default=True)
@click.option("--max-index-gap", type=int, default=10)
@click.option("--device", type=str, default="cuda")
@click.option("--render-low-pass-filter-eps", type=float, default=1e-2, show_default=True)
@click.option("--ddp-timeout-hours", type=float, default=8.0)
@click.option("--save-every", type=int, default=5000)
@click.option("--log-every", type=int, default=50)
@click.option("--vis-every", type=int, default=500)
@click.option("--unik3d-backbone", type=click.Choice(["vitb", "vitl"]), default="vitl")
@click.option("--unik3d-resolution-level", type=click.IntRange(0, 9), default=0, show_default=True)
@click.option("--initializer-stride", type=click.IntRange(1, 2), default=1)
@click.option("--initializer-scale-factor", type=float, default=1.5, show_default=True)
@click.option("--lambda-aux-ray", type=float, default=3.0)
@click.option("--lambda-aux-depth-scale", type=float, default=3.0)
@click.option("--lambda-aux-depth2-scale", type=float, default=1.0)
@click.option("--lambda-color", type=float, default=1.0)
@click.option("--lambda-alpha", type=float, default=1.5)
@click.option("--alpha-tail-min", type=float, default=0.99, show_default=True, help="Alpha value below which local tail coverage loss is applied.")
@click.option("--alpha-tail-weight", type=float, default=0.0, show_default=True, help="Extra normalized tail weight for local low-alpha holes.")
@click.option("--lambda-percep", type=float, default=1.0)
@click.option("--lambda-depth", type=float, default=0.5)
@click.option("--lambda-tv", type=float, default=1.0)
@click.option("--lambda-grad", type=float, default=1.0)
@click.option("--lambda-grad-img", type=float, default=0.2)
@click.option("--lambda-edge-rgb", type=float, default=0.0, show_default=True, help="Weight for GT RGB edge-band gradient matching.")
@click.option("--lambda-delta", type=float, default=1.0)
@click.option("--lambda-delta-rho", type=float, default=0.01, show_default=True)
@click.option("--lambda-splat", type=float, default=1.0)
@click.option("--lambda-edge-splat", type=float, default=0.0, show_default=True, help="Weight for stricter projected-sigma penalty on GT depth-edge bands.")
@click.option("--lambda-grid", type=float, default=0.05, show_default=True, help="Weight for Gaussian-grid 2x2 checkerboard residual regularization.")
@click.option("--delta-clip", type=float, default=10.0, show_default=True)
@click.option("--raw-delta-clip", type=float, default=400.0, show_default=True)
@click.option("--raw-delta-rho-clip", type=float, default=5.0, show_default=True)
@click.option("--delta-rho-limit", type=float, default=2.0, show_default=True)
@click.option("--splat-sigma-min", type=float, default=1e-1, show_default=True, help="Minimum projected screen-space variance for L_splat.")
@click.option("--splat-sigma-max", type=float, default=1e2, show_default=True, help="Maximum projected screen-space variance for L_splat.")
@click.option("--edge-splat-sigma-max", type=float, default=2.0, show_default=True, help="Maximum projected variance on depth-edge bands for L_edge_splat.")
@click.option("--depth-edge-log-threshold", type=float, default=0.05, show_default=True, help="Log-depth jump threshold used to build L_edge_splat edge bands.")
@click.option("--depth-edge-dilate-px", type=int, default=2, show_default=True, help="Dilation radius in pixels for L_edge_splat depth-edge bands.")
@click.option("--target-mask-erode-px", type=int, default=0, show_default=True, help="Erode source-visible target masks by this many pixels before target supervision.")
@click.option("--dataset-weight-re10k", type=float, default=1.0)
@click.option("--dataset-weight-hm3d", type=float, default=1.0)
@click.option("--dataset-weight-sim", type=float, default=1.0)
@click.option("--dataset-weight-wildrgbd", type=float, default=1.0)
@click.option("--dataset-weight-dl3dv", type=float, default=1.0)
@click.option("--dataset-weight-scanetpp", type=float, default=0.0)
@click.option(
    "--re10k-pseudo-depth-root",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("/media/team_data/ML4_team/datasets/nopose/re10k_unik3d_pseudo_depth"),
)
@click.option("--re10k-pseudo-depth-autogen/--no-re10k-pseudo-depth-autogen", default=True)
@click.option("--re10k-pseudo-depth-backbone", type=click.Choice(["vitb", "vitl"]), default="vitl")
@click.option("--re10k-pseudo-depth-device", type=str, default="cpu")
@click.option("--re10k-pseudo-lock-timeout-sec", type=float, default=120.0)
@click.option("--re10k-pseudo-lock-stale-sec", type=float, default=1800.0)
@click.option("--re10k-pseudo-far-depth-invalid-m", type=float, default=30.0)
@click.option("--seed", type=int, default=None)
@click.option("-v", "--verbose", is_flag=True)
def train_feature_cli(
    data_root_re10k: Path | None,
    data_root_hm3d: Path | None,
    data_root_sim: Path | None,
    sim_pose_root: Path | None,
    data_root_wildrgbd: Path | None,
    wild_roots_file: Path,
    data_root_dl3dv: Path | None,
    data_root_dl3dv_depth: Path | None,
    data_root_scanetpp: Path | None,
    dataset_manifest_dir: Path | None,
    out_root: Path,
    run_name: str | None,
    steps: int,
    batch_size: int,
    num_workers: int,
    warmup: int,
    lr0: float,
    lr1: float,
    unik3d_lr0: float,
    unik3d_lr1: float,
    unik3d_encoder_lr0: float,
    unik3d_encoder_lr1: float,
    grad_clip_norm: float,
    max_step_grad_norm: float,
    max_depth_m: float,
    sim_far_depth_invalid_m: float,
    sim_far_depth_invalid_max_frac: float,
    sim_max_long_edge: int,
    train_resize_multiple: int,
    pinhole_train_size: int,
    scanetpp_fisheye_far_depth_invalid_m: float,
    max_index_gap: int,
    device: str,
    render_low_pass_filter_eps: float,
    ddp_timeout_hours: float,
    save_every: int,
    log_every: int,
    vis_every: int,
    unik3d_backbone: str,
    unik3d_resolution_level: int,
    initializer_stride: int,
    initializer_scale_factor: float,
    lambda_aux_ray: float,
    lambda_aux_depth_scale: float,
    lambda_aux_depth2_scale: float,
    lambda_color: float,
    lambda_alpha: float,
    alpha_tail_min: float,
    alpha_tail_weight: float,
    lambda_percep: float,
    lambda_depth: float,
    lambda_tv: float,
    lambda_grad: float,
    lambda_grad_img: float,
    lambda_edge_rgb: float,
    lambda_delta: float,
    lambda_delta_rho: float,
    lambda_splat: float,
    lambda_edge_splat: float,
    lambda_grid: float,
    delta_clip: float,
    raw_delta_clip: float,
    raw_delta_rho_clip: float,
    delta_rho_limit: float,
    splat_sigma_min: float,
    splat_sigma_max: float,
    edge_splat_sigma_max: float,
    depth_edge_log_threshold: float,
    depth_edge_dilate_px: int,
    target_mask_erode_px: int,
    dataset_weight_re10k: float,
    dataset_weight_hm3d: float,
    dataset_weight_sim: float,
    dataset_weight_wildrgbd: float,
    dataset_weight_dl3dv: float,
    dataset_weight_scanetpp: float,
    re10k_pseudo_depth_root: Path,
    re10k_pseudo_depth_autogen: bool,
    re10k_pseudo_depth_backbone: str,
    re10k_pseudo_depth_device: str,
    re10k_pseudo_lock_timeout_sec: float,
    re10k_pseudo_lock_stale_sec: float,
    re10k_pseudo_far_depth_invalid_m: float,
    seed: int | None,
    verbose: bool,
) -> None:
    detach_init_layer0_distance = True

    log_level = logging.DEBUG if verbose else logging.INFO
    logging_utils.configure(log_level)
    if float(max_depth_m) <= 0.0:
        raise ValueError("--max-depth-m must be positive.")
    if float(grad_clip_norm) <= 0.0:
        raise ValueError("--grad-clip-norm must be positive.")
    if float(max_step_grad_norm) < 0.0:
        raise ValueError("--max-step-grad-norm must be non-negative.")
    if float(render_low_pass_filter_eps) < 0.0:
        raise ValueError("--render-low-pass-filter-eps must be non-negative.")
    if not (0.0 <= float(sim_far_depth_invalid_max_frac) <= 1.0):
        raise ValueError("--sim-far-depth-invalid-max-frac must be in [0, 1].")
    if int(sim_max_long_edge) < 0:
        raise ValueError("--sim-max-long-edge must be non-negative.")
    if int(train_resize_multiple) < 0:
        raise ValueError("--train-resize-multiple must be non-negative.")
    if int(pinhole_train_size) < 0:
        raise ValueError("--pinhole-train-size must be non-negative.")
    if float(scanetpp_fisheye_far_depth_invalid_m) < 0.0:
        raise ValueError("--scanetpp-fisheye-far-depth-invalid-m must be non-negative.")
    if float(delta_clip) < 0.0:
        raise ValueError("--delta-clip must be non-negative.")
    if float(raw_delta_clip) < 0.0:
        raise ValueError("--raw-delta-clip must be non-negative.")
    if float(raw_delta_rho_clip) < 0.0:
        raise ValueError("--raw-delta-rho-clip must be non-negative.")
    if float(lambda_grid) < 0.0:
        raise ValueError("--lambda-grid must be non-negative.")
    if float(lambda_edge_rgb) < 0.0:
        raise ValueError("--lambda-edge-rgb must be non-negative.")
    if float(lambda_edge_splat) < 0.0:
        raise ValueError("--lambda-edge-splat must be non-negative.")
    if float(edge_splat_sigma_max) < 0.0:
        raise ValueError("--edge-splat-sigma-max must be non-negative.")
    if float(depth_edge_log_threshold) < 0.0:
        raise ValueError("--depth-edge-log-threshold must be non-negative.")
    if int(depth_edge_dilate_px) < 0:
        raise ValueError("--depth-edge-dilate-px must be non-negative.")
    if int(target_mask_erode_px) < 0:
        raise ValueError("--target-mask-erode-px must be non-negative.")
    if not (0.0 <= float(alpha_tail_min) <= 1.0):
        raise ValueError("--alpha-tail-min must be in [0, 1].")
    if float(alpha_tail_weight) < 0.0:
        raise ValueError("--alpha-tail-weight must be non-negative.")
    if float(delta_rho_limit) < 0.0:
        raise ValueError("--delta-rho-limit must be non-negative.")
    if float(splat_sigma_min) < 0.0:
        raise ValueError("--splat-sigma-min must be non-negative.")
    if float(splat_sigma_max) <= float(splat_sigma_min):
        raise ValueError("--splat-sigma-max must be greater than --splat-sigma-min.")
    dev, rank, world_size, is_main = _ddp_setup(device, ddp_timeout_hours=ddp_timeout_hours)

    if seed is not None:
        s = int(seed)
        random.seed(s + rank)
        np.random.seed(s + rank)
        torch.manual_seed(s + rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s + rank)

    if is_main and (run_name is None or run_name.strip() == ""):
        run_name = f"unified_feature_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if run_name is None:
        run_name = "unified_feature_ddp"
    out_dir = _ddp_broadcast_path(Path(out_root) / run_name, is_main=is_main)
    logging_utils.configure(log_level)
    if not is_main:
        logging.getLogger().setLevel(logging.WARNING)
        LOGGER.setLevel(logging.WARNING)
    _configure_torchhub_cache()
    re10k_enabled_for_train = bool(float(dataset_weight_re10k) > 0.0)
    hm3d_enabled_for_train = bool(float(dataset_weight_hm3d) > 0.0)
    sim_enabled_for_train = bool(float(dataset_weight_sim) > 0.0)
    dl3dv_enabled_for_train = bool(float(dataset_weight_dl3dv) > 0.0)
    scanetpp_enabled_for_train = bool(float(dataset_weight_scanetpp) > 0.0)
    wild_roots = _read_nonempty_lines(wild_roots_file) if wild_roots_file.exists() else []
    re10k_manifest = _resolve_manifest_file(dataset_manifest_dir, "re10k_train_chunks.txt")
    hm3d_manifest = _resolve_manifest_file(dataset_manifest_dir, "hm3d_train_scenes.txt")
    sim_manifest = _resolve_manifest_file(dataset_manifest_dir, "sim_train_scenes.txt")
    wildrgbd_manifest = _resolve_manifest_file(dataset_manifest_dir, "wildrgbd_train_scenes.txt")
    dl3dv_manifest = _resolve_manifest_file(dataset_manifest_dir, "dl3dv_train_scenes.txt")
    scanetpp_manifest = _resolve_manifest_file(dataset_manifest_dir, "scanetpp_fisheye_train_scenes.txt")
    wildrgbd_enabled_for_train = bool(
        ((data_root_wildrgbd is not None) or bool(wild_roots)) and (float(dataset_weight_wildrgbd) > 0.0)
    )
    if re10k_enabled_for_train and data_root_re10k is None:
        raise ValueError("dataset_weight_re10k>0 but --data-root-re10k is not provided.")
    if hm3d_enabled_for_train and data_root_hm3d is None:
        raise ValueError("dataset_weight_hm3d>0 but --data-root-hm3d is not provided.")
    if sim_enabled_for_train and (data_root_sim is None or sim_pose_root is None):
        raise ValueError("dataset_weight_sim>0 but --data-root-sim / --sim-pose-root is missing.")
    if sim_enabled_for_train and sim_manifest is None:
        raise ValueError("dataset_weight_sim>0 but sim_train_scenes.txt is missing from --dataset-manifest-dir.")
    if float(dataset_weight_wildrgbd) > 0.0 and (data_root_wildrgbd is None) and (not wild_roots):
        raise ValueError("dataset_weight_wildrgbd>0 but neither --data-root-wildrgbd nor --wild-roots-file is provided.")
    if dl3dv_enabled_for_train and (data_root_dl3dv is None or data_root_dl3dv_depth is None):
        raise ValueError("dataset_weight_dl3dv>0 but --data-root-dl3dv / --data-root-dl3dv-depth is missing.")
    if scanetpp_enabled_for_train and data_root_scanetpp is None:
        raise ValueError("dataset_weight_scanetpp>0 but --data-root-scanetpp is missing.")

    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info(
            "Training start: out=%s branch=gt-override scratch_unik3d_pretrained backbone=%s steps=%d batch=%d",
            str(out_dir),
            str(unik3d_backbone),
            int(steps),
            int(batch_size),
        )
        LOGGER.info(
            "Loss weights: color=%.3g alpha=%.3g depth=%.3g percep=%.3g aux_ray=%.3g aux_depth0=%.3g aux_depth1=%.3g",
            float(lambda_color),
            float(lambda_alpha),
            float(lambda_depth),
            float(lambda_percep),
            float(lambda_aux_ray),
            float(lambda_aux_depth_scale),
            float(lambda_aux_depth2_scale),
        )

    dataset_seed = int(seed) if seed is not None else 12345
    pinhole_output_h = int(pinhole_train_size) if int(pinhole_train_size) > 0 else None
    pinhole_output_w = int(pinhole_train_size) if int(pinhole_train_size) > 0 else None

    re10k_ds = None
    if re10k_enabled_for_train:
        re10k_ds = Re10KDataset(
            root=data_root_re10k,
            chunks_file=re10k_manifest,
            split="train",
            min_frame_gap=1,
            max_frame_gap=int(max_index_gap),
            pair_max_translation_m=0.5,
            pair_min_overlap=0.6,
            output_h=pinhole_output_h,
            output_w=pinhole_output_w,
            shuffle_chunk=True,
            shuffle_example=True,
            ddp_rank=rank,
            ddp_world_size=world_size,
            pseudo_depth_root=re10k_pseudo_depth_root,
            pseudo_depth_autogen=bool(re10k_pseudo_depth_autogen),
            pseudo_depth_backbone=str(re10k_pseudo_depth_backbone),
            pseudo_depth_device=str(re10k_pseudo_depth_device),
            pseudo_lock_timeout_sec=float(re10k_pseudo_lock_timeout_sec),
            pseudo_lock_stale_sec=float(re10k_pseudo_lock_stale_sec),
            batch_size_hint=int(batch_size),
            depth_max_m=float(max_depth_m),
            pseudo_far_depth_invalid_m=float(re10k_pseudo_far_depth_invalid_m),
            seed=dataset_seed,
        )
    hm3d_train_root = None
    if data_root_hm3d is not None:
        hm3d_train_root = data_root_hm3d / "train" if (data_root_hm3d / "train").exists() else data_root_hm3d

    hm3d_ds = None
    if hm3d_enabled_for_train:
        hm3d_ds = PanOGSDataset(
            root=hm3d_train_root,
            index_manifest_path=hm3d_manifest,
            src_tgt_max_index_gap=int(max_index_gap),
            use_cubemap_supervision=True,
            pair_sampling=True,
            pair_max_translation_m=0.5,
            pair_min_depth_overlap=0.6,
            pair_overlap_face_w=64,
            pair_overlap_margin=1.05,
            pair_max_tries=48,
            depth_max_m=float(max_depth_m),
        )
    sim_ds = None
    if sim_enabled_for_train:
        sim_ds = SimPanoramaDataset(
            root=data_root_sim,
            pose_root=sim_pose_root,
            scene_list_file=sim_manifest,
            max_index_gap=int(max_index_gap),
            pair_max_translation_m=0.5,
            pair_min_depth_overlap=0.6,
            pairs_per_chunk=15,
            chunk_size=30,
            shuffle_scene=True,
            ddp_rank=rank,
            ddp_world_size=world_size,
            depth_max_m=float(max_depth_m),
            far_depth_invalid_m=float(sim_far_depth_invalid_m),
            far_depth_invalid_max_frac=float(sim_far_depth_invalid_max_frac),
            max_long_edge=int(sim_max_long_edge),
            seed=dataset_seed,
        )
    wildrgbd_ds = None
    if wildrgbd_enabled_for_train:
        wild_dataset_roots = [Path(p) for p in wild_roots]
        if data_root_wildrgbd is not None:
            wild_dataset_roots.append(data_root_wildrgbd)
        wildrgbd_ds = WildRGBDDataset(
            root=None,
            scene_list_file=wildrgbd_manifest,
            split="scenes",
            min_frame_gap=1,
            max_frame_gap=int(max_index_gap),
            pair_max_translation_m=0.5,
            pair_min_overlap=0.6,
            output_h=pinhole_output_h,
            output_w=pinhole_output_w,
            shuffle_scene=True,
            shuffle_frame=False,
            ddp_rank=rank,
            ddp_world_size=world_size,
            roots=wild_dataset_roots,
            depth_max_m=float(max_depth_m),
            seed=dataset_seed,
        )
    dl3dv_ds = None
    if dl3dv_enabled_for_train:
        dl3dv_ds = DL3DVDataset(
            root=data_root_dl3dv,
            depth_root=data_root_dl3dv_depth,
            scene_specs_file=dl3dv_manifest,
            min_frame_gap=1,
            max_frame_gap=int(max_index_gap),
            pair_max_translation_m=0.5,
            pair_min_overlap=0.6,
            output_h=pinhole_output_h,
            output_w=pinhole_output_w,
            shuffle_scene=True,
            shuffle_frame=False,
            ddp_rank=rank,
            ddp_world_size=world_size,
            batch_size_hint=int(batch_size),
            depth_max_m=float(max_depth_m),
            seed=dataset_seed,
        )

    scanetpp_ds = None
    if scanetpp_enabled_for_train:
        scanetpp_ds = ScannetppFisheyeDataset(
            root=data_root_scanetpp,
            scene_list_file=scanetpp_manifest,
            min_frame_gap=1,
            max_frame_gap=int(max_index_gap),
            pair_max_translation_m=0.5,
            shuffle_scene=True,
            shuffle_frame=False,
            ddp_rank=rank,
            ddp_world_size=world_size,
            batch_size_hint=int(batch_size),
            depth_max_m=float(max_depth_m),
            far_depth_invalid_m=float(scanetpp_fisheye_far_depth_invalid_m),
            seed=dataset_seed,
        )

    hm3d_sampler = None
    if hm3d_ds is not None and _ddp_is_enabled():
        hm3d_sampler = DistributedSampler(hm3d_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)

    re10k_num_workers = int(num_workers)
    if re10k_ds is not None and bool(re10k_pseudo_depth_autogen) and re10k_num_workers > 0:
        re10k_num_workers = 0
        if is_main:
            LOGGER.warning(
                "RE10K pseudo-depth auto-generate enabled: force re10k dataloader num_workers=%d (requested=%d).",
                int(re10k_num_workers),
                int(num_workers),
            )
    if re10k_ds is not None and batch_size > 1 and re10k_num_workers > 0:
        re10k_num_workers = 0
        if is_main:
            LOGGER.warning(
                "Dynamic-resolution RE10K batching requires ordered same-resolution samples: force re10k dataloader num_workers=%d (requested=%d).",
                int(re10k_num_workers),
                int(num_workers),
            )

    highres_pin_memory = os.environ.get("HIGHRES_TRAIN_PIN_MEMORY", "0").strip().lower() in {"1", "true", "yes", "on"}
    standard_pin_memory = os.environ.get("TRAIN_PIN_MEMORY", "1").strip().lower() in {"1", "true", "yes", "on"}
    try:
        train_prefetch_factor = max(1, int(os.environ.get("TRAIN_PREFETCH_FACTOR", "1").strip()))
    except Exception:
        train_prefetch_factor = 1
    def _loader_worker_kwargs(worker_count: int, *, pin_memory: bool) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "num_workers": int(worker_count),
            "pin_memory": bool(pin_memory),
        }
        if int(worker_count) > 0:
            kwargs["prefetch_factor"] = int(train_prefetch_factor)
        return kwargs

    re10k_dl = None
    if re10k_ds is not None:
        re10k_dl = DataLoader(
            re10k_ds,
            batch_size=None,
            **_loader_worker_kwargs(re10k_num_workers, pin_memory=standard_pin_memory),
            collate_fn=re10k_passthrough,
        )

    hm3d_dl = None
    if hm3d_ds is not None:
        hm3d_dl = DataLoader(
            hm3d_ds,
            batch_size=batch_size,
            shuffle=(hm3d_sampler is None),
            sampler=hm3d_sampler,
            **_loader_worker_kwargs(num_workers, pin_memory=highres_pin_memory),
            collate_fn=panogs_collate,
        )

    sim_dl = None
    if sim_ds is not None:
        sim_dl = DataLoader(
            sim_ds,
            batch_size=batch_size,
            **_loader_worker_kwargs(num_workers, pin_memory=highres_pin_memory),
            collate_fn=panogs_collate,
        )

    wildrgbd_dl = None
    if wildrgbd_ds is not None:
        wildrgbd_dl = DataLoader(
            wildrgbd_ds,
            batch_size=batch_size,
            **_loader_worker_kwargs(num_workers, pin_memory=standard_pin_memory),
            collate_fn=wildrgbd_collate,
        )

    dl3dv_dl = None
    if dl3dv_ds is not None:
        dl3dv_dl = DataLoader(
            dl3dv_ds,
            batch_size=None,
            **_loader_worker_kwargs(num_workers, pin_memory=standard_pin_memory),
            collate_fn=re10k_passthrough,
        )

    scanetpp_dl = None
    if scanetpp_ds is not None:
        scanetpp_dl = DataLoader(
            scanetpp_ds,
            batch_size=None,
            **_loader_worker_kwargs(num_workers, pin_memory=highres_pin_memory),
            collate_fn=scannetpp_fisheye_passthrough,
        )

    candidate_datasets: dict[str, Any] = {}
    candidate_dataloaders: dict[str, DataLoader] = {}
    candidate_weights: dict[str, float] = {}
    if re10k_ds is not None and re10k_dl is not None:
        candidate_datasets["re10k"] = re10k_ds
        candidate_dataloaders["re10k"] = re10k_dl
        candidate_weights["re10k"] = float(dataset_weight_re10k)
    if hm3d_ds is not None and hm3d_dl is not None:
        candidate_datasets["hm3d"] = hm3d_ds
        candidate_dataloaders["hm3d"] = hm3d_dl
        candidate_weights["hm3d"] = float(dataset_weight_hm3d)
    if sim_ds is not None and sim_dl is not None:
        candidate_datasets["sim"] = sim_ds
        candidate_dataloaders["sim"] = sim_dl
        candidate_weights["sim"] = float(dataset_weight_sim)
    if wildrgbd_ds is not None and wildrgbd_dl is not None:
        candidate_datasets["wildrgbd"] = wildrgbd_ds
        candidate_dataloaders["wildrgbd"] = wildrgbd_dl
        candidate_weights["wildrgbd"] = float(dataset_weight_wildrgbd)
    if dl3dv_ds is not None and dl3dv_dl is not None:
        candidate_datasets["dl3dv"] = dl3dv_ds
        candidate_dataloaders["dl3dv"] = dl3dv_dl
        candidate_weights["dl3dv"] = float(dataset_weight_dl3dv)
    if scanetpp_ds is not None and scanetpp_dl is not None:
        candidate_datasets["scanetpp_fisheye"] = scanetpp_ds
        candidate_dataloaders["scanetpp_fisheye"] = scanetpp_dl
        candidate_weights["scanetpp_fisheye"] = float(dataset_weight_scanetpp)

    datasets: dict[str, Any] = {}
    dataloaders: dict[str, DataLoader] = {}
    sampling: dict[str, float] = {}
    for name, w in candidate_weights.items():
        if float(w) > 0.0:
            datasets[name] = candidate_datasets[name]
            dataloaders[name] = candidate_dataloaders[name]
            sampling[name] = float(w)
        elif is_main:
            LOGGER.warning("Skip dataset in mixed sampler: %s (weight=%.4f <= 0)", name, float(w))

    if len(datasets) == 0:
        raise ValueError("No dataset selected for mixed sampler (all dataset weights <= 0).")
    for name, dataset in datasets.items():
        _maybe_set_dataset_epoch(dataset, 0)
    iterators = {name: LazyDataLoaderIterator(dl) for name, dl in dataloaders.items()}
    sampler_seed = int(seed + rank) if seed is not None else int(12345 + rank)
    sampler = MixedDatasetSampler(
        datasets=datasets,
        weights=sampling,
        iterators=iterators,
        seed=sampler_seed,
    )

    config = UnisharpFeatureConfig(
        unik3d_backbone=unik3d_backbone,
        unik3d_resolution_level=int(unik3d_resolution_level),
        initializer_stride=int(initializer_stride),
        initializer_scale_factor=float(initializer_scale_factor),
        detach_init_layer0_distance=bool(detach_init_layer0_distance),
        delta_rho_limit=float(delta_rho_limit),
    )
    setattr(config, "max_distance_m", float(max_depth_m))
    
    model = UnisharpFeatureModel(config).to(dev).train()

    if _ddp_is_enabled():
        model = DDP(
            model,
            device_ids=[dev.index],
            output_device=dev.index,
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
        )

    raw_model = model.module if isinstance(model, DDP) else model
    base_params, unik3d_encoder_params, unik3d_decoder_params = _build_optimizer_param_groups(raw_model)
    unik3d_params = unik3d_encoder_params + unik3d_decoder_params
    trainable_params = base_params + unik3d_params
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters found.")
    if len(unik3d_params) == 0:
        raise RuntimeError(
            "No UniK3D parameters were collected for the default unfreeze training path. "
            "Please check parameter naming."
        )
    depth_head_params = [p for p in raw_model.second_layer_depth_head.parameters() if p.requires_grad]
    if len(depth_head_params) == 0:
        raise RuntimeError("Depth heads have no trainable parameters; depth branch would not train.")

    opt_groups: list[dict[str, Any]] = [{"params": base_params, "lr": float(lr0), "group_name": "base"}]
    if len(unik3d_encoder_params) > 0:
        opt_groups.append(
            {
                "params": unik3d_encoder_params,
                "lr": float(unik3d_encoder_lr0),
                "group_name": "unik3d_encoder",
            }
        )
    if len(unik3d_decoder_params) > 0:
        opt_groups.append(
            {
                "params": unik3d_decoder_params,
                "lr": float(unik3d_lr0),
                "group_name": "unik3d_decoder",
            }
        )
    opt = torch.optim.Adam(opt_groups)
    if is_main:
        LOGGER.info(
            "Model ready: scratch heads, pretrained UniK3D, trainable_params=%d",
            _count_numel(trainable_params),
        )
    if dev.type == "cuda":
        scaler = torch.amp.GradScaler("cuda", enabled=True)
    else:
        scaler = torch.amp.GradScaler("cpu", enabled=False)

    renderer = GSplatRenderer(
        color_space="sRGB",
        background_color="black",
        low_pass_filter_eps=float(render_low_pass_filter_eps),
    ).to(dev)

    loss_w = UnisharpLossWeights(
        lambda_color=float(lambda_color),
        lambda_alpha=float(lambda_alpha),
        lambda_percep=float(lambda_percep),
        lambda_depth=float(lambda_depth),
        lambda_tv=float(lambda_tv),
        lambda_grad=float(lambda_grad),
        lambda_grad_img=float(lambda_grad_img),
        lambda_edge_rgb=float(lambda_edge_rgb),
        lambda_delta=float(lambda_delta),
        lambda_delta_rho=float(lambda_delta_rho),
        lambda_splat=float(lambda_splat),
        lambda_edge_splat=float(lambda_edge_splat),
        lambda_grid=float(lambda_grid),
    )
    loss_fn = UnisharpLoss(
        weights=loss_w,
        delta_clip=float(delta_clip),
        raw_delta_clip=float(raw_delta_clip),
        raw_delta_rho_clip=float(raw_delta_rho_clip),
        alpha_tail_min=float(alpha_tail_min),
        alpha_tail_weight=float(alpha_tail_weight),
        splat_sigma_min=float(splat_sigma_min),
        splat_sigma_max=float(splat_sigma_max),
        edge_splat_sigma_max=float(edge_splat_sigma_max),
        depth_edge_log_threshold=float(depth_edge_log_threshold),
        depth_edge_dilate_px=int(depth_edge_dilate_px),
    ).to(dev)
    loss_fn.SUPERVISION_MAX_DEPTH_M = float(max_depth_m)

    if is_main:
        config_dict = {
            "max_depth_m": float(max_depth_m),
            "sim_far_depth_invalid_m": float(sim_far_depth_invalid_m),
            "sim_far_depth_invalid_max_frac": float(sim_far_depth_invalid_max_frac),
            "re10k_pseudo_far_depth_invalid_m": float(re10k_pseudo_far_depth_invalid_m),
            "scanetpp_fisheye_far_depth_invalid_m": float(scanetpp_fisheye_far_depth_invalid_m),
            "render_low_pass_filter_eps": float(render_low_pass_filter_eps),
        }
        (out_dir / "config.json").write_text(
            json.dumps(config_dict, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    loss_csv = out_dir / "losses.csv"
    loss_csv_fields = [
        "loss",
        "src_loss",
        "tgt_loss",
        "dataset",
    ]
    if is_main:
        with loss_csv.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=loss_csv_fields).writeheader()

    if is_main:
        LOGGER.info("Training loop started.")

    from unisharp.cli.unified_trainer import UnifiedTrainer
    
    trainer = UnifiedTrainer(
        model=model,
        renderer=renderer,
        loss_fn=loss_fn,
        device=dev,
        max_depth_m=float(max_depth_m),
        sim_far_depth_invalid_m=float(sim_far_depth_invalid_m),
        re10k_pseudo_far_depth_invalid_m=float(re10k_pseudo_far_depth_invalid_m),
        scanetpp_fisheye_far_depth_invalid_m=float(scanetpp_fisheye_far_depth_invalid_m),
        aux_ray_loss_weight=float(lambda_aux_ray),
        aux_depth_scale_loss_weight=float(lambda_aux_depth_scale),
        aux_depth2_scale_loss_weight=float(lambda_aux_depth2_scale),
        target_mask_erode_px=int(target_mask_erode_px),
    )
    skip_forward_oom = _env_flag("TRAIN_SKIP_FORWARD_OOM", default=True)

    dataset_epochs: dict[str, int] = {name: 0 for name in dataloaders.keys()}
    dataset_samplers: dict[str, DistributedSampler | None] = {"hm3d": hm3d_sampler}

    for step in range(1, steps + 1):
        lr = warmup_cosine_lr(step, warmup, steps, lr0, lr1)
        lr_unik3d_encoder = warmup_cosine_lr(step, warmup, steps, unik3d_encoder_lr0, unik3d_encoder_lr1)
        lr_unik3d_decoder = warmup_cosine_lr(step, warmup, steps, unik3d_lr0, unik3d_lr1)
        for g in opt.param_groups:
            if g.get("group_name") == "unik3d_encoder":
                g["lr"] = lr_unik3d_encoder
            elif g.get("group_name") == "unik3d_decoder":
                g["lr"] = lr_unik3d_decoder
            else:
                g["lr"] = lr

        if _ddp_is_enabled():
            batch = None
            available_dataset_names = list(dataloaders.keys())
            dataset_name = ""
            for _dataset_attempt in range(max(1, len(dataloaders))):
                dataset_name = _ddp_broadcast_str(
                    sampler.choose_dataset_name(available_dataset_names) if is_main else "",
                    is_main=is_main,
                )

                local_exhausted = False
                try:
                    batch = sampler.next_batch(dataset_name)
                except StopIteration:
                    local_exhausted = True

                exhausted_any = _ddp_any_bool(local_exhausted, device=dev)
                if exhausted_any:
                    dataset_epochs[dataset_name] = dataset_epochs.get(dataset_name, 0) + 1
                    ds_sampler = dataset_samplers.get(dataset_name, None)
                    if ds_sampler is not None:
                        ds_sampler.set_epoch(dataset_epochs[dataset_name])
                    _maybe_set_dataset_epoch(datasets[dataset_name], dataset_epochs[dataset_name])
                    iterators[dataset_name] = iter(dataloaders[dataset_name])
                    sampler.iterators = iterators
                    batch = None

                    local_exhausted = False
                    try:
                        batch = sampler.next_batch(dataset_name)
                    except StopIteration:
                        local_exhausted = True
                    exhausted_any = _ddp_any_bool(local_exhausted, device=dev)

                if not exhausted_any:
                    break

                batch = None
                available_dataset_names = [name for name in available_dataset_names if name != dataset_name]
                if len(available_dataset_names) == 0:
                    break
            if batch is None:
                raise RuntimeError(f"Failed to fetch synchronized DDP batch for dataset={dataset_name}")
        else:
            try:
                dataset_name, batch = sampler.sample()
            except StopIteration as e:
                msg = str(e)
                exhausted_name = None
                if msg.startswith("Dataset ") and msg.endswith(" exhausted"):
                    exhausted_name = msg[len("Dataset ") : -len(" exhausted")]
                if exhausted_name is None or exhausted_name not in dataloaders:
                    raise
                dataset_epochs[exhausted_name] = dataset_epochs.get(exhausted_name, 0) + 1
                ds_sampler = dataset_samplers.get(exhausted_name, None)
                if ds_sampler is not None:
                    ds_sampler.set_epoch(dataset_epochs[exhausted_name])
                _maybe_set_dataset_epoch(datasets[exhausted_name], dataset_epochs[exhausted_name])
                iterators[exhausted_name] = iter(dataloaders[exhausted_name])
                sampler.iterators = iterators
                dataset_name, batch = sampler.sample()

        batch = _resize_training_batch_to_multiple(batch, int(train_resize_multiple))

        opt.zero_grad(set_to_none=True)

        autocast_enabled = dev.type == "cuda"
        if autocast_enabled and torch.cuda.is_bf16_supported():
            autocast_dtype = torch.bfloat16
        else:
            autocast_dtype = torch.float16 if autocast_enabled else torch.bfloat16

        need_vis = bool(is_main and vis_every > 0 and (step % vis_every == 0))
        result: dict[str, Any] | None = None
        forward_oom_local = False
        forward_oom_error = ""
        try:
            with torch.autocast(device_type=dev.type, enabled=autocast_enabled, dtype=autocast_dtype):
                result = trainer.process_batch(
                    batch,
                    dataset_name,
                    step,
                    need_vis=need_vis,
                )
        except Exception as e:
            if skip_forward_oom and _is_oom_exception(e):
                forward_oom_local = True
                forward_oom_error = str(e)
                opt.zero_grad(set_to_none=True)
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
            else:
                raise

        forward_oom_any = _ddp_any_bool(forward_oom_local, device=dev)
        if forward_oom_any:
            opt.zero_grad(set_to_none=True)
            if result is not None:
                del result
                result = None
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            if is_main:
                LOGGER.error(
                    "Skipping optimizer step=%d because forward OOM occurred on at least one rank | dataset=%s",
                    int(step),
                    str(dataset_name),
                )
            continue

        if result is None:
            raise RuntimeError(f"Forward returned no result for dataset={dataset_name} step={step}")
        total_loss = result["total"]
        local_nonfinite_loss = not bool(torch.isfinite(total_loss.detach()).item())
        nonfinite_loss_any = _ddp_any_bool(local_nonfinite_loss, device=dev)
        if nonfinite_loss_any:
            opt.zero_grad(set_to_none=True)
            if is_main:
                LOGGER.error(
                    "Skipping optimizer step=%d because loss is non-finite on at least one rank | dataset=%s",
                    int(step),
                    str(dataset_name),
                )
            continue

        try:
            scaler.scale(total_loss).backward()
        except Exception as e:
            raise
        try:
            scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=float(grad_clip_norm))
        except Exception as e:
            LOGGER.error("Gradient unscale/clip failed at step=%d: %s", int(step), str(e))
            raise
        grad_norm_value = float(grad_norm.detach().to(dtype=torch.float32).cpu().item()) if torch.is_tensor(grad_norm) else float(grad_norm)
        local_nonfinite_grad = not np.isfinite(grad_norm_value)
        nonfinite_grad_any = _ddp_any_bool(local_nonfinite_grad, device=dev)
        if nonfinite_grad_any:
            opt.zero_grad(set_to_none=True)
            scaler.update()
            if is_main:
                LOGGER.error(
                    "Skipping optimizer step=%d because grad norm is non-finite on at least one rank | dataset=%s | local_grad_norm=%s",
                    int(step),
                    str(dataset_name),
                    str(grad_norm_value),
                )
            continue
        local_huge_grad = bool(float(max_step_grad_norm) > 0.0 and grad_norm_value > float(max_step_grad_norm))
        huge_grad_any = _ddp_any_bool(local_huge_grad, device=dev)
        if huge_grad_any:
            opt.zero_grad(set_to_none=True)
            scaler.update()
            if is_main:
                LOGGER.error(
                    "Skipping optimizer step=%d because grad norm exceeded max-step-grad-norm on at least one rank | dataset=%s | local_grad_norm=%.6g | threshold=%.6g",
                    int(step),
                    str(dataset_name),
                    float(grad_norm_value),
                    float(max_step_grad_norm),
                )
            continue
        scaler.step(opt)
        scaler.update()

        if log_every > 0 and step % log_every == 0:
            loss_v = float(_ddp_mean(total_loss.detach()).item())
            src_v = float(_ddp_mean(result["src"].detach()).item())
            tgt_v = float(_ddp_mean(result["tgt"].detach()).item())
            row = {
                "loss": loss_v,
                "src_loss": src_v,
                "tgt_loss": tgt_v,
                "dataset": str(dataset_name),
            }
            if is_main:
                LOGGER.info(
                    "step=%d dataset=%s loss=%.6f src_loss=%.6f tgt_loss=%.6f",
                    step,
                    dataset_name,
                    loss_v,
                    src_v,
                    tgt_v,
                )
                row_csv = dict(row)
                for k in ("loss", "src_loss", "tgt_loss"):
                    v = float(row_csv.get(k, float("nan")))
                    row_csv[k] = "" if not np.isfinite(v) else f"{v:.4f}"
                with loss_csv.open("a", newline="") as f:
                    csv.DictWriter(f, fieldnames=loss_csv_fields).writerow(row_csv)

        if need_vis and result.get("vis_payload"):
            vis = result["vis_payload"]
            _save_train_vis(
                out_dir,
                step,
                vis["src_gt"],
                vis["src_pred"],
                vis["src_alpha"],
                vis["tgt_gt"],
                vis["tgt_pred"],
                vis["tgt_alpha"],
                src_gt_depth=vis.get("src_gt_depth"),
                tgt_gt_depth=vis.get("tgt_gt_depth"),
                src_pred_depth=vis.get("src_pred_depth"),
                tgt_pred_depth=vis.get("tgt_pred_depth"),
                src_unik3d_depth=vis.get("src_unik3d_depth"),
                tgt_unik3d_depth=vis.get("tgt_unik3d_depth"),
                dataset_name=vis.get("dataset_name"),
                scene=vis.get("scene"),
                src_idx=vis.get("src_idx"),
                tgt_idx=vis.get("tgt_idx"),
                src_pose_w2c=vis.get("src_pose_w2c"),
                tgt_pose_w2c=vis.get("tgt_pose_w2c"),
                src_metric_mask=vis.get("src_metric_mask"),
                tgt_metric_mask=vis.get("tgt_metric_mask"),
                src_cube_gt_u8=vis.get("src_cube_gt_u8"),
                src_cube_pred_linear=vis.get("src_cube_pred_linear"),
                src_cube_alpha=vis.get("src_cube_alpha"),
                tgt_cube_gt_u8=vis.get("tgt_cube_gt_u8"),
                tgt_cube_pred_linear=vis.get("tgt_cube_pred_linear"),
                tgt_cube_alpha=vis.get("tgt_cube_alpha"),
            )

        if need_vis:
            if "vis" in locals():
                del vis
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        del result
        del total_loss
        batch = None

        if is_main and (save_every > 0) and (step % save_every == 0):
            path = out_dir / f"step_{step:07d}.pt"
            raw_model.save_checkpoint(str(path), step, opt)
            LOGGER.info("💾 Saved checkpoint: %s", str(path))


    if _ddp_is_enabled():
        _ddp_barrier(dev)
        dist.destroy_process_group()

    if is_main:
        LOGGER.info("✅ Training completed!")

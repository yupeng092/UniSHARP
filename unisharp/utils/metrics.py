
from __future__ import annotations

import logging
import os
from pathlib import Path
import re
from typing import Any
from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F

LOGGER = logging.getLogger(__name__)
METRIC_MASK_CACHE_VERSION = "v3_source_bounds"


def _compute_psnr(gt: Tensor, pred: Tensor, eps: float = 1e-8) -> Tensor:
    mse = torch.mean((gt - pred) ** 2, dim=(1, 2, 3)).clamp_min(eps)
    return -10.0 * torch.log10(mse)


def _gaussian_kernel(size: int = 11, sigma: float = 1.5, device: Optional[torch.device] = None) -> Tensor:
    coords = torch.arange(size, device=device, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma * sigma))
    g = g / g.sum().clamp_min(1e-8)
    k2d = torch.outer(g, g)
    return k2d


def _compute_ssim_map(gt: Tensor, pred: Tensor) -> Tensor:
    b, c, _, _ = gt.shape
    kernel = _gaussian_kernel(device=gt.device).to(dtype=gt.dtype)
    kernel = kernel.view(1, 1, kernel.shape[0], kernel.shape[1]).repeat(c, 1, 1, 1)

    mu_x = F.conv2d(pred, kernel, padding=5, groups=c)
    mu_y = F.conv2d(gt, kernel, padding=5, groups=c)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(pred * pred, kernel, padding=5, groups=c) - mu_x2
    sigma_y2 = F.conv2d(gt * gt, kernel, padding=5, groups=c) - mu_y2
    sigma_xy = F.conv2d(pred * gt, kernel, padding=5, groups=c) - mu_xy

    c1 = (0.01**2)
    c2 = (0.03**2)
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + 1e-8
    )
    return ssim_map.view(b, c, gt.shape[-2], gt.shape[-1])


def _compute_ssim(gt: Tensor, pred: Tensor) -> Tensor:
    ssim_map = _compute_ssim_map(gt, pred)
    return ssim_map.view(int(ssim_map.shape[0]), -1).mean(dim=1)


class _LPIPSLike:

    def __init__(self, device: torch.device):
        self.device = device
        self.net = None
        try:
            import lpips  # type: ignore

            self.net = lpips.LPIPS(net="alex").to(device).eval()
            LOGGER.info("LPIPS backend: lpips/alex")
        except Exception:
            LOGGER.warning("LPIPS package not available, fallback to normalized L1 proxy.")

    @torch.no_grad()
    def __call__(self, gt: Tensor, pred: Tensor) -> Tensor:
        if self.net is not None:
            gt_n = gt * 2.0 - 1.0
            pred_n = pred * 2.0 - 1.0
            val = self.net(pred_n, gt_n, normalize=False)
            return val.view(val.shape[0])
        l1 = torch.mean(torch.abs(gt - pred), dim=(1, 2, 3))
        return l1.clamp_min(0.0)


class MetricsCalculator:

    def __init__(self, device: torch.device = None, compute_lpips: bool = True):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_lpips = bool(compute_lpips)
        self.lpips_calculator = _LPIPSLike(self.device) if self.compute_lpips else None

    @torch.no_grad()
    def compute_lpips_value(self, gt: Tensor, pred: Tensor) -> Tensor:
        if not self.compute_lpips or self.lpips_calculator is None:
            return torch.full((int(gt.shape[0]),), float("nan"), device=self.device, dtype=torch.float32)
        return self.lpips_calculator(gt, pred)

    @torch.no_grad()
    def compute_rgb_metrics(
        self,
        pred: Tensor,
        gt: Tensor,
    ) -> dict[str, float]:
        pred = pred.to(self.device)
        gt = gt.to(self.device)

        return {
            "psnr": _compute_psnr(gt, pred).mean().item(),
            "ssim": _compute_ssim(gt, pred).mean().item(),
            "lpips": self.compute_lpips_value(gt, pred).mean().item(),
        }


@torch.no_grad()
def compute_masked_rgb_metrics(
    pred: Tensor,
    gt: Tensor,
    mask: Tensor,
    metrics_calc: MetricsCalculator,
) -> dict[str, float]:
    m = (mask.to(dtype=torch.float32) > 0.5).to(dtype=torch.float32)
    valid_px = float(m.sum().item())
    if valid_px < 8.0:
        return {
            "psnr": float("nan"),
            "ssim": float("nan"),
            "lpips": float("nan"),
            "coverage": float(0.0),
        }

    denom = (m.sum() * float(pred.shape[1])).clamp(min=1.0)
    mse = (((pred - gt) ** 2) * m).sum() / denom
    psnr = float((-10.0 * torch.log10(mse.clamp_min(1e-8))).item())

    ssim_map = _compute_ssim_map(gt, pred).mean(dim=1, keepdim=True)
    ssim = float(((ssim_map * m).sum() / m.sum().clamp(min=1.0)).item())

    idx = torch.nonzero(m[0, 0] > 0.5, as_tuple=False)
    if idx.numel() == 0:
        lpips_val = float("nan")
    else:
        y0 = int(idx[:, 0].min().item())
        y1 = int(idx[:, 0].max().item()) + 1
        x0 = int(idx[:, 1].min().item())
        x1 = int(idx[:, 1].max().item()) + 1
        pred_c = pred[:, :, y0:y1, x0:x1]
        gt_c = gt[:, :, y0:y1, x0:x1]
        m_c = m[:, :, y0:y1, x0:x1]
        pred_blend = pred_c * m_c + gt_c * (1.0 - m_c)
        try:
            lpips_val = float(metrics_calc.compute_lpips_value(gt_c, pred_blend).mean().item())
        except Exception:
            l1 = torch.abs(pred_c - gt_c)
            lpips_val = float(((l1 * m_c).sum() / (m_c.sum() * 3.0).clamp(min=1.0)).item())

    return {
        "psnr": psnr,
        "ssim": ssim,
        "lpips": lpips_val,
        "coverage": float((m > 0.5).float().mean().item()),
    }


def default_metric_mask_cache_dir(repo_root: Path | None = None) -> Path:
    env = os.environ.get("VALIDATION_METRIC_MASK_CACHE_DIR", "").strip()
    if env:
        return Path(env)
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "validation_metric_masks"


def _safe_part(value: Any) -> str:
    text = str(value)
    text = text.replace(os.sep, "__")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)[:180]


def metric_mask_cache_path(
    cache_dir: Path,
    *,
    dataset: str,
    scene: Any,
    src_idx: Any,
    tgt_idx: Any,
    height: int,
    width: int,
) -> Path:
    name = (
        f"{METRIC_MASK_CACHE_VERSION}__{_safe_part(scene)}__s{_safe_part(src_idx)}"
        f"__t{_safe_part(tgt_idx)}__{int(height)}x{int(width)}.pt"
    )
    return Path(cache_dir) / str(dataset) / name


def _as_batched_depth(depth: Tensor | None) -> Tensor | None:
    if not torch.is_tensor(depth):
        return None
    if depth.ndim == 3:
        depth = depth.unsqueeze(1)
    if depth.ndim != 4:
        return None
    if depth.shape[1] != 1:
        depth = depth[:, :1]
    return depth.to(dtype=torch.float32)


def _as_batched_k(k: Tensor | None) -> Tensor | None:
    if not torch.is_tensor(k):
        return None
    if k.ndim == 2:
        k = k.unsqueeze(0)
    if k.ndim != 3 or tuple(k.shape[-2:]) != (3, 3):
        return None
    return k.to(dtype=torch.float32)


def _as_batched_pose(pose: Tensor | None) -> Tensor | None:
    if not torch.is_tensor(pose):
        return None
    if pose.ndim == 2:
        pose = pose.unsqueeze(0)
    if pose.ndim != 3 or tuple(pose.shape[-2:]) != (4, 4):
        return None
    return pose.to(dtype=torch.float32)


def _item_at(value: Any, index: int, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return value[index] if 0 <= int(index) < len(value) else default
    if torch.is_tensor(value):
        if value.numel() == 0:
            return default
        if value.ndim == 0:
            return value.item()
        if 0 <= int(index) < int(value.shape[0]):
            item = value[int(index)]
            return item.item() if item.numel() == 1 else item
        return default
    return value


def _batch_get(batch: Any, *names: str) -> Any:
    for name in names:
        if isinstance(batch, dict) and name in batch:
            value = batch[name]
            if value is not None:
                return value
        if hasattr(batch, name):
            value = getattr(batch, name)
            if value is not None:
                return value
    return None


@torch.no_grad()
def compute_source_frustum_mask(
    *,
    depth: Tensor,
    tgt_w2c: Tensor,
    src_w2c: Tensor,
    src_k3: Tensor,
    tgt_k3: Tensor,
    target_hw: tuple[int, int],
    source_hw: tuple[int, int] | None = None,
) -> Tensor:
    depth = _as_batched_depth(depth)
    tgt_w2c = _as_batched_pose(tgt_w2c)
    src_w2c = _as_batched_pose(src_w2c)
    src_k3 = _as_batched_k(src_k3)
    tgt_k3 = _as_batched_k(tgt_k3)
    if depth is None or tgt_w2c is None or src_w2c is None or src_k3 is None or tgt_k3 is None:
        raise ValueError("Invalid geometry inputs for source frustum mask.")

    device = depth.device
    dtype = torch.float32
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    source_h, source_w = (target_h, target_w) if source_hw is None else (int(source_hw[0]), int(source_hw[1]))
    if tuple(depth.shape[-2:]) != (target_h, target_w):
        depth = F.interpolate(depth, size=(target_h, target_w), mode="nearest")

    y_coords, x_coords = torch.meshgrid(
        torch.arange(target_h, device=device, dtype=dtype),
        torch.arange(target_w, device=device, dtype=dtype),
        indexing="ij",
    )

    z = depth[0, 0].to(dtype=dtype)
    valid_depth = torch.isfinite(z) & (z > 0.0)
    fx_t = tgt_k3[0, 0, 0].to(device=device, dtype=dtype).clamp(min=1e-6)
    fy_t = tgt_k3[0, 1, 1].to(device=device, dtype=dtype).clamp(min=1e-6)
    cx_t = tgt_k3[0, 0, 2].to(device=device, dtype=dtype)
    cy_t = tgt_k3[0, 1, 2].to(device=device, dtype=dtype)

    x_cam = (x_coords - cx_t) * z / fx_t
    y_cam = (y_coords - cy_t) * z / fy_t
    pts_tgt = torch.stack([x_cam, y_cam, z, torch.ones_like(z)], dim=0).reshape(4, -1)

    tgt_c2w = torch.linalg.inv(tgt_w2c[0].to(device=device, dtype=dtype))
    pts_world = tgt_c2w @ pts_tgt
    pts_src = src_w2c[0].to(device=device, dtype=dtype) @ pts_world

    fx_s = src_k3[0, 0, 0].to(device=device, dtype=dtype).clamp(min=1e-6)
    fy_s = src_k3[0, 1, 1].to(device=device, dtype=dtype).clamp(min=1e-6)
    cx_s = src_k3[0, 0, 2].to(device=device, dtype=dtype)
    cy_s = src_k3[0, 1, 2].to(device=device, dtype=dtype)
    z_src = pts_src[2]
    u_src = fx_s * (pts_src[0] / z_src.clamp(min=1e-6)) + cx_s
    v_src = fy_s * (pts_src[1] / z_src.clamp(min=1e-6)) + cy_s

    valid = (
        valid_depth.reshape(-1)
        & (z_src > 0.0)
        & (u_src >= 0.0)
        & (u_src < float(source_w))
        & (v_src >= 0.0)
        & (v_src < float(source_h))
    )
    return valid.reshape(target_h, target_w).to(dtype=torch.float32)[None, None]


def load_cached_metric_mask(
    cache_dir: Path | None,
    *,
    dataset: str,
    scene: Any,
    src_idx: Any,
    tgt_idx: Any,
    height: int,
    width: int,
    device: torch.device,
) -> Tensor | None:
    if cache_dir is None:
        return None
    path = metric_mask_cache_path(
        Path(cache_dir),
        dataset=dataset,
        scene=scene,
        src_idx=src_idx,
        tgt_idx=tgt_idx,
        height=height,
        width=width,
    )
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu")
    value = payload.get("mask", payload) if isinstance(payload, dict) else payload
    if not torch.is_tensor(value):
        return None
    if value.ndim == 2:
        value = value[None, None]
    elif value.ndim == 3:
        value = value.unsqueeze(1)
    if value.ndim != 4:
        return None
    if tuple(value.shape[-2:]) != (int(height), int(width)):
        value = F.interpolate(value.to(dtype=torch.float32), size=(int(height), int(width)), mode="nearest")
    return (value.to(device=device, dtype=torch.float32) > 0.5).to(dtype=torch.float32)


@torch.no_grad()
def metric_mask_from_pinhole_batch(
    batch: Any,
    *,
    dataset: str,
    cache_dir: Path | None = None,
    device: torch.device | None = None,
) -> Tensor | None:
    tgt_img = _batch_get(batch, "tgt_img", "tgt_rgb_u8")
    src_img = _batch_get(batch, "src_img", "src_rgb_u8")
    src_w2c = _as_batched_pose(_batch_get(batch, "src_w2c"))
    tgt_w2c = _as_batched_pose(_batch_get(batch, "tgt_w2c"))
    src_k = _as_batched_k(_batch_get(batch, "src_k", "src_intrinsics"))
    tgt_k = _as_batched_k(_batch_get(batch, "tgt_k", "tgt_intrinsics"))
    tgt_depth = _as_batched_depth(_batch_get(batch, "tgt_depth", "tgt_depth_m", "tgt_depth_m_orig"))
    if not torch.is_tensor(tgt_img) or src_w2c is None or tgt_w2c is None or src_k is None or tgt_k is None:
        return None

    if device is None:
        device = tgt_img.device if torch.is_tensor(tgt_img) else torch.device("cpu")
    target_h, target_w = int(tgt_img.shape[-2]), int(tgt_img.shape[-1])
    source_h, source_w = (target_h, target_w)
    if torch.is_tensor(src_img):
        source_h, source_w = int(src_img.shape[-2]), int(src_img.shape[-1])
    batch_size = int(tgt_img.shape[0])
    scene_value = _batch_get(batch, "scene")
    src_idx_value = _batch_get(batch, "src_idx")
    tgt_idx_value = _batch_get(batch, "tgt_idx")

    masks: list[Tensor] = []
    for b in range(batch_size):
        scene = _item_at(scene_value, b, "unknown")
        src_idx = _item_at(src_idx_value, b, -1)
        tgt_idx = _item_at(tgt_idx_value, b, -1)
        depth_b = tgt_depth[b : b + 1].to(device=device) if tgt_depth is not None and b < int(tgt_depth.shape[0]) else None
        if depth_b is not None:
            try:
                mask_b = compute_source_frustum_mask(
                    depth=depth_b,
                    tgt_w2c=tgt_w2c[min(b, int(tgt_w2c.shape[0]) - 1) : min(b, int(tgt_w2c.shape[0]) - 1) + 1].to(device),
                    src_w2c=src_w2c[min(b, int(src_w2c.shape[0]) - 1) : min(b, int(src_w2c.shape[0]) - 1) + 1].to(device),
                    src_k3=src_k[min(b, int(src_k.shape[0]) - 1) : min(b, int(src_k.shape[0]) - 1) + 1].to(device),
                    tgt_k3=tgt_k[min(b, int(tgt_k.shape[0]) - 1) : min(b, int(tgt_k.shape[0]) - 1) + 1].to(device),
                    target_hw=(target_h, target_w),
                    source_hw=(source_h, source_w),
                )
                masks.append(mask_b)
                continue
            except Exception:
                pass
        cached = load_cached_metric_mask(
            cache_dir,
            dataset=dataset,
            scene=scene,
            src_idx=src_idx,
            tgt_idx=tgt_idx,
            height=target_h,
            width=target_w,
            device=device,
        )
        if cached is None:
            return None
        masks.append(cached)
    if not masks:
        return None
    return torch.cat(masks, dim=0).to(device=device, dtype=torch.float32)


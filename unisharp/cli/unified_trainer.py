from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn

from unisharp.utils.gsplat import GSplatRenderer
from unisharp.losses import UnisharpLoss
from unisharp.utils.camera_utils import (
    transform_gaussians_to_world,
    to_k4,
    compute_frustum_mask,
)
from unisharp.utils.fisheye_geer import (
    compute_fisheye624_frustum_mask,
    render_gaussians_fisheye624,
)
from unisharp.utils.camera_projection import cubemap_face_cameras, build_extrinsics_w2c, view_frustum_mask_cubemap_union
from unisharp.utils.pano import Cube2Equirec, get_pinhole_intrinsics_4x4
from unisharp import DEFAULT_MAX_DEPTH_M
from unisharp.utils.pixel_convention import integer_pixel_center_grid


@dataclass
class _ModeStrategy:

    batch_size: int
    gaussians: Any
    make_world_gaussians: Callable[[int, Any], Any]
    make_sample: Callable[[int, Any, bool], dict[str, Any]]
    collect_all_vis: bool = False


class UnifiedTrainer:
    
    def __init__(
        self,
        model: nn.Module,
        renderer: GSplatRenderer,
        loss_fn: UnisharpLoss,
        device: torch.device,
        enable_tgt_unik3d_vis: bool = True,
        max_depth_m: float = DEFAULT_MAX_DEPTH_M,
        sim_far_depth_invalid_m: float = 30.0,
        re10k_pseudo_far_depth_invalid_m: float = 30.0,
        scanetpp_fisheye_far_depth_invalid_m: float = 30.0,
        aux_ray_loss_weight: float = 3.0,
        aux_depth_scale_loss_weight: float = 3.0,
        aux_depth2_scale_loss_weight: float = 1.0,
        target_mask_erode_px: int = 0,
    ):
        self.model = model
        self.renderer = renderer
        self.loss_fn = loss_fn
        self.device = device
        self.enable_tgt_unik3d_vis = bool(enable_tgt_unik3d_vis)
        self.max_depth_m = float(max_depth_m)
        self.sim_far_depth_invalid_m = float(sim_far_depth_invalid_m)
        self.re10k_pseudo_far_depth_invalid_m = float(re10k_pseudo_far_depth_invalid_m)
        self.scanetpp_fisheye_far_depth_invalid_m = float(scanetpp_fisheye_far_depth_invalid_m)
        self.aux_ray_loss_weight = float(aux_ray_loss_weight)
        self.aux_depth_scale_loss_weight = float(aux_depth_scale_loss_weight)
        self.aux_depth2_scale_loss_weight = float(aux_depth2_scale_loss_weight)
        self.target_mask_erode_px = max(int(target_mask_erode_px), 0)

    @staticmethod
    def _erode_supervision_mask(mask: torch.Tensor, radius_px: int, *, circular_h: bool = False) -> torch.Tensor:
        radius = max(int(radius_px), 0)
        if radius <= 0:
            return mask
        if not torch.is_tensor(mask):
            return mask
        m = mask.to(dtype=torch.float32).clamp(0.0, 1.0)
        if m.ndim == 3:
            m = m.unsqueeze(1)
        invalid = 1.0 - m
        kernel = 2 * radius + 1
        if bool(circular_h):
            invalid = F.pad(invalid, (radius, radius, 0, 0), mode="circular")
            invalid = F.pad(invalid, (0, 0, radius, radius), mode="constant", value=0.0)
            dilated_invalid = F.max_pool2d(invalid, kernel_size=kernel, stride=1)
        else:
            dilated_invalid = F.max_pool2d(invalid, kernel_size=kernel, stride=1, padding=radius)
        return (m * (1.0 - dilated_invalid)).to(device=mask.device, dtype=mask.dtype)

    def _aux_ray_losses(
        self,
        *,
        pred_rays: torch.Tensor | None,
        gt_rays: torch.Tensor | None,
        mask: torch.Tensor | None,
        pred_distance: torch.Tensor | None = None,
        pred_distance2: torch.Tensor | None = None,
        gt_distance: torch.Tensor | None = None,
        gt_distance2: torch.Tensor | None = None,
        depth_mask: torch.Tensor | None = None,
        depth_mask2: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        if torch.is_tensor(pred_rays) and torch.is_tensor(gt_rays) and self.aux_ray_loss_weight > 0.0:
            out["unik3d_ray"] = self.aux_ray_loss_weight * self._unik3d_polar_ray_loss(
                pred_rays,
                gt_rays,
                mask,
            )
        if torch.is_tensor(pred_distance) and torch.is_tensor(gt_distance):
            out["unik3d_depth_scale"] = self.aux_depth_scale_loss_weight * self._unik3d_scale_depth_loss(
                pred_distance,
                gt_distance,
                depth_mask if torch.is_tensor(depth_mask) else mask,
            )
        depth2_target = gt_distance2 if torch.is_tensor(gt_distance2) else gt_distance
        if torch.is_tensor(pred_distance2) and torch.is_tensor(depth2_target):
            depth2_mask = depth_mask2 if torch.is_tensor(depth_mask2) else depth_mask
            out["unik3d_depth2_scale"] = self.aux_depth2_scale_loss_weight * self._unik3d_scale_depth_loss(
                pred_distance2,
                depth2_target,
                depth2_mask if torch.is_tensor(depth2_mask) else mask,
            )
        return out

    DEPTH_SUPERVISION_MAX_M: float = DEFAULT_MAX_DEPTH_M

    def _distance_init_cap_for_dataset(self, dataset_name: str) -> float | None:
        name = str(dataset_name).lower()
        if name == "re10k" and self.re10k_pseudo_far_depth_invalid_m > 0.0:
            return self.re10k_pseudo_far_depth_invalid_m
        if name == "sim" and self.sim_far_depth_invalid_m > 0.0:
            return self.sim_far_depth_invalid_m
        if name in {"scanetpp_fisheye", "scannetpp_fisheye"} and self.scanetpp_fisheye_far_depth_invalid_m > 0.0:
            return self.scanetpp_fisheye_far_depth_invalid_m
        return None

    @staticmethod
    def _unik3d_polar_ray_loss(
        pred_rays: torch.Tensor | None,
        gt_rays: torch.Tensor | None,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if not torch.is_tensor(pred_rays) or not torch.is_tensor(gt_rays):
            device = pred_rays.device if torch.is_tensor(pred_rays) else torch.device("cpu")
            return torch.zeros((), device=device, dtype=torch.float32)
        pred = pred_rays.to(dtype=torch.float32)
        gt = gt_rays.to(device=pred.device, dtype=torch.float32)
        if pred.ndim == 3:
            pred = pred.unsqueeze(0)
        if gt.ndim == 3:
            gt = gt.unsqueeze(0)
        if tuple(pred.shape) != tuple(gt.shape):
            gt = F.interpolate(gt, size=pred.shape[-2:], mode="bilinear", align_corners=False)
            gt = gt / torch.norm(gt, dim=1, keepdim=True).clamp(min=1e-5)
        pred = pred / torch.norm(pred, dim=1, keepdim=True).clamp(min=1e-5)
        gt = gt / torch.norm(gt, dim=1, keepdim=True).clamp(min=1e-5)

        px, py, pz = pred.unbind(dim=1)
        gx, gy, gz = gt.unbind(dim=1)
        polar_pred = torch.acos(pz.clamp(min=-0.99999, max=0.99999))
        polar_gt = torch.acos(gz.clamp(min=-0.99999, max=0.99999))
        az_pred = torch.atan2(py, px.abs().clamp(min=1e-5) * (2.0 * (px > 0).to(px.dtype) - 1.0))
        az_gt = torch.atan2(gy, gx.abs().clamp(min=1e-5) * (2.0 * (gx > 0).to(gx.dtype) - 1.0))
        polar_error = (polar_pred - polar_gt).abs()
        az_delta = az_pred - az_gt
        az_error = torch.atan2(torch.sin(az_delta), torch.cos(az_delta)).abs()
        quantile_weight = torch.ones_like(polar_error)
        quantile_weight[(polar_gt > polar_pred) & (polar_gt > torch.pi / 2)] = 1.4
        quantile_weight[(polar_gt <= polar_pred) & (polar_gt > torch.pi / 2)] = 0.6

        if torch.is_tensor(mask):
            m = mask.to(device=pred.device, dtype=torch.float32)
            if m.ndim == 3:
                m = m.unsqueeze(1)
            if tuple(m.shape[-2:]) != tuple(pred.shape[-2:]):
                m = F.interpolate(m, size=pred.shape[-2:], mode="nearest")
            m = m[:, 0].clamp(0.0, 1.0)
        else:
            m = torch.ones_like(polar_error)
        denom = m.sum(dim=(-1, -2), keepdim=False).clamp(min=1.0)
        mean_polar = (polar_error * quantile_weight * m).sum(dim=(-1, -2)) / denom
        mean_azimuth = (az_error * m).sum(dim=(-1, -2)) / denom
        mean_error = (3.0 * mean_polar + mean_azimuth) / 4.0
        return torch.sqrt(mean_error + 1e-4).mean()

    @staticmethod
    def _unik3d_scale_depth_loss(
        pred_distance: torch.Tensor,
        gt_distance: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        pred = UnifiedTrainer._as_b1hw_depth(pred_distance).to(dtype=torch.float32)
        gt = UnifiedTrainer._as_b1hw_depth(gt_distance).to(device=pred.device, dtype=torch.float32)
        if tuple(gt.shape[-2:]) != tuple(pred.shape[-2:]):
            gt = F.interpolate(gt, size=pred.shape[-2:], mode="nearest")
        valid = torch.isfinite(pred) & torch.isfinite(gt) & (pred > 0.0) & (gt > 0.0)
        if torch.is_tensor(mask):
            m = mask.to(device=pred.device)
            if m.ndim == 3:
                m = m.unsqueeze(1)
            if tuple(m.shape[-2:]) != tuple(pred.shape[-2:]):
                m = F.interpolate(m.to(dtype=torch.float32), size=pred.shape[-2:], mode="nearest")
            valid = valid & (m[:, :1] > 0.5)
        err = (gt.clamp(min=1e-4).log() - pred.clamp(min=1e-4).log()).abs()
        err = torch.where(valid, err, torch.zeros_like(err))
        denom = valid.to(dtype=err.dtype).sum(dim=(-2, -1)).clamp(min=1.0)
        per_image = err.sum(dim=(-2, -1)) / denom
        return torch.sqrt(per_image.clamp(min=0.0)).mean()

    def _base_model(self) -> nn.Module:
        return self.model.module if hasattr(self.model, "module") else self.model
        
    def process_batch(
        self,
        batch: Any,
        dataset_name: str,
        step: int,
        need_vis: bool = False,
    ) -> dict[str, Any]:
        if hasattr(batch, "src_rgb_u8") and hasattr(batch, "src_intrinsics"):
            strategy = self._build_pinhole_strategy(
                batch,
                step,
                need_vis=need_vis,
                dataset_name=str(dataset_name),
            )
        elif hasattr(batch, "src_rgb_u8") and hasattr(batch, "src_camera_params"):
            strategy = self._build_fisheye_strategy(
                batch,
                step,
                need_vis=need_vis,
                dataset_name=str(dataset_name),
            )
        elif hasattr(batch, "src_erp_rgb_u8") and hasattr(batch, "src_cube_depth_m"):
            strategy = self._build_spherical_strategy(
                batch,
                step,
                need_vis=need_vis,
                dataset_name=str(dataset_name),
            )
        else:
            raise ValueError(f"Unknown batch schema for dataset={dataset_name}")
        return self._run_strategy_loop(
            strategy,
            need_vis=need_vis,
        )

    def _run_strategy_loop(
        self,
        strategy: _ModeStrategy,
        need_vis: bool = False,
    ) -> dict[str, Any]:
        total_loss = torch.zeros((), device=self.device)
        src_sum = torch.zeros((), device=self.device)
        tgt_sum = torch.zeros((), device=self.device)
        src_log_sum: dict[str, torch.Tensor] = {}
        tgt_log_sum: dict[str, torch.Tensor] = {}
        aux_log_sum: dict[str, torch.Tensor] = {}
        vis_payload = None
        vis_payloads: list[dict[str, Any]] = []

        def _accumulate_loss_terms(term_specs: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
            merged: dict[str, torch.Tensor] = {}
            for spec in term_specs:
                term_losses = self._compute_view_loss(**spec)
                for k, v in term_losses.items():
                    merged[k] = merged.get(k, torch.zeros((), device=self.device)) + v
            return merged

        collect_all_vis = bool(getattr(strategy, "collect_all_vis", False))
        for b in range(int(strategy.batch_size)):
            g = strategy.gaussians
            g_b = type(g)(
                mean_vectors=g.mean_vectors[b : b + 1],
                singular_values=g.singular_values[b : b + 1],
                quaternions=g.quaternions[b : b + 1],
                colors=g.colors[b : b + 1],
                opacities=g.opacities[b : b + 1],
            )
            g_world = strategy.make_world_gaussians(b, g_b)
            sample = strategy.make_sample(
                b,
                g_world,
                bool(need_vis and (collect_all_vis or b == 0)),
            )

            if isinstance(sample.get("src_loss_terms", None), list):
                src_losses = _accumulate_loss_terms(sample["src_loss_terms"])
            else:
                src_losses = self._compute_view_loss(
                    pred_rgb_linear=sample["src_pred_rgb_linear"],
                    pred_alpha=sample["src_pred_alpha"],
                    pred_depth_m=sample["src_pred_depth_m"],
                    pred_depth2_m=sample.get("src_pred_depth2_m", None),
                    gt_rgb_u8=sample["src_gt_rgb_u8"],
                    gt_depth_m=sample["src_gt_depth_m"],
                    mask=sample["src_mask"],
                    apply_color=bool(sample.get("src_apply_color", True)),
                    apply_alpha=bool(sample.get("src_apply_alpha", True)),
                    apply_depth=bool(sample.get("src_apply_depth", True)),
                    apply_percep=False,
                    apply_tv=True,
                    apply_grad=bool(sample.get("src_apply_grad", True)),
                    apply_grad_img=bool(sample.get("src_apply_grad_img", True)),
                    apply_splat=bool(sample.get("src_apply_splat", True)),
                    grad_img_circular_h=sample.get("src_grad_img_circular_h", None),
                    gaussian_scales=sample.get("gaussian_scales", None),
                    gaussian_quaternions=sample.get("gaussian_quaternions", None),
                    gaussian_angular_cell=sample.get("gaussian_angular_cell", None),
                    delta_xy=sample.get("delta_xy", None),
                    delta_rho=sample.get("delta_rho", None),
                    delta_grid=sample.get("delta_grid", None),
                    gaussian_mean_vectors=sample.get("gaussian_mean_vectors", None),
                    gaussian_base_mean_vectors=sample.get("gaussian_base_mean_vectors", None),
                    gaussian_opacities=sample.get("gaussian_opacities", None),
                    gauss_grid_shape=sample.get("gauss_grid_shape", None),
                    projected_scale_factor=sample.get("projected_scale_factor", None),
                    projection_model=sample.get("projection_model", None),
                    projection_intrinsics=sample.get("projection_intrinsics", None),
                    projection_camera_params=sample.get("projection_camera_params", None),
                    depth_mask=sample.get("src_depth_mask", None),
                )
            if isinstance(sample.get("src_extra_loss_terms", None), list):
                extra_src_losses = _accumulate_loss_terms(sample["src_extra_loss_terms"])
                for k, v in extra_src_losses.items():
                    src_losses[k] = src_losses.get(k, torch.zeros((), device=self.device)) + v
            if isinstance(sample.get("tgt_loss_terms", None), list):
                tgt_losses = _accumulate_loss_terms(sample["tgt_loss_terms"])
            else:
                tgt_losses = self._compute_view_loss(
                    pred_rgb_linear=sample["tgt_pred_rgb_linear"],
                    pred_alpha=sample["tgt_pred_alpha"],
                    pred_depth_m=sample["tgt_pred_depth_m"],
                    pred_depth2_m=sample.get("tgt_pred_depth2_m", None),
                    gt_rgb_u8=sample["tgt_gt_rgb_u8"],
                    gt_depth_m=sample["tgt_gt_depth_m"],
                    mask=sample["tgt_mask"],
                    apply_color=bool(sample.get("tgt_apply_color", True)),
                    apply_alpha=bool(sample.get("tgt_apply_alpha", True)),
                    apply_depth=bool(sample.get("tgt_apply_depth", True)),
                    apply_percep=bool(sample.get("tgt_apply_percep", False)),
                    apply_tv=False,
                    apply_grad=False,
                    apply_grad_img=bool(sample.get("tgt_apply_grad_img", True)),
                    apply_splat=bool(sample.get("tgt_apply_splat", False)),
                    grad_img_circular_h=sample.get("tgt_grad_img_circular_h", None),
                    gaussian_scales=None,
                    gaussian_quaternions=None,
                    delta_xy=None,
                    delta_rho=None,
                    gaussian_mean_vectors=None,
                    gaussian_base_mean_vectors=None,
                    gaussian_opacities=None,
                    gauss_grid_shape=None,
                    projected_scale_factor=sample.get("projected_scale_factor", None),
                    projection_model=sample.get("projection_model", None),
                    projection_intrinsics=sample.get("projection_intrinsics", None),
                    projection_camera_params=sample.get("projection_camera_params", None),
                    depth_mask=sample.get("tgt_depth_mask", None),
                )
            if isinstance(sample.get("tgt_extra_loss_terms", None), list):
                extra_tgt_losses = _accumulate_loss_terms(sample["tgt_extra_loss_terms"])
                for k, v in extra_tgt_losses.items():
                    tgt_losses[k] = tgt_losses.get(k, torch.zeros((), device=self.device)) + v

            aux_total = torch.zeros((), device=self.device)
            raw_aux = sample.get("aux_losses", None)
            if isinstance(raw_aux, dict):
                for k, v in raw_aux.items():
                    if torch.is_tensor(v):
                        vv = v.to(device=self.device)
                    else:
                        vv = torch.tensor(float(v), device=self.device, dtype=torch.float32)
                    aux_total = aux_total + vv
                    aux_log_sum[str(k)] = aux_log_sum.get(str(k), torch.zeros((), device=self.device)) + vv.detach()

            src_sum = src_sum + src_losses["total"]
            tgt_sum = tgt_sum + tgt_losses["total"]
            total_loss = total_loss + src_losses["total"] + tgt_losses["total"] + aux_total
            for k, v in src_losses.items():
                src_log_sum[k] = src_log_sum.get(k, torch.zeros((), device=self.device)) + v.detach()
            for k, v in tgt_losses.items():
                tgt_log_sum[k] = tgt_log_sum.get(k, torch.zeros((), device=self.device)) + v.detach()

            if need_vis and isinstance(sample.get("vis_payload", None), dict):
                vis_payloads.append(sample["vis_payload"])
                if b == 0:
                    vis_payload = sample["vis_payload"]

        bs = float(strategy.batch_size)
        total_loss = total_loss / bs
        src_sum = src_sum / bs
        tgt_sum = tgt_sum / bs
        loss_breakdown: dict[str, torch.Tensor] = {}
        for k, v in src_log_sum.items():
            loss_breakdown[f"src_{k}"] = v / bs
        for k, v in tgt_log_sum.items():
            loss_breakdown[f"tgt_{k}"] = v / bs
        for k, v in aux_log_sum.items():
            loss_breakdown[f"aux_{k}"] = v / bs
        batch_stats = {
            "batch_size": int(strategy.batch_size),
            "gaussian_count": int(strategy.gaussians.mean_vectors.shape[1]),
        }

        return {
            "total": total_loss,
            "src": src_sum,
            "tgt": tgt_sum,
            "loss_breakdown": loss_breakdown,
            "batch_stats": batch_stats,
            "vis_payload": vis_payload,
            "vis_payloads": vis_payloads,
        }

    @staticmethod
    def _first_item(x: Any, default: Any = None) -> Any:
        if x is None:
            return default
        if isinstance(x, (list, tuple)):
            return x[0] if len(x) > 0 else default
        if torch.is_tensor(x):
            if x.numel() == 0:
                return default
            return x.flatten()[0].item()
        return x

    @staticmethod
    def _item_at(x: Any, index: int, default: Any = None) -> Any:
        if x is None:
            return default
        if isinstance(x, (list, tuple)):
            return x[index] if 0 <= int(index) < len(x) else default
        if torch.is_tensor(x):
            if x.numel() == 0:
                return default
            if x.ndim == 0:
                return x.item()
            if 0 <= int(index) < int(x.shape[0]):
                item = x[int(index)]
                return item.item() if item.numel() == 1 else item
            return default
        return x

    @staticmethod
    def _finite_quantile(x: torch.Tensor, q: float, default: float = float("nan")) -> torch.Tensor:
        vals = x[torch.isfinite(x)]
        if int(vals.numel()) <= 0:
            return torch.tensor(float(default), device=x.device, dtype=torch.float32)
        vals = vals.to(torch.float32).flatten()
        if int(vals.numel()) > 262144:
            step = max(1, int(vals.numel()) // 262144)
            vals = vals[::step]
        return torch.quantile(vals, float(q))

    def _clamp_distance_for_supervision(
        self,
        depth_m: torch.Tensor | None,
        *,
        max_depth_m: float | None = None,
        clamp_max: bool = True,
    ) -> torch.Tensor | None:
        if not torch.is_tensor(depth_m):
            return None
        cap = float(self.max_depth_m if max_depth_m is None else max_depth_m)
        out = depth_m.to(dtype=torch.float32)
        valid = torch.isfinite(out) & (out > 0.0)
        if bool(clamp_max):
            sanitized = out.clamp(min=1e-4, max=cap)
        else:
            sanitized = out.clamp(min=1e-4)
        return torch.where(valid, sanitized, torch.zeros_like(out))

    @staticmethod
    def _rendered_depth_valid_for_inv_loss(
        depth_m: torch.Tensor,
        alpha: torch.Tensor,
        *,
        alpha_min: float | None = None,
        depth_min_m: float = 1e-3,
    ) -> torch.Tensor:
        depth = depth_m.detach()
        valid = torch.isfinite(depth) & (depth > float(depth_min_m))
        if alpha_min is not None:
            a = alpha.detach().to(device=depth.device)
            valid = valid & (a[:, :1] > float(alpha_min))
        return valid.to(dtype=depth.dtype)

    def _pinhole_z_to_supervision_distance(
        self,
        z_depth_b1hw: torch.Tensor | None,
        k3_b33: torch.Tensor | None,
        *,
        clamp_max: bool = True,
    ) -> torch.Tensor | None:
        if not torch.is_tensor(z_depth_b1hw) or not torch.is_tensor(k3_b33):
            return None
        dist = self._z_depth_to_distance_pinhole(z_depth_b1hw, k3_b33)
        return self._clamp_distance_for_supervision(dist, clamp_max=bool(clamp_max))

    @staticmethod
    def _sanitize_positive_depth(depth_m: torch.Tensor | None) -> torch.Tensor | None:
        if not torch.is_tensor(depth_m):
            return None
        out = depth_m.to(dtype=torch.float32)
        valid = torch.isfinite(out) & (out > 0.0)
        return torch.where(valid, out, torch.zeros_like(out))

    @staticmethod
    def _as_b1hw_depth(depth: torch.Tensor) -> torch.Tensor:
        if depth.ndim == 3:
            return depth.unsqueeze(1)
        if depth.ndim == 4 and depth.shape[1] == 1:
            return depth
        raise ValueError(f"Expected depth shape (B,H,W) or (B,1,H,W), got {tuple(depth.shape)}")

    @staticmethod
    def _as_bchw_rgb_u8(image: torch.Tensor) -> torch.Tensor:
        if image.ndim == 3 and image.shape[0] == 3:
            return image.unsqueeze(0)
        if image.ndim == 4 and image.shape[1] == 3:
            return image
        raise ValueError(f"Expected image shape (3,H,W) or (B,3,H,W), got {tuple(image.shape)}")

    @staticmethod
    def _as_b33_intrinsics(intrinsics: torch.Tensor) -> torch.Tensor:
        if intrinsics.ndim == 2 and tuple(intrinsics.shape) == (3, 3):
            return intrinsics.unsqueeze(0)
        if intrinsics.ndim == 3 and tuple(intrinsics.shape[1:]) == (3, 3):
            return intrinsics
        raise ValueError(
            f"Expected intrinsics shape (3,3) or (B,3,3), got {tuple(intrinsics.shape)}"
        )

    @staticmethod
    def _as_b9_camera_params(camera_params: torch.Tensor) -> torch.Tensor:
        if camera_params.ndim == 1 and int(camera_params.shape[0]) == 9:
            return camera_params.unsqueeze(0)
        if camera_params.ndim == 2 and int(camera_params.shape[1]) == 9:
            return camera_params
        raise ValueError(f"Expected camera_params shape (9,) or (B,9), got {tuple(camera_params.shape)}")

    @staticmethod
    def _as_b16_camera_params(camera_params: torch.Tensor) -> torch.Tensor:
        if camera_params.ndim == 1 and int(camera_params.shape[0]) == 16:
            return camera_params.unsqueeze(0)
        if camera_params.ndim == 2 and int(camera_params.shape[1]) == 16:
            return camera_params
        raise ValueError(f"Expected camera_params shape (16,) or (B,16), got {tuple(camera_params.shape)}")

    @staticmethod
    def _as_b44_pose(extrinsics: torch.Tensor) -> torch.Tensor:
        if extrinsics.ndim == 2 and tuple(extrinsics.shape) == (4, 4):
            return extrinsics.unsqueeze(0)
        if extrinsics.ndim == 3 and tuple(extrinsics.shape[1:]) == (4, 4):
            return extrinsics
        raise ValueError(
            f"Expected extrinsics shape (4,4) or (B,4,4), got {tuple(extrinsics.shape)}"
        )

    @staticmethod
    def _pick_depth_for_pinhole_frustum_mask(
        gt_depth: torch.Tensor | None,
        pred_depth: torch.Tensor,
        min_valid_px: int = 8,
    ) -> torch.Tensor:
        if torch.is_tensor(gt_depth):
            gt_depth = UnifiedTrainer._as_b1hw_depth(gt_depth)
            valid = torch.isfinite(gt_depth) & (gt_depth > 0.0)
            if int(valid.sum().item()) >= int(min_valid_px):
                return gt_depth
        return pred_depth

    @staticmethod
    def _pick_depth_for_fisheye_frustum_mask(
        gt_depth: torch.Tensor | None,
        pred_depth: torch.Tensor,
        gt_valid_mask: torch.Tensor | None = None,
        min_valid_px: int = 8,
    ) -> torch.Tensor:
        if torch.is_tensor(gt_depth):
            gt_depth = UnifiedTrainer._as_b1hw_depth(gt_depth)
            if torch.is_tensor(gt_valid_mask):
                gt_valid = gt_depth > 0.0
                gt_valid = gt_valid & (gt_valid_mask > 0.5)
            else:
                gt_valid = torch.isfinite(gt_depth) & (gt_depth > 0.0)
            if int(gt_valid.sum().item()) >= int(min_valid_px):
                return gt_depth
        return pred_depth

    @staticmethod
    def _as_cubemap_depth_hw1(depth: torch.Tensor) -> torch.Tensor:
        if depth.ndim != 4:
            raise ValueError(f"Expected 4D cubemap depth, got shape={tuple(depth.shape)}")
        if depth.shape[-1] == 1:
            return depth
        if depth.shape[1] == 1:
            return depth.permute(0, 2, 3, 1).contiguous()
        raise ValueError(f"Unsupported cubemap depth shape={tuple(depth.shape)}")

    def _pick_depth_for_cubemap_frustum_mask(
        self,
        gt_depth_cube: torch.Tensor | None,
        pred_depth_cube: torch.Tensor,
        face_w: int,
        min_valid_px: int = 8,
    ) -> torch.Tensor:
        pred_hw1 = self._as_cubemap_depth_hw1(pred_depth_cube)
        if torch.is_tensor(gt_depth_cube):
            gt_hw1 = self._as_cubemap_depth_hw1(gt_depth_cube)
            gt_dist = self._cubemap_z_depth_to_distance(gt_hw1)
            gt_hw1 = self._as_cubemap_depth_hw1(gt_dist)
            if gt_hw1.shape[1] != int(face_w) or gt_hw1.shape[2] != int(face_w):
                gt_hw1 = F.interpolate(
                    gt_hw1.permute(0, 3, 1, 2),
                    size=(int(face_w), int(face_w)),
                    mode="nearest",
                ).permute(0, 2, 3, 1).contiguous()
            valid = torch.isfinite(gt_hw1[..., 0]) & (gt_hw1[..., 0] > 0.0)
            if int(valid.sum().item()) >= int(min_valid_px):
                return gt_hw1
        return pred_hw1

    @staticmethod
    def _distance_to_z_depth_pinhole(
        distance_b1hw: torch.Tensor,
        intrinsics_b33: torch.Tensor,
    ) -> torch.Tensor:
        distance_b1hw = UnifiedTrainer._as_b1hw_depth(distance_b1hw)
        intrinsics_b33 = UnifiedTrainer._as_b33_intrinsics(intrinsics_b33)
        b, _, h, w = distance_b1hw.shape
        dev = distance_b1hw.device
        dtype = distance_b1hw.dtype
        uu, vv = integer_pixel_center_grid(h, w, device=dev, dtype=dtype)
        uu = uu.unsqueeze(0).expand(b, -1, -1)
        vv = vv.unsqueeze(0).expand(b, -1, -1)
        fx = intrinsics_b33[:, 0, 0].view(b, 1, 1).to(dtype=dtype, device=dev)
        fy = intrinsics_b33[:, 1, 1].view(b, 1, 1).to(dtype=dtype, device=dev)
        cx = intrinsics_b33[:, 0, 2].view(b, 1, 1).to(dtype=dtype, device=dev)
        cy = intrinsics_b33[:, 1, 2].view(b, 1, 1).to(dtype=dtype, device=dev)
        x = (uu - cx) / fx
        y = (vv - cy) / fy
        ray_z = 1.0 / torch.sqrt(x * x + y * y + 1.0).clamp(min=1e-8)
        return distance_b1hw * ray_z.unsqueeze(1)

    @staticmethod
    def _z_depth_to_distance_pinhole(
        z_depth_b1hw: torch.Tensor,
        intrinsics_b33: torch.Tensor,
    ) -> torch.Tensor:
        z_depth_b1hw = UnifiedTrainer._as_b1hw_depth(z_depth_b1hw)
        intrinsics_b33 = UnifiedTrainer._as_b33_intrinsics(intrinsics_b33)
        b, _, h, w = z_depth_b1hw.shape
        dev = z_depth_b1hw.device
        dtype = z_depth_b1hw.dtype
        uu, vv = integer_pixel_center_grid(h, w, device=dev, dtype=dtype)
        uu = uu.unsqueeze(0).expand(b, -1, -1)
        vv = vv.unsqueeze(0).expand(b, -1, -1)
        fx = intrinsics_b33[:, 0, 0].view(b, 1, 1).to(dtype=dtype, device=dev)
        fy = intrinsics_b33[:, 1, 1].view(b, 1, 1).to(dtype=dtype, device=dev)
        cx = intrinsics_b33[:, 0, 2].view(b, 1, 1).to(dtype=dtype, device=dev)
        cy = intrinsics_b33[:, 1, 2].view(b, 1, 1).to(dtype=dtype, device=dev)
        x = (uu - cx) / fx
        y = (vv - cy) / fy
        ray_z = 1.0 / torch.sqrt(x * x + y * y + 1.0).clamp(min=1e-8)
        return z_depth_b1hw / ray_z.unsqueeze(1).clamp(min=1e-8)

    def _cubemap_z_depth_to_distance(
        self,
        depth_cube: torch.Tensor,
    ) -> torch.Tensor:
        if depth_cube.ndim != 4:
            raise ValueError(f"Expected 4D cubemap depth, got {tuple(depth_cube.shape)}")
        if depth_cube.shape[-1] == 1:
            depth_61hw = depth_cube.permute(0, 3, 1, 2).contiguous()
        elif depth_cube.shape[1] == 1:
            depth_61hw = depth_cube
        else:
            raise ValueError(f"Unsupported cubemap depth shape={tuple(depth_cube.shape)}")

        _, _, h, w = depth_61hw.shape
        intr = get_pinhole_intrinsics_4x4(int(w)).to(
            device=depth_61hw.device,
            dtype=depth_61hw.dtype,
        )
        fx = intr[0, 0]
        fy = intr[1, 1]
        cx = intr[0, 2]
        cy = intr[1, 2]
        uu, vv = integer_pixel_center_grid(h, w, device=depth_61hw.device, dtype=depth_61hw.dtype)
        x = (uu - cx) / fx
        y = (vv - cy) / fy
        ray_z = 1.0 / torch.sqrt(x * x + y * y + 1.0).clamp(min=1e-8)
        dist = depth_61hw / ray_z.view(1, 1, h, w).clamp(min=1e-8)
        valid = torch.isfinite(dist) & (depth_61hw > 0.0)
        dist = torch.where(valid, dist.clamp(min=1e-4), torch.zeros_like(dist))
        return dist

    def _collect_regularization_inputs(
        self,
        out: dict[str, Any],
        gaussians: Any,
        b: int,
        projected_scale_factor: float | None,
    ) -> dict[str, Any]:
        delta_b = out.get("delta", None)
        delta_xy_raw = None
        if torch.is_tensor(delta_b):
            delta_xy_raw = delta_b[b : b + 1, 0:2]
            delta_rho_raw = delta_b[b : b + 1, 2:3]
            delta_grid_raw = delta_b[b : b + 1]
        else:
            delta_rho_raw = None
            delta_grid_raw = None
        delta_rho_applied_all = out.get("delta_rho_applied", None)
        delta_rho_applied = (
            delta_rho_applied_all[b : b + 1]
            if torch.is_tensor(delta_rho_applied_all)
            else None
        )
        scale_factor_applied_all = out.get("scale_factor_applied", None)
        scale_factor_applied = (
            scale_factor_applied_all[b : b + 1]
            if torch.is_tensor(scale_factor_applied_all)
            else None
        )

        scales_b = gaussians.singular_values[b : b + 1]
        means_b = gaussians.mean_vectors[b : b + 1]
        quats_b = gaussians.quaternions[b : b + 1]
        opac_b = gaussians.opacities[b : b + 1]

        base_values = out.get("gaussian_base_values", None)
        gauss_grid_shape = None
        base_means_b = None
        base_scales_b = None
        angular_cell_b = None
        if base_values is not None and hasattr(base_values, "rays"):
            _, _, l, hb, wb = base_values.rays.shape
            gauss_grid_shape = (int(l), int(hb), int(wb))
            inv_dist_b = base_values.inv_distance[b : b + 1].clamp(min=1e-6)
            base_rays_b = F.normalize(base_values.rays[b : b + 1], dim=1, eps=1e-6)
            base_means_grid = base_rays_b / inv_dist_b
            base_scales_b = base_values.scales[b : b + 1]
            init_output = out.get("initializer_output", None)
            global_scale = (
                init_output.global_scale[b : b + 1]
                if init_output is not None
                and getattr(init_output, "global_scale", None) is not None
                else None
            )
            if torch.is_tensor(global_scale):
                base_means_grid = base_means_grid * global_scale.view(-1, 1, 1, 1, 1)
                base_scales_b = base_scales_b * global_scale.view(-1, 1, 1, 1, 1)
            base_means_b = base_means_grid.permute(0, 2, 3, 4, 1).flatten(1, 3)
            angular_cell = getattr(base_values, "angular_cell", None)
            angular_cell_b = angular_cell[b : b + 1] if torch.is_tensor(angular_cell) else None

        return {
            "delta_xy_eff": delta_xy_raw,
            "delta_rho_raw": delta_rho_raw,
            "delta_grid": delta_grid_raw,
            "delta_rho_applied": delta_rho_applied,
            "scale_factor_applied": scale_factor_applied,
            "gaussian_scales": scales_b,
            "gaussian_quaternions": quats_b,
            "gaussian_angular_cell": angular_cell_b,
            "gaussian_mean_vectors": means_b,
            "gaussian_base_mean_vectors": base_means_b,
            "gaussian_base_scales": base_scales_b,
            "gaussian_opacities": opac_b,
            "gauss_grid_shape": gauss_grid_shape,
            "projected_scale_factor": projected_scale_factor,
        }

    def _compute_view_loss(
        self,
        *,
        pred_rgb_linear: torch.Tensor,
        pred_alpha: torch.Tensor,
        pred_depth_m: torch.Tensor,
        pred_depth2_m: torch.Tensor | None,
        gt_rgb_u8: torch.Tensor,
        gt_depth_m: torch.Tensor,
        mask: torch.Tensor,
        apply_color: bool,
        apply_alpha: bool,
        apply_depth: bool,
        apply_percep: bool,
        apply_tv: bool,
        apply_grad: bool,
        apply_grad_img: bool,
        grad_img_circular_h: bool | None = None,
        gaussian_scales: torch.Tensor | None = None,
        gaussian_quaternions: torch.Tensor | None = None,
        gaussian_angular_cell: torch.Tensor | None = None,
        delta_xy: torch.Tensor | None = None,
        delta_rho: torch.Tensor | None = None,
        delta_grid: torch.Tensor | None = None,
        gaussian_mean_vectors: torch.Tensor | None = None,
        gaussian_base_mean_vectors: torch.Tensor | None = None,
        gaussian_opacities: torch.Tensor | None = None,
        gauss_grid_shape: tuple[int, int, int] | None = None,
        projected_scale_factor: float | torch.Tensor | None = None,
        projection_model: str | None = None,
        projection_intrinsics: torch.Tensor | None = None,
        projection_camera_params: torch.Tensor | None = None,
        loss_scale: float = 1.0,
        apply_splat: bool | None = None,
        depth_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        losses = self.loss_fn(
            pred_rgb_linear=pred_rgb_linear,
            pred_alpha=pred_alpha,
            pred_depth_m=pred_depth_m,
            pred_depth2_m=pred_depth2_m,
            gt_rgb_u8=gt_rgb_u8,
            gt_depth_m=gt_depth_m,
            mask=mask,
            depth_mask=depth_mask,
            gaussian_scales=gaussian_scales,
            gaussian_quaternions=gaussian_quaternions,
            gaussian_angular_cell=gaussian_angular_cell,
            delta_xy=delta_xy,
            delta_rho=delta_rho,
            delta_grid=delta_grid,
            apply_color=bool(apply_color),
            apply_alpha=bool(apply_alpha),
            apply_depth=bool(apply_depth),
            apply_percep=bool(apply_percep),
            apply_tv=bool(apply_tv),
            apply_grad=bool(apply_grad),
            apply_grad_img=bool(apply_grad_img),
            grad_img_circular_h=grad_img_circular_h,
            apply_delta=bool(torch.is_tensor(delta_xy) or torch.is_tensor(delta_rho)),
            apply_splat=bool(torch.is_tensor(gaussian_scales)) if apply_splat is None else bool(apply_splat),
            gaussian_mean_vectors=gaussian_mean_vectors,
            gaussian_base_mean_vectors=gaussian_base_mean_vectors,
            gaussian_opacities=gaussian_opacities,
            gauss_grid_shape=gauss_grid_shape,
            projected_scale_factor=projected_scale_factor,
            projection_model=projection_model,
            projection_intrinsics=projection_intrinsics,
            projection_camera_params=projection_camera_params,
        )
        scale = float(loss_scale)
        if abs(scale - 1.0) > 1e-8:
            losses = {k: (v * scale) for k, v in losses.items()}
        return losses
    
    def _build_pinhole_strategy(
        self,
        batch: Any,
        step: int,
        need_vis: bool = False,
        dataset_name: str = "re10k",
    ) -> _ModeStrategy:
        src_u8 = self._as_bchw_rgb_u8(batch.src_rgb_u8.to(self.device, non_blocking=True))
        tgt_u8 = self._as_bchw_rgb_u8(batch.tgt_rgb_u8.to(self.device, non_blocking=True))
        src_u8_orig = getattr(batch, "src_rgb_u8_orig", None)
        tgt_u8_orig = getattr(batch, "tgt_rgb_u8_orig", None)
        src_depth_gt = getattr(batch, "src_depth_m", None)
        tgt_depth_gt = getattr(batch, "tgt_depth_m", None)
        src_depth_gt_orig = getattr(batch, "src_depth_m_orig", None)
        tgt_depth_gt_orig = getattr(batch, "tgt_depth_m_orig", None)
        has_depth_gt = torch.is_tensor(src_depth_gt) and torch.is_tensor(tgt_depth_gt)
        if has_depth_gt:
            src_depth_gt = self._as_b1hw_depth(
                src_depth_gt.to(self.device, non_blocking=True).to(torch.float32)
            )
            tgt_depth_gt = self._as_b1hw_depth(
                tgt_depth_gt.to(self.device, non_blocking=True).to(torch.float32)
            )
        has_depth_gt_orig = torch.is_tensor(src_depth_gt_orig) and torch.is_tensor(tgt_depth_gt_orig)
        if has_depth_gt_orig:
            src_depth_gt_orig = self._as_b1hw_depth(
                src_depth_gt_orig.to(self.device, non_blocking=True).to(torch.float32)
            )
            tgt_depth_gt_orig = self._as_b1hw_depth(
                tgt_depth_gt_orig.to(self.device, non_blocking=True).to(torch.float32)
            )
        src_w2c = self._as_b44_pose(batch.src_w2c.to(self.device, non_blocking=True).to(torch.float32))
        tgt_w2c = self._as_b44_pose(batch.tgt_w2c.to(self.device, non_blocking=True).to(torch.float32))
        src_k3 = self._as_b33_intrinsics(batch.src_intrinsics.to(self.device, non_blocking=True).to(torch.float32))
        tgt_k3 = self._as_b33_intrinsics(batch.tgt_intrinsics.to(self.device, non_blocking=True).to(torch.float32))
        src_k3_orig = getattr(batch, "src_intrinsics_orig", None)
        tgt_k3_orig = getattr(batch, "tgt_intrinsics_orig", None)
        has_orig_vis = (
            torch.is_tensor(src_u8_orig)
            and torch.is_tensor(tgt_u8_orig)
            and torch.is_tensor(src_k3_orig)
            and torch.is_tensor(tgt_k3_orig)
        )
        if has_orig_vis:
            src_u8_orig = self._as_bchw_rgb_u8(src_u8_orig.to(self.device, non_blocking=True))
            tgt_u8_orig = self._as_bchw_rgb_u8(tgt_u8_orig.to(self.device, non_blocking=True))
            src_k3_orig = self._as_b33_intrinsics(
                src_k3_orig.to(self.device, non_blocking=True).to(torch.float32)
            )
            tgt_k3_orig = self._as_b33_intrinsics(
                tgt_k3_orig.to(self.device, non_blocking=True).to(torch.float32)
            )
        src_depth_gt_dist = None
        tgt_depth_gt_dist = None
        src_unik3d_gt_dist = None
        if has_depth_gt:
            src_unik3d_gt_dist = self._pinhole_z_to_supervision_distance(src_depth_gt, src_k3)
            src_depth_gt_dist = src_unik3d_gt_dist
            tgt_depth_gt_dist = self._pinhole_z_to_supervision_distance(tgt_depth_gt, tgt_k3)
        
        src = src_u8.float().clamp(0, 255) / 255.0
        tgt = tgt_u8.float().clamp(0, 255) / 255.0
        distance_init_cap_m = self._distance_init_cap_for_dataset(dataset_name)
        
        share_src_forward = bool(getattr(batch, "share_src_forward", False)) and int(src.shape[0]) > 1

        def _repeat_first_dim(value: Any, batch_size: int) -> Any:
            if torch.is_tensor(value):
                if value.ndim > 0 and int(value.shape[0]) == 1:
                    return value.repeat(batch_size, *([1] * (value.ndim - 1)))
                return value
            if hasattr(value, "_fields"):
                return type(value)(*[_repeat_first_dim(getattr(value, field), batch_size) for field in value._fields])
            return value

        if share_src_forward:
            out_single = self.model(
                image=src[0:1],
                image_u8=src_u8[0:1],
                camera_intrinsics=src_k3[0:1],
                camera_model="pinhole",
                depth_gt=(src_depth_gt_dist[0:1] if torch.is_tensor(src_depth_gt_dist) else None),
                distance_init_cap_m=distance_init_cap_m,
                return_aux=True,
            )
            out = {k: _repeat_first_dim(v, int(src.shape[0])) for k, v in out_single.items()}
        else:
            out = self.model(
                image=src,
                image_u8=src_u8,
                camera_intrinsics=src_k3,
                camera_model="pinhole",
                depth_gt=src_depth_gt_dist,
                distance_init_cap_m=distance_init_cap_m,
                return_aux=True,
            )
        gaussians = out["gaussians"]
        src_render_k3 = src_k3
        tgt_render_k3 = tgt_k3
        src_depth_gt_z_render = src_depth_gt if has_depth_gt else None
        tgt_depth_gt_z_render = tgt_depth_gt if has_depth_gt else None
        src_depth_gt_render_valid = (torch.isfinite(src_depth_gt) & (src_depth_gt > 0.0)) if has_depth_gt else None
        tgt_depth_gt_render_valid = (torch.isfinite(tgt_depth_gt) & (tgt_depth_gt > 0.0)) if has_depth_gt else None
        aux_ray_target_all = out.get("unik3d_gt_rays", None)
        def make_world_gaussians(b: int, g_b: Any) -> Any:
            return g_b

        def make_sample(b: int, g_world: Any, enable_vis: bool) -> dict[str, Any]:
            src_h = int(src_u8.shape[-2])
            src_w = int(src_u8.shape[-1])
            tgt_h = int(tgt_u8.shape[-2])
            tgt_w = int(tgt_u8.shape[-1])
            ident = torch.eye(4, dtype=src_w2c.dtype, device=self.device).unsqueeze(0)
            rel_tgt_w2c = tgt_w2c[b : b + 1] @ torch.linalg.inv(src_w2c[b : b + 1])
            src_k_render_b = src_render_k3[b : b + 1]
            tgt_k_render_b = tgt_render_k3[b : b + 1]
            src_out = self.renderer(
                g_world,
                extrinsics=ident,
                intrinsics=to_k4(src_k_render_b),
                image_width=src_w,
                image_height=src_h,
            )
            tgt_out = self.renderer(
                g_world,
                extrinsics=rel_tgt_w2c,
                intrinsics=to_k4(tgt_k_render_b),
                image_width=tgt_w,
                image_height=tgt_h,
            )

            zeros_src_depth = torch.zeros((1, 1, src_h, src_w), dtype=torch.float32, device=self.device)
            zeros_tgt_depth = torch.zeros((1, 1, tgt_h, tgt_w), dtype=torch.float32, device=self.device)
            ones_mask = torch.ones_like(zeros_src_depth)
            fx_b = float(src_k_render_b[0, 0, 0].item())
            fy_b = float(src_k_render_b[0, 1, 1].item())
            proj_scale_pinhole = 0.5 * (fx_b + fy_b)
            reg_inputs = self._collect_regularization_inputs(
                out=out,
                gaussians=gaussians,
                b=b,
                projected_scale_factor=proj_scale_pinhole,
            )

            src_depth_for_visibility = None
            tgt_gt_depth_for_mask = (
                tgt_depth_gt_z_render[b : b + 1]
                if has_depth_gt and torch.is_tensor(tgt_depth_gt_z_render)
                else None
            )
            if has_depth_gt:
                src_depth_for_visibility = (
                    src_depth_gt_z_render[b : b + 1]
                    if torch.is_tensor(src_depth_gt_z_render)
                    else src_depth_gt[b : b + 1]
                )

            tgt_depth_for_mask = self._pick_depth_for_pinhole_frustum_mask(
                gt_depth=tgt_gt_depth_for_mask,
                pred_depth=tgt_out.depth,
            )
            tgt_frustum_mask = compute_frustum_mask(
                depth=tgt_depth_for_mask,
                tgt_w2c=tgt_w2c[b : b + 1],
                src_w2c=src_w2c[b : b + 1],
                src_k3=src_k_render_b,
                tgt_k3=tgt_k_render_b,
                img_h=tgt_h,
                img_w=tgt_w,
                source_img_h=src_h,
                source_img_w=src_w,
                source_depth=src_depth_for_visibility,
            )
            tgt_frustum_mask_raw = tgt_frustum_mask
            tgt_frustum_mask = self._erode_supervision_mask(
                tgt_frustum_mask,
                self.target_mask_erode_px,
                circular_h=False,
            )
            src_depth_pred = self._clamp_distance_for_supervision(
                out["distance_layers"][b : b + 1, 0:1],
                clamp_max=False,
            )
            src_depth2_pred = (
                self._clamp_distance_for_supervision(out["distance_layers"][b : b + 1, 1:2], clamp_max=False)
                if out["distance_layers"] is not None and out["distance_layers"].shape[1] > 1
                else None
            )
            src_depth2_gt_for_aux = (
                src_unik3d_gt_dist[b : b + 1]
                if torch.is_tensor(src_unik3d_gt_dist)
                else None
            )
            src_depth2_mask_for_aux = src_depth_gt[b : b + 1] > 0.0 if has_depth_gt else None
            tgt_depth_pred = self._pinhole_z_to_supervision_distance(
                tgt_out.depth,
                tgt_k_render_b,
                clamp_max=False,
            )
            tgt_depth_loss_mask = self._rendered_depth_valid_for_inv_loss(tgt_depth_pred, tgt_out.alpha)
            if torch.is_tensor(tgt_depth_gt_render_valid):
                tgt_depth_loss_mask = tgt_depth_loss_mask * tgt_depth_gt_render_valid[b : b + 1].to(
                    device=tgt_depth_loss_mask.device,
                    dtype=tgt_depth_loss_mask.dtype,
                )
            tgt_extra_loss_terms: list[dict[str, Any]] = []
            vis_payload = None
            if enable_vis:
                vis_src_u8 = src_u8[b : b + 1]
                vis_tgt_u8 = tgt_u8[b : b + 1]
                vis_src_depth_gt = (src_depth_gt[b : b + 1] if has_depth_gt else None)
                vis_tgt_depth_gt = (tgt_depth_gt[b : b + 1] if has_depth_gt else None)
                vis_src_out = src_out
                vis_tgt_out = tgt_out
                if has_orig_vis:
                    vis_src_u8 = src_u8_orig[b : b + 1]
                    vis_tgt_u8 = tgt_u8_orig[b : b + 1]
                    vis_src_depth_gt = (src_depth_gt_orig[b : b + 1] if has_depth_gt_orig else None)
                    vis_tgt_depth_gt = (tgt_depth_gt_orig[b : b + 1] if has_depth_gt_orig else None)
                    vis_src_render_k3 = src_k3_orig[b : b + 1]
                    vis_tgt_render_k3 = tgt_k3_orig[b : b + 1]
                    vis_src_out = self.renderer(
                        g_world,
                        extrinsics=ident,
                        intrinsics=to_k4(vis_src_render_k3),
                        image_width=int(vis_src_u8.shape[-1]),
                        image_height=int(vis_src_u8.shape[-2]),
                    )
                    vis_tgt_out = self.renderer(
                        g_world,
                        extrinsics=rel_tgt_w2c,
                        intrinsics=to_k4(vis_tgt_render_k3),
                        image_width=int(vis_tgt_u8.shape[-1]),
                        image_height=int(vis_tgt_u8.shape[-2]),
                    )
                src_unik3d_depth = None
                tgt_unik3d_depth = None
                raw_dist = out.get("unik3d_distance", None)
                if torch.is_tensor(raw_dist):
                    try:
                        conditioning_rays = out.get("unik3d_ray_conditioning_rays", None)
                        if not torch.is_tensor(conditioning_rays):
                            conditioning_rays = out.get("unik3d_rays", None)
                        ray_z = (
                            conditioning_rays[b : b + 1, 2:3].detach()
                            if torch.is_tensor(conditioning_rays)
                            else None
                        )
                        if torch.is_tensor(ray_z):
                            if tuple(ray_z.shape[-2:]) != tuple(raw_dist.shape[-2:]):
                                ray_z = F.interpolate(ray_z, size=raw_dist.shape[-2:], mode="bilinear", align_corners=False)
                            src_unik3d_depth = raw_dist[b : b + 1, 0:1].detach() * ray_z
                        else:
                            src_unik3d_depth = self._distance_to_z_depth_pinhole(
                                raw_dist[b : b + 1, 0:1].detach(),
                                src_k_render_b,
                            )
                    except Exception:
                        src_unik3d_depth = raw_dist[b : b + 1, 0:1].detach()
                if self.enable_tgt_unik3d_vis:
                    try:
                        with torch.no_grad():
                            from unisharp.utils.unik3d_adapter import forward_unik3d_pinhole

                            unik_tgt = forward_unik3d_pinhole(
                                self._base_model().feature_extractor.unik3d,
                                rgb_u8=tgt_u8[b : b + 1],
                                intrinsics=tgt_k3[b : b + 1],
                                normalize=True,
                            )
                            dist_tgt = unik_tgt.get("distance", None) if isinstance(unik_tgt, dict) else None
                            if torch.is_tensor(dist_tgt):
                                try:
                                    tgt_unik3d_depth = self._distance_to_z_depth_pinhole(
                                        dist_tgt[:, 0:1].detach(),
                                        tgt_k_render_b,
                                    )
                                except Exception:
                                    tgt_unik3d_depth = dist_tgt[:, 0:1].detach()
                    except Exception:
                        tgt_unik3d_depth = None

                vis_payload = {
                    "src_gt": (vis_src_u8.float() / 255.0).detach(),
                    "src_pred": vis_src_out.color.clamp(0, 1).detach(),
                    "src_alpha": vis_src_out.alpha.detach(),
                    "src_gt_depth": (vis_src_depth_gt.detach() if torch.is_tensor(vis_src_depth_gt) else None),
                    "src_pred_depth": vis_src_out.depth.detach(),
                    "src_unik3d_depth": src_unik3d_depth,
                    "tgt_gt": (vis_tgt_u8.float() / 255.0).detach(),
                    "tgt_pred": vis_tgt_out.color.clamp(0, 1).detach(),
                    "tgt_alpha": vis_tgt_out.alpha.detach(),
                    "tgt_gt_depth": (vis_tgt_depth_gt.detach() if torch.is_tensor(vis_tgt_depth_gt) else None),
                    "tgt_pred_depth": vis_tgt_out.depth.detach(),
                    "tgt_unik3d_depth": tgt_unik3d_depth,
                    "dataset_name": str(dataset_name),
                    "scene": str(self._item_at(getattr(batch, "scene", None), b, "unknown")),
                    "src_idx": int(self._item_at(getattr(batch, "src_idx", None), b, -1)),
                    "tgt_idx": int(self._item_at(getattr(batch, "tgt_idx", None), b, -1)),
                    "src_pose_w2c": src_w2c[b : b + 1].detach(),
                    "tgt_pose_w2c": tgt_w2c[b : b + 1].detach(),
                    "tgt_metric_mask_raw": tgt_frustum_mask_raw.detach(),
                    "tgt_metric_mask": tgt_frustum_mask.detach(),
                }

            return {
                "src_pred_rgb_linear": src_out.color,
                "src_pred_alpha": src_out.alpha,
                "src_pred_depth_m": src_depth_pred,
                "src_pred_depth2_m": src_depth2_pred,
                "src_gt_rgb_u8": src_u8[b : b + 1],
                "src_gt_depth_m": (src_depth_gt_dist[b : b + 1] if has_depth_gt and src_depth_gt_dist is not None else zeros_src_depth),
                "src_mask": ones_mask,
                "src_apply_depth": False,
                "src_apply_grad": bool(has_depth_gt),
                "src_apply_grad_img": bool(has_depth_gt),
                "src_grad_img_circular_h": False,
                "tgt_pred_rgb_linear": tgt_out.color,
                "tgt_pred_alpha": tgt_out.alpha,
                "tgt_pred_depth_m": tgt_depth_pred,
                "tgt_gt_rgb_u8": tgt_u8[b : b + 1],
                "tgt_gt_depth_m": (tgt_depth_gt_dist[b : b + 1] if has_depth_gt and tgt_depth_gt_dist is not None else zeros_tgt_depth),
                "tgt_mask": tgt_frustum_mask,
                "tgt_depth_mask": tgt_depth_loss_mask,
                "tgt_apply_depth": bool(has_depth_gt),
                "tgt_apply_grad_img": bool(has_depth_gt),
                "tgt_grad_img_circular_h": False,
                "tgt_apply_percep": bool(float(self.loss_fn.w.lambda_percep) > 0.0),
                "tgt_extra_loss_terms": tgt_extra_loss_terms,
                "aux_losses": self._aux_ray_losses(
                    pred_rays=(
                        out.get("unik3d_rays", None)[b : b + 1]
                        if torch.is_tensor(out.get("unik3d_rays", None))
                        else None
                    ),
                    gt_rays=(
                        aux_ray_target_all[b : b + 1]
                        if torch.is_tensor(aux_ray_target_all)
                        else None
                    ),
                    mask=ones_mask,
                    pred_distance=(
                        out["unik3d_distance"][b : b + 1, 0:1]
                        if torch.is_tensor(out.get("unik3d_distance", None))
                        else None
                    ),
                    pred_distance2=src_depth2_pred,
                    gt_distance=(
                        src_unik3d_gt_dist[b : b + 1]
                        if torch.is_tensor(src_unik3d_gt_dist)
                        else None
                    ),
                    gt_distance2=src_depth2_gt_for_aux,
                    depth_mask=(src_depth_gt[b : b + 1] > 0.0 if has_depth_gt else None),
                    depth_mask2=src_depth2_mask_for_aux,
                ),
                "gaussian_scales": reg_inputs["gaussian_scales"],
                "gaussian_quaternions": reg_inputs["gaussian_quaternions"],
                "gaussian_angular_cell": reg_inputs["gaussian_angular_cell"],
                "delta_xy": reg_inputs["delta_xy_eff"],
                "delta_rho": reg_inputs["delta_rho_raw"],
                "delta_grid": reg_inputs["delta_grid"],
                "gaussian_mean_vectors": reg_inputs["gaussian_mean_vectors"],
                "gaussian_base_mean_vectors": reg_inputs["gaussian_base_mean_vectors"],
                "gaussian_opacities": reg_inputs["gaussian_opacities"],
                "gauss_grid_shape": reg_inputs["gauss_grid_shape"],
                "projected_scale_factor": reg_inputs["projected_scale_factor"],
                "projection_model": "pinhole",
                "projection_intrinsics": src_k_render_b,
                "vis_payload": vis_payload,
            }

        return _ModeStrategy(
            batch_size=int(src.shape[0]),
            gaussians=gaussians,
            make_world_gaussians=make_world_gaussians,
            make_sample=make_sample,
            collect_all_vis=bool(getattr(batch, "collect_all_vis", False)),
        )

    def _build_fisheye624_strategy(
        self,
        batch: Any,
        step: int,
        need_vis: bool = False,
        dataset_name: str = "scannetpp_fisheye",
    ) -> _ModeStrategy:
        del step
        src_u8 = self._as_bchw_rgb_u8(batch.src_rgb_u8.to(self.device, non_blocking=True))
        tgt_u8 = self._as_bchw_rgb_u8(batch.tgt_rgb_u8.to(self.device, non_blocking=True))
        src_depth_gt = self._clamp_distance_for_supervision(
            self._as_b1hw_depth(batch.src_depth_m.to(self.device, non_blocking=True).to(torch.float32))
        )
        tgt_depth_gt = self._clamp_distance_for_supervision(
            self._as_b1hw_depth(batch.tgt_depth_m.to(self.device, non_blocking=True).to(torch.float32))
        )
        src_valid_mask = self._as_b1hw_depth(batch.src_valid_mask.to(self.device, non_blocking=True).to(torch.float32))
        tgt_valid_mask = self._as_b1hw_depth(batch.tgt_valid_mask.to(self.device, non_blocking=True).to(torch.float32))
        src_w2c = self._as_b44_pose(batch.src_w2c.to(self.device, non_blocking=True).to(torch.float32))
        tgt_w2c = self._as_b44_pose(batch.tgt_w2c.to(self.device, non_blocking=True).to(torch.float32))
        src_cam_params = self._as_b16_camera_params(
            batch.src_camera_params.to(self.device, non_blocking=True).to(torch.float32)
        )
        tgt_cam_params = self._as_b16_camera_params(
            batch.tgt_camera_params.to(self.device, non_blocking=True).to(torch.float32)
        )
        distance_init_cap_m = self._distance_init_cap_for_dataset(dataset_name)

        out = self.model(
            image=src_u8.float().clamp(0, 255) / 255.0,
            image_u8=src_u8,
            camera_intrinsics=None,
            camera_params=src_cam_params,
            camera_model="fisheye624",
            depth_gt=src_depth_gt,
            distance_init_cap_m=distance_init_cap_m,
            validity_mask=src_valid_mask,
            return_aux=True,
        )

        gaussians = out["gaussians"]
        src_render_cam_params = src_cam_params
        tgt_render_cam_params = tgt_cam_params
        src_render_valid_mask = src_valid_mask
        tgt_render_valid_mask = tgt_valid_mask
        aux_ray_target_all = out.get("unik3d_gt_rays", None)

        def make_world_gaussians(b: int, g_b: Any) -> Any:
            return transform_gaussians_to_world(g_b, src_w2c[b])

        def make_sample(b: int, g_world: Any, enable_vis: bool) -> dict[str, Any]:
            src_h = int(src_u8.shape[-2])
            src_w = int(src_u8.shape[-1])
            tgt_h = int(tgt_u8.shape[-2])
            tgt_w = int(tgt_u8.shape[-1])
            src_render = render_gaussians_fisheye624(
                g_world,
                extrinsics_w2c=src_w2c[b : b + 1],
                camera_params=src_render_cam_params[b : b + 1],
                image_h=src_h,
                image_w=src_w,
                valid_mask=src_render_valid_mask[b : b + 1],
            )
            tgt_render = render_gaussians_fisheye624(
                g_world,
                extrinsics_w2c=tgt_w2c[b : b + 1],
                camera_params=tgt_render_cam_params[b : b + 1],
                image_h=tgt_h,
                image_w=tgt_w,
                valid_mask=tgt_render_valid_mask[b : b + 1],
            )
            reg_inputs = self._collect_regularization_inputs(
                out=out,
                gaussians=gaussians,
                b=b,
                projected_scale_factor=None,
            )
            tgt_depth_for_mask = self._pick_depth_for_fisheye_frustum_mask(
                gt_depth=tgt_depth_gt[b : b + 1],
                pred_depth=tgt_render["depth_distance"],
                gt_valid_mask=tgt_valid_mask[b : b + 1],
            )
            tgt_frustum_mask = compute_fisheye624_frustum_mask(
                depth_distance_m=tgt_depth_for_mask,
                tgt_w2c=tgt_w2c[b : b + 1],
                src_w2c=src_w2c[b : b + 1],
                tgt_camera_params=tgt_render_cam_params[b : b + 1],
                src_camera_params=src_render_cam_params[b : b + 1],
                src_valid_mask=src_render_valid_mask[b : b + 1] * src_render["valid_mask"],
                source_depth_distance_m=src_depth_gt[b : b + 1],
            )
            src_mask = src_render_valid_mask[b : b + 1] * src_render["valid_mask"]
            src_depth_mask = src_mask
            tgt_mask = tgt_render_valid_mask[b : b + 1] * tgt_render["valid_mask"] * tgt_frustum_mask
            tgt_mask_raw = tgt_mask
            tgt_mask = self._erode_supervision_mask(
                tgt_mask,
                self.target_mask_erode_px,
                circular_h=False,
            )
            src_depth_pred = self._clamp_distance_for_supervision(
                out["distance_layers"][b : b + 1, 0:1],
                clamp_max=False,
            )
            src_depth2_pred = (
                self._clamp_distance_for_supervision(out["distance_layers"][b : b + 1, 1:2], clamp_max=False)
                if out["distance_layers"] is not None and out["distance_layers"].shape[1] > 1
                else None
            )
            tgt_depth_pred = self._clamp_distance_for_supervision(tgt_render["depth_distance"], clamp_max=False)
            tgt_depth_loss_mask = self._rendered_depth_valid_for_inv_loss(tgt_depth_pred, tgt_render["alpha"])

            src_loss_terms = [
                {
                    "pred_rgb_linear": src_render["color"],
                    "pred_alpha": src_render["alpha"],
                    "pred_depth_m": src_render["depth_distance"],
                    "pred_depth2_m": None,
                    "gt_rgb_u8": src_u8[b : b + 1],
                    "gt_depth_m": src_depth_gt[b : b + 1],
                    "mask": src_mask,
                    "apply_color": True,
                    "apply_alpha": True,
                    "apply_depth": False,
                    "apply_percep": False,
                    "apply_tv": False,
                    "apply_grad": False,
                    "apply_grad_img": False,
                    "grad_img_circular_h": False,
                    "gaussian_scales": None,
                    "gaussian_quaternions": None,
                    "gaussian_angular_cell": None,
                    "delta_xy": None,
                    "gaussian_mean_vectors": None,
                    "gaussian_opacities": None,
                    "gauss_grid_shape": None,
                    "projected_scale_factor": None,
                    "apply_splat": False,
                    "loss_scale": 1.0,
                }
            ]
            src_extra_loss_terms = [
                {
                    "pred_rgb_linear": torch.zeros((1, 3, src_h, src_w), dtype=torch.float32, device=self.device),
                    "pred_alpha": torch.zeros((1, 1, src_h, src_w), dtype=torch.float32, device=self.device),
                    "pred_depth_m": src_depth_pred,
                    "pred_depth2_m": src_depth2_pred,
                    "gt_rgb_u8": torch.zeros((1, 3, src_h, src_w), dtype=torch.uint8, device=self.device),
                    "gt_depth_m": src_depth_gt[b : b + 1],
                    "mask": src_depth_mask,
                    "apply_color": False,
                    "apply_alpha": False,
                    "apply_depth": False,
                    "apply_percep": False,
                    "apply_tv": True,
                    "apply_grad": True,
                    "apply_grad_img": True,
                    "grad_img_circular_h": False,
                    "gaussian_scales": reg_inputs["gaussian_scales"],
                    "gaussian_quaternions": reg_inputs["gaussian_quaternions"],
                    "gaussian_angular_cell": reg_inputs["gaussian_angular_cell"],
                    "delta_xy": reg_inputs["delta_xy_eff"],
                    "delta_rho": reg_inputs["delta_rho_raw"],
                    "delta_grid": reg_inputs["delta_grid"],
                    "gaussian_mean_vectors": reg_inputs["gaussian_mean_vectors"],
                    "gaussian_base_mean_vectors": reg_inputs["gaussian_base_mean_vectors"],
                    "gaussian_opacities": reg_inputs["gaussian_opacities"],
                    "gauss_grid_shape": reg_inputs["gauss_grid_shape"],
                    "projected_scale_factor": None,
                    "projection_model": "fisheye624",
                    "projection_camera_params": src_render_cam_params[b : b + 1],
                    "apply_splat": True,
                    "loss_scale": 1.0,
                }
            ]
            tgt_extra_loss_terms = []

            vis_payload = None
            if enable_vis:
                src_unik3d_depth = out["unik3d_distance"][b : b + 1, 0:1].detach() if torch.is_tensor(out.get("unik3d_distance", None)) else None
                tgt_unik3d_depth = None
                if (
                    tgt_unik3d_depth is None
                    and self.enable_tgt_unik3d_vis
                ):
                    try:
                        with torch.no_grad():
                            from unisharp.utils.unik3d_adapter import forward_unik3d_fisheye624

                            unik_tgt = forward_unik3d_fisheye624(
                                self._base_model().feature_extractor.unik3d,
                                rgb_u8=tgt_u8[b : b + 1],
                                camera_params=tgt_render_cam_params[b : b + 1],
                                normalize=True,
                                validity_mask=tgt_valid_mask[b : b + 1],
                            )
                            dist_tgt = unik_tgt.get("distance", None) if isinstance(unik_tgt, dict) else None
                            if torch.is_tensor(dist_tgt):
                                tgt_unik3d_depth = dist_tgt[:, 0:1].detach()
                    except Exception:
                        tgt_unik3d_depth = None
                vis_payload = {
                    "src_gt": (src_u8[b : b + 1].float() / 255.0).detach(),
                    "src_pred": src_render["color"].clamp(0, 1).detach(),
                    "src_alpha": src_render["alpha"].detach(),
                    "src_gt_depth": src_depth_gt[b : b + 1].detach(),
                    "src_pred_depth": src_render["depth_distance"].detach(),
                    "src_unik3d_depth": src_unik3d_depth,
                    "src_metric_mask": src_mask.detach(),
                    "tgt_gt": (tgt_u8[b : b + 1].float() / 255.0).detach(),
                    "tgt_pred": tgt_render["color"].clamp(0, 1).detach(),
                    "tgt_alpha": tgt_render["alpha"].detach(),
                    "tgt_gt_depth": tgt_depth_gt[b : b + 1].detach(),
                    "tgt_pred_depth": tgt_depth_pred.detach(),
                    "tgt_unik3d_depth": tgt_unik3d_depth,
                    "dataset_name": str(dataset_name),
                    "scene": str(self._first_item(getattr(batch, "scene", None), "unknown")),
                    "src_idx": int(self._first_item(getattr(batch, "src_idx", None), -1)),
                    "tgt_idx": int(self._first_item(getattr(batch, "tgt_idx", None), -1)),
                    "src_pose_w2c": src_w2c[b : b + 1].detach(),
                    "tgt_pose_w2c": tgt_w2c[b : b + 1].detach(),
                    "tgt_metric_mask_raw": tgt_mask_raw.detach(),
                    "tgt_metric_mask": tgt_mask.detach(),
                }


            return {
                "src_loss_terms": src_loss_terms,
                "src_extra_loss_terms": src_extra_loss_terms,
                "tgt_pred_rgb_linear": tgt_render["color"],
                "tgt_pred_alpha": tgt_render["alpha"],
                "tgt_pred_depth_m": tgt_depth_pred,
                "tgt_gt_rgb_u8": tgt_u8[b : b + 1],
                "tgt_gt_depth_m": tgt_depth_gt[b : b + 1],
                "tgt_mask": tgt_mask,
                "tgt_depth_mask": tgt_depth_loss_mask,
                "tgt_apply_depth": True,
                "tgt_apply_grad_img": True,
                "tgt_apply_splat": False,
                "tgt_grad_img_circular_h": False,
                "tgt_apply_percep": bool(float(self.loss_fn.w.lambda_percep) > 0.0),
                "tgt_extra_loss_terms": tgt_extra_loss_terms,
                "aux_losses": self._aux_ray_losses(
                    pred_rays=(
                        out.get("unik3d_rays", None)[b : b + 1]
                        if torch.is_tensor(out.get("unik3d_rays", None))
                        else None
                    ),
                    gt_rays=(
                        aux_ray_target_all[b : b + 1]
                        if torch.is_tensor(aux_ray_target_all)
                        else None
                    ),
                    mask=src_render_valid_mask[b : b + 1],
                    pred_distance=(
                        out["unik3d_distance"][b : b + 1, 0:1]
                        if torch.is_tensor(out.get("unik3d_distance", None))
                        else None
                    ),
                    pred_distance2=None,
                    gt_distance=src_depth_gt[b : b + 1],
                    depth_mask=src_valid_mask[b : b + 1],
                ),
                "gaussian_scales": reg_inputs["gaussian_scales"],
                "gaussian_quaternions": reg_inputs["gaussian_quaternions"],
                "gaussian_angular_cell": reg_inputs["gaussian_angular_cell"],
                "delta_xy": reg_inputs["delta_xy_eff"],
                "delta_rho": reg_inputs["delta_rho_raw"],
                "delta_grid": reg_inputs["delta_grid"],
                "gaussian_mean_vectors": reg_inputs["gaussian_mean_vectors"],
                "gaussian_base_mean_vectors": reg_inputs["gaussian_base_mean_vectors"],
                "gaussian_opacities": reg_inputs["gaussian_opacities"],
                "gauss_grid_shape": reg_inputs["gauss_grid_shape"],
                "projected_scale_factor": reg_inputs["projected_scale_factor"],
                "projection_model": "fisheye624",
                "projection_camera_params": src_render_cam_params[b : b + 1],
                "vis_payload": vis_payload,
            }

        return _ModeStrategy(
            batch_size=int(src_u8.shape[0]),
            gaussians=gaussians,
            make_world_gaussians=make_world_gaussians,
            make_sample=make_sample,
            collect_all_vis=bool(getattr(batch, "collect_all_vis", False)),
        )

    def _build_fisheye_strategy(
        self,
        batch: Any,
        step: int,
        need_vis: bool = False,
        dataset_name: str = "fisheye",
    ) -> _ModeStrategy:
        camera_model = str(getattr(batch, "camera_model", "fisheye624")).lower()
        if camera_model != "fisheye624":
            raise ValueError(
                f"Unsupported fisheye camera_model={camera_model!r}; expected 'fisheye624'."
            )
        return self._build_fisheye624_strategy(
            batch,
            step,
            need_vis=need_vis,
            dataset_name=dataset_name,
        )

    def _build_spherical_strategy(
        self,
        batch: Any,
        step: int,
        need_vis: bool = False,
        dataset_name: str = "hm3d",
    ) -> _ModeStrategy:
        src_erp_u8 = batch.src_erp_rgb_u8.to(self.device, non_blocking=True)
        tgt_erp_u8 = batch.tgt_erp_rgb_u8.to(self.device, non_blocking=True)
        src_erp_depth = self._clamp_distance_for_supervision(
            batch.src_erp_depth_m.to(self.device, non_blocking=True)
        )
        tgt_erp_depth = self._clamp_distance_for_supervision(
            batch.tgt_erp_depth_m.to(self.device, non_blocking=True)
        )
        src_cdep = self._sanitize_positive_depth(
            batch.src_cube_depth_m.to(self.device, non_blocking=True)
        )
        tgt_cdep = self._sanitize_positive_depth(
            batch.tgt_cube_depth_m.to(self.device, non_blocking=True)
        )
        disable_depth_gt = bool(getattr(batch, "disable_depth_gt", False))
        
        src_R = batch.src_R.to(self.device, non_blocking=True)
        src_t = batch.src_t.to(self.device, non_blocking=True)
        tgt_R = batch.tgt_R.to(self.device, non_blocking=True)
        tgt_t = batch.tgt_t.to(self.device, non_blocking=True)
        
        cur_bs = int(src_erp_u8.shape[0])
        erp_h = int(src_erp_u8.shape[-2])
        erp_w = int(src_erp_u8.shape[-1])
        cube_face_w = int(batch.src_cube_depth_m.shape[2]) if torch.is_tensor(batch.src_cube_depth_m) else max(1, erp_h // 2)
        
        use_flip_yz = str(dataset_name).lower() not in {"sim", "smx_sim_fisheye"}
        pose_convs_per_sample = ["c2w"] * cur_bs
        flip_yz_per_sample = [bool(use_flip_yz)] * cur_bs
        
        extr_src_base = torch.stack(
            [build_extrinsics_w2c(src_R[i], src_t[i], pose_convs_per_sample[i]) for i in range(cur_bs)],
            dim=0
        )
        extr_tgt_base = torch.stack(
            [build_extrinsics_w2c(tgt_R[i], tgt_t[i], pose_convs_per_sample[i]) for i in range(cur_bs)],
            dim=0
        )
        
        with torch.autocast("cuda", enabled=False):
            c2w_src = torch.linalg.inv(extr_src_base.to(torch.float32))
            c2w_tgt = torch.linalg.inv(extr_tgt_base.to(torch.float32))
            
            flip_mask = torch.tensor(flip_yz_per_sample, device=c2w_src.device, dtype=torch.bool)
            negate_relative_z = False
            if bool(flip_mask.any().item()):
                flip_mode = os.environ.get("PANO_POSE_FLIP_CONVENTION", "flip_yz_negate_rel_z").strip().lower()
                negate_relative_z = flip_mode in {
                    "flip_yz_negate_rel_z",
                    "flip_yz_invert_z_translation",
                    "flip_yz_neg_z",
                }
                if flip_mode in {"flip_y_only", "y", "y_only"}:
                    diag = [1.0, -1.0, 1.0, 1.0]
                elif flip_mode in {"none", "identity", "no_flip"}:
                    diag = [1.0, 1.0, 1.0, 1.0]
                else:
                    diag = [1.0, -1.0, -1.0, 1.0]
                D = torch.diag(torch.tensor(diag, device=c2w_src.device, dtype=torch.float32))
                c2w_src = c2w_src.clone()
                c2w_tgt = c2w_tgt.clone()
                c2w_src[flip_mask] = c2w_src[flip_mask] @ D
                c2w_tgt[flip_mask] = c2w_tgt[flip_mask] @ D
            
            ref_inv = torch.linalg.inv(c2w_src.to(torch.float32))
            c2w_src = ref_inv @ c2w_src
            c2w_tgt = ref_inv @ c2w_tgt
            if negate_relative_z:
                c2w_tgt = c2w_tgt.clone()
                c2w_tgt[flip_mask, 2, 3] *= -1.0
            
            extr_src = torch.linalg.inv(c2w_src).to(dtype=extr_src_base.dtype)
            extr_tgt = torch.linalg.inv(c2w_tgt).to(dtype=extr_tgt_base.dtype)
        
        src_erp = (src_erp_u8.float() / 255.0).clamp(0, 1)
        distance_init_cap_m = self._distance_init_cap_for_dataset(dataset_name)
        
        out = self.model(
            image=src_erp,
            image_u8=src_erp_u8,
            camera_intrinsics=None,
            camera_model="spherical",
            depth_gt=None if disable_depth_gt else src_erp_depth,
            distance_init_cap_m=distance_init_cap_m,
            return_aux=True,
        )
        gaussians = out["gaussians"]
        aux_ray_target_all = out.get("unik3d_gt_rays", None)

        def make_world_gaussians(b: int, g_b: Any) -> Any:
            return transform_gaussians_to_world(g_b, extr_src[b])

        def make_sample(b: int, g_world: Any, enable_vis: bool) -> dict[str, Any]:
            src_rgb, src_depth, src_alpha = self._render_cubemap(g_world, extr_src[b], face_w=cube_face_w)
            tgt_rgb, tgt_depth, tgt_alpha = self._render_cubemap(g_world, extr_tgt[b], face_w=cube_face_w)

            src_erp_pred = self._cube_to_erp(src_rgb, equ_h=erp_h, equ_w=erp_w, face_w=cube_face_w)
            tgt_erp_pred = self._cube_to_erp(tgt_rgb, equ_h=erp_h, equ_w=erp_w, face_w=cube_face_w)
            src_erp_alpha = self._cube_to_erp(src_alpha, equ_h=erp_h, equ_w=erp_w, face_w=cube_face_w)
            tgt_erp_alpha = self._cube_to_erp(tgt_alpha, equ_h=erp_h, equ_w=erp_w, face_w=cube_face_w)
            src_depth_dist = self._clamp_distance_for_supervision(
                self._cubemap_z_depth_to_distance(src_depth),
                clamp_max=False,
            )
            tgt_depth_dist = self._clamp_distance_for_supervision(
                self._cubemap_z_depth_to_distance(tgt_depth),
                clamp_max=False,
            )
            src_erp_depth_render = self._cube_to_erp(
                src_depth_dist, equ_h=erp_h, equ_w=erp_w, face_w=cube_face_w
            ).clamp(min=1e-4)

            src_erp_depth_pred = self._clamp_distance_for_supervision(
                out["distance_layers"][b : b + 1, 0:1],
                clamp_max=False,
            )
            src_erp_depth2_pred = (
                self._clamp_distance_for_supervision(out["distance_layers"][b : b + 1, 1:2], clamp_max=False)
                if out["distance_layers"] is not None and out["distance_layers"].shape[1] > 1
                else None
            )
            tgt_erp_depth_pred = self._cube_to_erp(
                tgt_depth_dist, equ_h=erp_h, equ_w=erp_w, face_w=cube_face_w
            ).clamp(min=1e-4)
            tgt_depth_loss_mask = self._rendered_depth_valid_for_inv_loss(tgt_erp_depth_pred, tgt_erp_alpha)
            depth_novel = self._pick_depth_for_cubemap_frustum_mask(
                gt_depth_cube=None if disable_depth_gt else (tgt_cdep[b : b + 1][0] if torch.is_tensor(tgt_cdep) else None),
                pred_depth_cube=tgt_depth_dist,
                face_w=cube_face_w,
            )
            source_depth_for_visibility = self._pick_depth_for_cubemap_frustum_mask(
                gt_depth_cube=None if disable_depth_gt else (src_cdep[b : b + 1][0] if torch.is_tensor(src_cdep) else None),
                pred_depth_cube=src_depth_dist,
                face_w=cube_face_w,
            )
            mask_bool = view_frustum_mask_cubemap_union(
                depth_novel=depth_novel,
                extr_novel_w2c=extr_tgt[b],
                extr_source_w2c=extr_src[b],
                face_w=int(cube_face_w),
                source_depth=source_depth_for_visibility,
            )
            mask_erp = self._cube_to_erp(
                mask_bool[:, None].to(torch.float32), equ_h=erp_h, equ_w=erp_w, face_w=cube_face_w
            )

            gt_src_erp_u8 = src_erp_u8[b : b + 1]
            gt_tgt_erp_u8 = tgt_erp_u8[b : b + 1]
            gt_src_erp_depth = src_erp_depth[b : b + 1]
            gt_tgt_erp_depth = tgt_erp_depth[b : b + 1]
            gt_src_cube_u8 = batch.src_cube_rgb_u8[b].to(self.device, non_blocking=True).permute(0, 3, 1, 2).contiguous()
            gt_tgt_cube_u8 = batch.tgt_cube_rgb_u8[b].to(self.device, non_blocking=True).permute(0, 3, 1, 2).contiguous()

            src_valid = torch.ones_like(gt_src_erp_depth) if disable_depth_gt else (gt_src_erp_depth > 0.0).to(dtype=torch.float32)
            tgt_valid = torch.ones_like(gt_tgt_erp_depth) if disable_depth_gt else (gt_tgt_erp_depth > 0.0).to(dtype=torch.float32)
            src_mask = torch.ones_like(src_valid)
            tgt_mask = (mask_erp.to(dtype=torch.float32) * tgt_valid).clamp(0.0, 1.0)
            tgt_mask_raw = tgt_mask
            tgt_mask = self._erode_supervision_mask(
                tgt_mask,
                self.target_mask_erode_px,
                circular_h=True,
            )
            src_cube_mask = torch.ones_like(src_alpha)
            if str(dataset_name).lower() == "hm3d" and (not disable_depth_gt) and torch.is_tensor(src_cdep):
                src_cube_valid = (src_cdep[b : b + 1][0, ..., 0] > 0.0).to(dtype=src_alpha.dtype).unsqueeze(1)
                if tuple(src_cube_valid.shape[-2:]) != tuple(src_alpha.shape[-2:]):
                    src_cube_valid = F.interpolate(
                        src_cube_valid,
                        size=src_alpha.shape[-2:],
                        mode="nearest",
                    )
                src_cube_mask = src_cube_valid.to(device=src_alpha.device, dtype=src_alpha.dtype).clamp(0.0, 1.0)
            tgt_cube_valid = (depth_novel[..., 0] > 0.0).to(dtype=torch.float32).unsqueeze(1)
            tgt_cube_mask = (mask_bool[:, None].to(dtype=torch.float32) * tgt_cube_valid).clamp(0.0, 1.0)
            tgt_cube_mask = self._erode_supervision_mask(
                tgt_cube_mask,
                self.target_mask_erode_px,
                circular_h=False,
            )

            src_cube_depth_zeros = torch.zeros_like(src_alpha)
            tgt_cube_depth_zeros = torch.zeros_like(tgt_alpha)
            src_erp_rgb_zeros = torch.zeros_like(src_erp_pred)
            tgt_erp_rgb_zeros = torch.zeros_like(tgt_erp_pred)
            src_erp_u8_zeros = torch.zeros_like(gt_src_erp_u8)
            tgt_erp_u8_zeros = torch.zeros_like(gt_tgt_erp_u8)

            erp_proj_scale = 0.5 * (
                float(erp_w) / (2.0 * 3.141592653589793)
                + float(erp_h) / 3.141592653589793
            )
            reg_inputs = self._collect_regularization_inputs(
                out=out,
                gaussians=gaussians,
                b=b,
                projected_scale_factor=erp_proj_scale,
            )

            vis_payload = None
            if enable_vis:
                src_unik3d_depth = None
                tgt_unik3d_depth = None
                raw_dist = out.get("unik3d_distance", None)
                if torch.is_tensor(raw_dist):
                    src_unik3d_depth = raw_dist[b : b + 1, 0:1].detach()

                vis_payload = {
                    "src_gt": (gt_src_erp_u8.float() / 255.0).detach(),
                    "src_pred": src_erp_pred.clamp(0, 1).detach(),
                    "src_alpha": src_erp_alpha.detach(),
                    "src_gt_depth": None if disable_depth_gt else gt_src_erp_depth.detach(),
                    "src_pred_depth": src_erp_depth_render.detach(),
                    "src_unik3d_depth": src_unik3d_depth,
                    "tgt_gt": (gt_tgt_erp_u8.float() / 255.0).detach(),
                    "tgt_pred": tgt_erp_pred.clamp(0, 1).detach(),
                    "tgt_alpha": tgt_erp_alpha.detach(),
                    "tgt_gt_depth": None if disable_depth_gt else gt_tgt_erp_depth.detach(),
                    "tgt_pred_depth": tgt_erp_depth_pred.detach(),
                    "tgt_unik3d_depth": tgt_unik3d_depth,
                    "dataset_name": str(dataset_name),
                    "scene": str(self._item_at(getattr(batch, "scene", None), b, "unknown")),
                    "src_idx": int(self._item_at(getattr(batch, "src_idx", None), b, -1)),
                    "tgt_idx": int(self._item_at(getattr(batch, "tgt_idx", None), b, -1)),
                    "src_pose_w2c": extr_src[b : b + 1].detach(),
                    "tgt_pose_w2c": extr_tgt[b : b + 1].detach(),
                    "src_cube_gt_u8": (
                        batch.src_cube_rgb_u8[b].detach()
                        if hasattr(batch, "src_cube_rgb_u8") and torch.is_tensor(batch.src_cube_rgb_u8)
                        else None
                    ),
                    "tgt_cube_gt_u8": (
                        batch.tgt_cube_rgb_u8[b].detach()
                        if hasattr(batch, "tgt_cube_rgb_u8") and torch.is_tensor(batch.tgt_cube_rgb_u8)
                        else None
                    ),
                    "src_cube_pred_linear": src_rgb.detach(),
                    "tgt_cube_pred_linear": tgt_rgb.detach(),
                    "src_cube_alpha": src_alpha.detach(),
                    "tgt_cube_alpha": tgt_alpha.detach(),
                    "tgt_metric_mask_raw": tgt_mask_raw.detach(),
                    "tgt_metric_mask": tgt_mask.detach(),
                }

            tgt_loss_terms = [
                {
                    "pred_rgb_linear": tgt_rgb,
                    "pred_alpha": tgt_alpha,
                    "pred_depth_m": tgt_cube_depth_zeros,
                    "pred_depth2_m": None,
                    "gt_rgb_u8": gt_tgt_cube_u8,
                    "gt_depth_m": tgt_cube_depth_zeros,
                    "mask": tgt_cube_mask,
                    "apply_color": True,
                    "apply_alpha": True,
                    "apply_depth": False,
                    "apply_percep": bool(float(self.loss_fn.w.lambda_percep) > 0.0),
                    "apply_tv": False,
                    "apply_grad": False,
                    "apply_grad_img": False,
                    "grad_img_circular_h": False,
                    "gaussian_scales": None,
                    "gaussian_quaternions": None,
                    "gaussian_angular_cell": None,
                    "delta_xy": None,
                    "gaussian_mean_vectors": None,
                    "gaussian_opacities": None,
                    "gauss_grid_shape": None,
                    "projected_scale_factor": None,
                },
                {
                    "pred_rgb_linear": tgt_erp_rgb_zeros,
                    "pred_alpha": torch.zeros_like(tgt_erp_depth_pred),
                    "pred_depth_m": tgt_erp_depth_pred,
                    "pred_depth2_m": None,
                    "gt_rgb_u8": tgt_erp_u8_zeros,
                    "gt_depth_m": gt_tgt_erp_depth,
                    "mask": tgt_mask,
                    "depth_mask": tgt_depth_loss_mask,
                    "apply_color": False,
                    "apply_alpha": False,
                    "apply_depth": not disable_depth_gt,
                    "apply_percep": False,
                    "apply_tv": False,
                    "apply_grad": False,
                    "apply_grad_img": not disable_depth_gt,
                    "grad_img_circular_h": True,
                    "gaussian_scales": None,
                    "gaussian_quaternions": None,
                    "gaussian_angular_cell": None,
                    "delta_xy": None,
                    "gaussian_mean_vectors": None,
                    "gaussian_opacities": None,
                    "gauss_grid_shape": None,
                    "projected_scale_factor": reg_inputs["projected_scale_factor"],
                },
            ]
            return {
                "src_loss_terms": [
                    {
                        "pred_rgb_linear": src_rgb,
                        "pred_alpha": src_alpha,
                        "pred_depth_m": src_cube_depth_zeros,
                        "pred_depth2_m": None,
                        "gt_rgb_u8": gt_src_cube_u8,
                        "gt_depth_m": src_cube_depth_zeros,
                        "mask": src_cube_mask,
                        "apply_color": True,
                        "apply_alpha": True,
                        "apply_depth": False,
                        "apply_percep": False,
                        "apply_tv": False,
                        "apply_grad": False,
                        "apply_grad_img": False,
                        "grad_img_circular_h": False,
                        "gaussian_scales": None,
                        "gaussian_quaternions": None,
                        "gaussian_angular_cell": None,
                        "delta_xy": None,
                        "gaussian_mean_vectors": None,
                        "gaussian_opacities": None,
                        "gauss_grid_shape": None,
                        "projected_scale_factor": None,
                    },
                    {
                        "pred_rgb_linear": src_erp_rgb_zeros,
                        "pred_alpha": torch.zeros_like(src_erp_depth_pred),
                        "pred_depth_m": src_erp_depth_pred,
                        "pred_depth2_m": src_erp_depth2_pred,
                        "gt_rgb_u8": src_erp_u8_zeros,
                        "gt_depth_m": gt_src_erp_depth,
                        "mask": src_mask,
                        "apply_color": False,
                        "apply_alpha": False,
                        "apply_depth": False,
                        "apply_percep": False,
                        "apply_tv": True,
                        "apply_grad": False,
                        "apply_grad_img": not disable_depth_gt,
                        "grad_img_circular_h": True,
                        "gaussian_scales": reg_inputs["gaussian_scales"],
                        "gaussian_quaternions": reg_inputs["gaussian_quaternions"],
                        "gaussian_angular_cell": reg_inputs["gaussian_angular_cell"],
                        "delta_xy": reg_inputs["delta_xy_eff"],
                        "delta_rho": reg_inputs["delta_rho_raw"],
                        "delta_grid": reg_inputs["delta_grid"],
                        "gaussian_mean_vectors": reg_inputs["gaussian_mean_vectors"],
                        "gaussian_base_mean_vectors": reg_inputs["gaussian_base_mean_vectors"],
                        "gaussian_opacities": reg_inputs["gaussian_opacities"],
                        "gauss_grid_shape": reg_inputs["gauss_grid_shape"],
                        "projected_scale_factor": reg_inputs["projected_scale_factor"],
                        "projection_model": "erp",
                    },
                ],
                "tgt_loss_terms": tgt_loss_terms,
                "gaussian_scales": reg_inputs["gaussian_scales"],
                "gaussian_quaternions": reg_inputs["gaussian_quaternions"],
                "gaussian_angular_cell": reg_inputs["gaussian_angular_cell"],
                "delta_xy": reg_inputs["delta_xy_eff"],
                "delta_rho": reg_inputs["delta_rho_raw"],
                "delta_grid": reg_inputs["delta_grid"],
                "gaussian_mean_vectors": reg_inputs["gaussian_mean_vectors"],
                "gaussian_base_mean_vectors": reg_inputs["gaussian_base_mean_vectors"],
                "gaussian_opacities": reg_inputs["gaussian_opacities"],
                "gauss_grid_shape": reg_inputs["gauss_grid_shape"],
                "projected_scale_factor": reg_inputs["projected_scale_factor"],
                "projection_model": "erp",
                "aux_losses": self._aux_ray_losses(
                    pred_rays=(
                        out.get("unik3d_rays", None)[b : b + 1]
                        if torch.is_tensor(out.get("unik3d_rays", None))
                        else None
                    ),
                    gt_rays=(
                        aux_ray_target_all[b : b + 1]
                        if torch.is_tensor(aux_ray_target_all)
                        else None
                    ),
                    mask=src_valid,
                    pred_distance=(
                        out["unik3d_distance"][b : b + 1, 0:1]
                        if torch.is_tensor(out.get("unik3d_distance", None))
                        else None
                    ),
                    pred_distance2=src_erp_depth2_pred,
                    gt_distance=None if disable_depth_gt else gt_src_erp_depth,
                    depth_mask=src_valid,
                ),
                "vis_payload": vis_payload,
            }

        return _ModeStrategy(
            batch_size=int(cur_bs),
            gaussians=gaussians,
            make_world_gaussians=make_world_gaussians,
            make_sample=make_sample,
            collect_all_vis=bool(getattr(batch, "collect_all_vis", False)),
        )
    
    def _render_cubemap(
        self,
        gaussians: Any,
        extr_w2c: torch.Tensor,
        face_w: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = gaussians.mean_vectors.device
        intr = get_pinhole_intrinsics_4x4(int(face_w)).to(device=device)[None].expand(6, -1, -1)
        extr_faces = cubemap_face_cameras(extr_w2c, device=device)
        out = self.renderer(
            gaussians,
            extrinsics=extr_faces,
            intrinsics=intr,
            image_width=int(face_w),
            image_height=int(face_w),
        )
        return out.color.contiguous(), out.depth.contiguous(), out.alpha.contiguous()
    
    def _cube_to_erp(self, cube: torch.Tensor, equ_h: int, equ_w: int, face_w: int) -> torch.Tensor:
        cube = cube.permute(1, 0, 2, 3).unsqueeze(0)
        c2e = Cube2Equirec(face_w=int(face_w), equ_h=int(equ_h), equ_w=int(equ_w)).to(device=cube.device)
        return c2e(cube)

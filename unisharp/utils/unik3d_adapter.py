from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch
from unisharp.utils.pixel_convention import integer_pixel_center_grid


def _enable_unik3d_decoder_feature_capture(model: torch.nn.Module) -> None:
    try:
        rm = model.pixel_decoder.radial_module  # type: ignore[attr-defined]
    except Exception:
        return
    if getattr(rm, "_unisharp_process_wrapped", False):
        return

    import types

    try:
        orig_process = rm.process
    except Exception:
        return

    def wrapped_process(self, features_list, rays_embeddings):  # type: ignore[no-untyped-def]
        if bool(getattr(self, "_unisharp_detach_rays_embeddings", False)):
            rays_embeddings = rays_embeddings.detach()
        out_features, init_latents = orig_process(features_list, rays_embeddings)
        self._unisharp_last_out_features = out_features
        self._unisharp_last_init_latents = init_latents
        return out_features, init_latents

    rm.process = types.MethodType(wrapped_process, rm)  # type: ignore[method-assign]
    rm._unisharp_process_wrapped = True


def _enable_unik3d_force_gt_rays(model: torch.nn.Module) -> None:
    try:
        pixel_decoder = model.pixel_decoder  # type: ignore[attr-defined]
    except Exception:
        return
    if getattr(pixel_decoder, "_unisharp_run_camera_wrapped", False):
        return

    import types
    from einops import rearrange

    try:
        orig_run_camera = pixel_decoder.run_camera
    except Exception:
        return

    def wrapped_run_camera(self, cls_tokens, original_shapes, rays_gt):  # type: ignore[no-untyped-def]
        force_gt = bool(getattr(self, "_unisharp_force_gt_rays", False)) and torch.is_tensor(rays_gt)
        if force_gt:
            old_camera_gt = getattr(self, "camera_gt", None)
            try:
                self.camera_gt = False
                intrinsics, pred_rays = orig_run_camera(cls_tokens, original_shapes, rays_gt)
            finally:
                if old_camera_gt is not None:
                    self.camera_gt = old_camera_gt
            self._unisharp_last_pred_rays_flat = pred_rays
            rays = rearrange(rays_gt, "b c h w -> b (h w) c")
            return intrinsics, rays

        intrinsics, rays = orig_run_camera(cls_tokens, original_shapes, rays_gt)
        self._unisharp_last_pred_rays_flat = rays
        return intrinsics, rays

    pixel_decoder.run_camera = types.MethodType(wrapped_run_camera, pixel_decoder)  # type: ignore[method-assign]
    pixel_decoder._unisharp_run_camera_wrapped = True


def postprocess_unik3d_tensor(
    tensor: torch.Tensor,
    *,
    padded_hw: tuple[int, int],
    paddings: tuple[int, int, int, int],
    interpolation_mode: str,
) -> torch.Tensor:
    _ = _try_import_unik3d()
    from unik3d.models.unik3d import _postprocess  # type: ignore

    return _postprocess(
        tensor,
        padded_hw,
        paddings=paddings,
        interpolation_mode=interpolation_mode,
    )


def _erp_rays_panosplatt3r_opencv(
    h: int, w: int, device: torch.device, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    xs = (torch.arange(w, device=device, dtype=dtype) + 0.5) / float(w)
    ys = (torch.arange(h, device=device, dtype=dtype) + 0.5) / float(h)
    lon = (xs - 0.5) * (2.0 * torch.pi)
    lat = -(ys - 0.5) * torch.pi
    lat2d, lon2d = torch.meshgrid(lat, lon, indexing="ij")
    cos_lat = torch.cos(lat2d)
    x = torch.sin(lon2d) * cos_lat
    y_up = torch.sin(lat2d)
    z = torch.cos(lon2d) * cos_lat
    y = -y_up
    rays = torch.stack([x, y, z], dim=0)
    rays = rays / torch.norm(rays, dim=0, keepdim=True).clamp(min=1e-6)
    return rays


def _pinhole_rays_opencv(
    intrinsics: torch.Tensor,
    h: int,
    w: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if intrinsics.ndim == 2:
        intrinsics = intrinsics.unsqueeze(0)
    if intrinsics.ndim != 3 or tuple(intrinsics.shape[1:]) != (3, 3):
        raise ValueError(f"Expected intrinsics (B,3,3), got {tuple(intrinsics.shape)}")
    B = int(intrinsics.shape[0])
    K = intrinsics.to(device=device, dtype=dtype)
    fx = K[:, 0, 0].view(B, 1, 1)
    fy = K[:, 1, 1].view(B, 1, 1)
    cx = K[:, 0, 2].view(B, 1, 1)
    cy = K[:, 1, 2].view(B, 1, 1)

    uu, vv = integer_pixel_center_grid(h, w, device=device, dtype=dtype)
    uu = uu[None].expand(B, -1, -1)
    vv = vv[None].expand(B, -1, -1)

    x = (uu - cx) / fx
    y = (vv - cy) / fy
    z = torch.ones_like(x)
    rays = torch.stack([x, y, z], dim=1)
    rays = rays / torch.norm(rays, dim=1, keepdim=True).clamp(min=1e-6)
    return rays


def _fisheye624_rays_opencv_integer_centers(
    fisheye624_cls: Any,
    params: torch.Tensor,
    h: int,
    w: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if params.ndim == 1:
        params = params.unsqueeze(0)
    params = params.to(device=device, dtype=torch.float32)
    uu, vv = integer_pixel_center_grid(int(h), int(w), device=device, dtype=torch.float32)
    uv = torch.stack([uu, vv], dim=0).unsqueeze(0).expand(int(params.shape[0]), -1, -1, -1)
    rays = fisheye624_cls(params=params).unproject(uv).to(dtype=dtype)
    return rays / torch.norm(rays, dim=1, keepdim=True).clamp(min=1e-6)


def _fill_invalid_rgb_with_valid_mean(
    rgb: torch.Tensor,
    validity_mask: torch.Tensor | None,
) -> torch.Tensor:
    if validity_mask is None:
        return rgb
    if rgb.ndim != 4:
        raise ValueError(f"Expected rgb shape (B,3,H,W), got {tuple(rgb.shape)}")
    mask = validity_mask.to(device=rgb.device, dtype=torch.float32)
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4 or int(mask.shape[1]) != 1:
        raise ValueError(f"Expected validity_mask shape (B,1,H,W), got {tuple(mask.shape)}")
    if int(mask.shape[0]) == 1 and int(rgb.shape[0]) > 1:
        mask = mask.expand(int(rgb.shape[0]), -1, -1, -1)
    if tuple(mask.shape[-2:]) != tuple(rgb.shape[-2:]):
        import torch.nn.functional as F

        mask = F.interpolate(mask, size=rgb.shape[-2:], mode="nearest")

    rgb_f = rgb.to(dtype=torch.float32)
    valid = mask > 0.5
    count = valid.to(dtype=rgb_f.dtype).sum(dim=(2, 3), keepdim=True).clamp(min=1.0)
    fill = (rgb_f * valid.to(dtype=rgb_f.dtype)).sum(dim=(2, 3), keepdim=True) / count
    return torch.where(valid.expand_as(rgb_f), rgb_f, fill)


def build_unik3d_camera_rays(
    rgb_u8: torch.Tensor,
    *,
    device: torch.device,
    intrinsics: torch.Tensor | None = None,
    camera_params: torch.Tensor | None = None,
    camera_model: str | None = None,
    hfov: float | None = None,
    vfov: float | None = None,
) -> tuple[Any, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if rgb_u8.ndim == 3:
        rgb_u8 = rgb_u8.unsqueeze(0)
    if rgb_u8.ndim != 4:
        raise ValueError(f"Expected rgb_u8 shape (3,H,W) or (B,3,H,W), got {tuple(rgb_u8.shape)}")
    bsz, _, h, w = rgb_u8.shape

    _ = _try_import_unik3d()
    from unik3d.utils.camera import BatchCamera, Fisheye624, Pinhole, Spherical  # type: ignore

    if intrinsics is not None:
        if intrinsics.ndim == 2:
            intrinsics = intrinsics.unsqueeze(0)
        if intrinsics.shape != (bsz, 3, 3):
            raise ValueError(
                f"Expected intrinsics shape {(bsz, 3, 3)}, got {tuple(intrinsics.shape)}"
            )
        intrinsics_orig = intrinsics.to(device=device, dtype=torch.float32).clone()
        cameras = [
            BatchCamera.from_camera(Pinhole(K=intrinsics_orig[i : i + 1].clone()))
            for i in range(bsz)
        ]
        camera = torch.cat(cameras, dim=0).to(device)
        rays_k = _pinhole_rays_opencv(intrinsics_orig, h, w, device=device, dtype=torch.float32)
        return camera, rays_k, intrinsics_orig, rays_k

    if camera_params is not None:
        cam_model = str(camera_model or "").lower()
        if cam_model != "fisheye624":
            raise ValueError(
                f"Unsupported camera_model={camera_model!r}; fisheye training now expects OPENCV_FISHEYE/Fisheye624."
            )
        if camera_params.ndim == 1:
            camera_params = camera_params.unsqueeze(0)
        expected_dim = 16
        if camera_params.shape != (bsz, expected_dim):
            raise ValueError(
                f"Expected camera_params shape {(bsz, expected_dim)}, got {tuple(camera_params.shape)}"
            )
        camera_params = camera_params.to(device=device, dtype=torch.float32).clone()
        cameras = [BatchCamera.from_camera(Fisheye624(params=camera_params[i : i + 1].clone())) for i in range(bsz)]
        camera = torch.cat(cameras, dim=0).to(device)
        rays = _fisheye624_rays_opencv_integer_centers(
            Fisheye624,
            camera_params,
            h,
            w,
            device=device,
            dtype=torch.float32,
        )
        return camera, rays, None, rays

    params = torch.tensor(
        [
            1.0,
            1.0,
            (w - 1) / 2.0,
            (h - 1) / 2.0,
            float(w),
            float(h),
            float(hfov if hfov is not None else 2.0 * torch.pi) / 2.0,
            float(vfov if vfov is not None else torch.pi) / 2.0,
        ],
        dtype=torch.float32,
        device=device,
    )
    camera = BatchCamera.from_camera(Spherical(params=params)).to(device)
    rays = _erp_rays_panosplatt3r_opencv(h, w, device=device, dtype=torch.float32)
    rays = rays.unsqueeze(0).expand(bsz, -1, -1, -1).contiguous()
    return camera, rays, None, rays


def _try_import_unik3d() -> Any:
    try:
        import unik3d  # type: ignore

        return unik3d
    except Exception:
        repo_root = Path(__file__).resolve()
        ml_unisharp_root = repo_root.parents[2]
        unik3d_root = ml_unisharp_root / "UniK3D"
        if unik3d_root.exists():
            sys.path.insert(0, str(unik3d_root))
        import unik3d  # type: ignore

        return unik3d


def _get_ml_unisharp_root() -> Path:
    repo_root = Path(__file__).resolve()
    return repo_root.parents[2]


def _setup_unik3d_repo_caches(cache_root: Path | None = None) -> None:
    if cache_root is None:
        cache_root = _get_ml_unisharp_root() / "UniK3D" / "checkpoints"
    hf_cache = cache_root / "huggingface"
    torchhub_cache = cache_root / "torchhub"
    hf_cache.mkdir(parents=True, exist_ok=True)
    torchhub_cache.mkdir(parents=True, exist_ok=True)

    try:
        torch.hub.set_dir(str(torchhub_cache))
    except Exception:
        pass

    import os

    os.environ["HF_HOME"] = str(hf_cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_cache)
    os.environ["HF_HUB_CACHE"] = str(hf_cache)


def load_unik3d_model(
    backbone: str = "vitl",
    pretrained: bool = True,
    device: torch.device | None = None,
    cache_root: Path | None = None,
) -> torch.nn.Module:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _ = _try_import_unik3d()
    _setup_unik3d_repo_caches(cache_root=cache_root)
    try:
        from hubconf import UniK3D as UniK3DHub  # type: ignore

        model = UniK3DHub(backbone=backbone, pretrained=pretrained, config_variant="eval", cache_root=cache_root)
    except Exception as exc:
        if os.environ.get("HF_HUB_OFFLINE", "").strip() in {"1", "ON", "YES", "TRUE"}:
            raise RuntimeError("UniK3D local/offline load failed; refusing to fall back to HuggingFace Hub.") from exc
        from unik3d.models import UniK3D  # type: ignore

        model = UniK3D.from_pretrained(f"lpiccinelli/unik3d-{backbone}")

    model.eval()
    if not hasattr(model, "resolution_level"):
        try:
            setattr(model, "resolution_level", 0)
        except Exception:
            pass
    model.to(device)
    return model


@torch.no_grad()
def infer_unik3d_spherical(
    model: torch.nn.Module,
    rgb_u8: torch.Tensor,
    hfov: float,
    vfov: float,
) -> dict[str, torch.Tensor]:
    if rgb_u8.ndim != 3:
        raise ValueError("Expected rgb_u8 shape (3,H,W).")

    dev = next(model.parameters()).device
    with torch.autocast(device_type=dev.type, enabled=False):
        out = forward_unik3d_spherical(model, rgb_u8, hfov=hfov, vfov=vfov, normalize=True)
    return out


def forward_unik3d_camera_rays(
    model: torch.nn.Module,
    rgb_u8: torch.Tensor,
    *,
    normalize: bool = True,
    validity_mask: torch.Tensor | None = None,
    rays: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if rgb_u8.ndim == 3:
        rgb_u8 = rgb_u8.unsqueeze(0)
    if rgb_u8.ndim != 4:
        raise ValueError(f"Expected rgb_u8 shape (3,H,W) or (B,3,H,W), got {tuple(rgb_u8.shape)}")

    _ = _try_import_unik3d()
    from unik3d.models.unik3d import (  # type: ignore
        IMAGENET_DATASET_MEAN,
        IMAGENET_DATASET_STD,
        _postprocess,
        get_paddings,
        get_resize_factor,
    )
    import torch.nn.functional as F
    import torchvision.transforms.v2.functional as TF

    device = next(model.parameters()).device
    bsz, _, h, w = rgb_u8.shape
    rgb = _fill_invalid_rgb_with_valid_mean(rgb_u8.to(device), validity_mask)

    ratio_bounds = model.shape_constraints["ratio_bounds"]  # type: ignore[attr-defined]
    pixels_bounds = [
        model.shape_constraints["pixels_min"],  # type: ignore[attr-defined]
        model.shape_constraints["pixels_max"],  # type: ignore[attr-defined]
    ]
    if hasattr(model, "resolution_level"):
        pixels_range = pixels_bounds[1] - pixels_bounds[0]
        interval = pixels_range / 10
        new_lowbound = model.resolution_level * interval + pixels_bounds[0]
        new_upbound = (model.resolution_level + 1) * interval + pixels_bounds[0]
        pixels_bounds = (new_lowbound, new_upbound)

    paddings, (padded_h, padded_w) = get_paddings((h, w), ratio_bounds)
    pad_left, pad_right, pad_top, pad_bottom = paddings
    resize_factor, (new_h, new_w) = get_resize_factor((padded_h, padded_w), pixels_bounds)

    if normalize:
        rgb_f = TF.normalize(
            rgb.float() / 255.0,
            mean=IMAGENET_DATASET_MEAN,
            std=IMAGENET_DATASET_STD,
        )
    else:
        rgb_f = rgb.float() / 255.0
    rgb_f = F.pad(rgb_f, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
    rgb_f = F.interpolate(rgb_f, size=(new_h, new_w), mode="bilinear", align_corners=False)

    validity_mask_resized = None
    if torch.is_tensor(validity_mask):
        validity_mask_resized = validity_mask.to(device=device, dtype=torch.float32)
        if validity_mask_resized.ndim == 3:
            validity_mask_resized = validity_mask_resized.unsqueeze(1)
        if int(validity_mask_resized.shape[0]) == 1 and bsz > 1:
            validity_mask_resized = validity_mask_resized.expand(bsz, -1, -1, -1)
        if tuple(validity_mask_resized.shape[-2:]) != (h, w):
            validity_mask_resized = F.interpolate(validity_mask_resized, size=(h, w), mode="nearest")
        validity_mask_resized = F.pad(
            validity_mask_resized,
            (max(0, pad_left), max(0, pad_right), max(0, pad_top), max(0, pad_bottom)),
            value=0.0,
        )
        validity_mask_resized = F.interpolate(validity_mask_resized, size=(new_h, new_w), mode="nearest")

    inputs: dict[str, Any] = {"image": rgb_f}
    if validity_mask_resized is not None:
        inputs["validity_mask"] = validity_mask_resized > 0.5
    rays_resized = None
    if torch.is_tensor(rays):
        rays_resized = rays.to(device=device, dtype=torch.float32)
        if rays_resized.ndim == 3:
            rays_resized = rays_resized.unsqueeze(0)
        if rays_resized.ndim != 4 or int(rays_resized.shape[1]) != 3:
            raise ValueError(f"Expected rays shape (3,H,W) or (B,3,H,W), got {tuple(rays_resized.shape)}")
        if int(rays_resized.shape[0]) == 1 and bsz > 1:
            rays_resized = rays_resized.expand(bsz, -1, -1, -1)
        if tuple(rays_resized.shape[-2:]) != (h, w):
            rays_resized = F.interpolate(rays_resized, size=(h, w), mode="bilinear", align_corners=False)
            rays_resized = rays_resized / torch.norm(rays_resized, dim=1, keepdim=True).clamp(min=1e-5)
        rays_resized = F.pad(
            rays_resized,
            (max(0, pad_left), max(0, pad_right), max(0, pad_top), max(0, pad_bottom)),
            value=0.0,
        )
        rays_resized = F.interpolate(rays_resized, size=(new_h, new_w), mode="bilinear", align_corners=False)
        rays_resized = rays_resized / torch.norm(rays_resized, dim=1, keepdim=True).clamp(min=1e-5)
        inputs["rays"] = rays_resized

    _enable_unik3d_decoder_feature_capture(model)
    _enable_unik3d_force_gt_rays(model)
    pixel_decoder = getattr(model, "pixel_decoder", None)
    old_force = bool(getattr(pixel_decoder, "_unisharp_force_gt_rays", False)) if pixel_decoder is not None else False
    if pixel_decoder is not None:
        pixel_decoder._unisharp_force_gt_rays = torch.is_tensor(rays_resized)
    try:
        _, model_outputs = model.encode_decode(inputs, image_metas={})
    finally:
        if pixel_decoder is not None:
            pixel_decoder._unisharp_force_gt_rays = old_force

    out: dict[str, torch.Tensor] = {}
    out["confidence"] = _postprocess(
        model_outputs["confidence"],
        (padded_h, padded_w),
        paddings=paddings,
        interpolation_mode=model.interpolation_mode,  # type: ignore[attr-defined]
    )
    distance = _postprocess(
        model_outputs["distance"],
        (padded_h, padded_w),
        paddings=paddings,
        interpolation_mode=model.interpolation_mode,  # type: ignore[attr-defined]
    ).clamp(min=1e-4)
    points = _postprocess(
        model_outputs["points"],
        (padded_h, padded_w),
        paddings=paddings,
        interpolation_mode=model.interpolation_mode,  # type: ignore[attr-defined]
    )
    rays_out = _postprocess(
        model_outputs["rays"],
        (padded_h, padded_w),
        paddings=paddings,
        interpolation_mode=model.interpolation_mode,  # type: ignore[attr-defined]
    )
    pred_rays_out = rays_out
    try:
        pred_flat = getattr(model.pixel_decoder, "_unisharp_last_pred_rays_flat", None)  # type: ignore[attr-defined]
        if torch.is_tensor(pred_flat):
            pred_internal = pred_flat.reshape(bsz, new_h, new_w, 3).permute(0, 3, 1, 2).contiguous()
            pred_rays_out = _postprocess(
                pred_internal,
                (padded_h, padded_w),
                paddings=paddings,
                interpolation_mode=model.interpolation_mode,  # type: ignore[attr-defined]
            )
    except Exception:
        pred_rays_out = rays_out
    out["points"] = points
    out["depth"] = points[:, -1:]
    out["distance"] = distance
    out["rays"] = pred_rays_out / torch.norm(pred_rays_out, dim=1, keepdim=True).clamp(min=1e-5)
    out["ray_conditioning_rays"] = rays_out / torch.norm(rays_out, dim=1, keepdim=True).clamp(min=1e-5)
    out["lowres_features"] = model_outputs.get("lowres_features", None)
    out["_unisharp_internal_rays"] = model_outputs["rays"]
    out["_unisharp_postprocess_padded_hw"] = (int(padded_h), int(padded_w))
    out["_unisharp_postprocess_paddings"] = tuple(int(x) for x in paddings)
    out["_unisharp_postprocess_interpolation_mode"] = str(model.interpolation_mode)  # type: ignore[attr-defined]

    try:
        rm = model.pixel_decoder.radial_module  # type: ignore[attr-defined]
        init_latents = getattr(rm, "_unisharp_last_init_latents", None)
        out_feats = getattr(rm, "_unisharp_last_out_features", None)
        if init_latents is not None and out_feats is not None:
            out["pyramid_features"] = [init_latents, *list(out_feats)]
    except Exception:
        pass
    return out


def forward_unik3d_spherical(
    model: torch.nn.Module,
    rgb_u8: torch.Tensor,
    hfov: float,
    vfov: float,
    normalize: bool = True,
) -> dict[str, torch.Tensor]:
    del hfov, vfov
    return forward_unik3d_camera_rays(
        model,
        rgb_u8,
        normalize=normalize,
    )


def forward_unik3d_pinhole(
    model: torch.nn.Module,
    rgb_u8: torch.Tensor,
    intrinsics: torch.Tensor,
    normalize: bool = True,
) -> dict[str, torch.Tensor]:
    device = next(model.parameters()).device
    _, rays, _, _ = build_unik3d_camera_rays(
        rgb_u8,
        device=device,
        intrinsics=intrinsics.to(device=device, dtype=torch.float32),
    )
    return forward_unik3d_camera_rays(
        model,
        rgb_u8,
        normalize=normalize,
        rays=rays,
    )


def forward_unik3d_fisheye624(
    model: torch.nn.Module,
    rgb_u8: torch.Tensor,
    camera_params: torch.Tensor,
    normalize: bool = True,
    validity_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    device = next(model.parameters()).device
    _, rays, _, _ = build_unik3d_camera_rays(
        rgb_u8,
        device=device,
        camera_params=camera_params.to(device=device, dtype=torch.float32),
        camera_model="fisheye624",
    )
    return forward_unik3d_camera_rays(
        model,
        rgb_u8,
        normalize=normalize,
        validity_mask=validity_mask,
        rays=rays,
    )


@torch.no_grad()
def infer_unik3d_pinhole(
    model: torch.nn.Module,
    rgb_u8: torch.Tensor,
    intrinsics: torch.Tensor,
) -> dict[str, torch.Tensor]:
    dev = next(model.parameters()).device
    with torch.autocast(device_type=dev.type, enabled=False):
        out = forward_unik3d_pinhole(model, rgb_u8, intrinsics=intrinsics, normalize=True)
    return out


from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from unisharp.models.unik3d_feature_extractor import UniK3DFeatureExtractor
from unisharp.models.gaussian_initializer import PanoInitializer
from unisharp.models.heads import (
    DirectPredictionHead,
)
from unisharp.models.gaussian_composer import PanoGaussianComposer
from unisharp.models.unisharp_params import PanoPredictorParams
from unisharp.models.feature_gaussian_decoder import ImageFeatures
from unisharp import DEFAULT_MAX_DEPTH_M
from unisharp.utils.gaussians import Gaussians3D


@dataclass
class UnisharpFeatureConfig:
    
    unik3d_backbone: str = "vitl"
    unik3d_resolution_level: int = 0
    
    initializer_stride: int = 1
    initializer_scale_factor: float = 1.5
    max_distance_m: float = DEFAULT_MAX_DEPTH_M
    detach_init_layer0_distance: bool = True
    delta_rho_limit: float = 2.0


class UniK3DCopiedDepthHead(nn.Module):

    def __init__(self, radial_module: nn.Module, *, out_channels: int = 1) -> None:
        super().__init__()
        self.out_channels = int(out_channels)
        if self.out_channels < 1:
            raise ValueError(f"out_channels must be >= 1, got {self.out_channels}")

        self.depth_mlp = copy.deepcopy(radial_module.depth_mlp)  # type: ignore[attr-defined]
        self.to_depth_lr = copy.deepcopy(radial_module.to_depth_lr)  # type: ignore[attr-defined]
        self.to_depth_hr = copy.deepcopy(radial_module.to_depth_hr)  # type: ignore[attr-defined]
        self._set_last_depth_conv_channels(self.out_channels)

        for p in self.parameters():
            p.requires_grad_(True)

    def _set_last_depth_conv_channels(self, out_channels: int) -> None:
        if not isinstance(self.to_depth_hr, nn.Sequential) or len(self.to_depth_hr) == 0:
            raise TypeError("Expected UniK3D radial_module.to_depth_hr to be a non-empty nn.Sequential.")
        last = self.to_depth_hr[-1]
        if not isinstance(last, nn.Conv2d):
            raise TypeError(f"Expected final UniK3D depth head layer to be Conv2d, got {type(last)!r}.")
        if int(last.out_channels) == int(out_channels):
            return

        new_last = nn.Conv2d(
            in_channels=int(last.in_channels),
            out_channels=int(out_channels),
            kernel_size=last.kernel_size,
            stride=last.stride,
            padding=last.padding,
            dilation=last.dilation,
            groups=int(last.groups),
            bias=last.bias is not None,
            padding_mode=last.padding_mode,
        )
        with torch.no_grad():
            if int(last.out_channels) == 1:
                new_last.weight.copy_(last.weight.repeat(int(out_channels), 1, 1, 1))
                if last.bias is not None and new_last.bias is not None:
                    new_last.bias.copy_(last.bias.repeat(int(out_channels)))
            elif int(last.out_channels) >= int(out_channels):
                new_last.weight.copy_(last.weight[: int(out_channels)])
                if last.bias is not None and new_last.bias is not None:
                    new_last.bias.copy_(last.bias[: int(out_channels)])
            else:
                repeat = (int(out_channels) + int(last.out_channels) - 1) // int(last.out_channels)
                new_last.weight.copy_(last.weight.repeat(repeat, 1, 1, 1)[: int(out_channels)])
                if last.bias is not None and new_last.bias is not None:
                    new_last.bias.copy_(last.bias.repeat(repeat)[: int(out_channels)])
        self.to_depth_hr[-1] = new_last

    def _radial_out_features(self, features_3d_pyramid: list[torch.Tensor]) -> list[torch.Tensor]:
        expected = len(self.depth_mlp)
        if len(features_3d_pyramid) == expected + 1:
            return list(features_3d_pyramid[1:])
        if len(features_3d_pyramid) == expected:
            return list(features_3d_pyramid)
        raise RuntimeError(
            f"Expected {expected} UniK3D radial out features (or {expected + 1} including init_latents), "
            f"got {len(features_3d_pyramid)}."
        )

    def forward(
        self,
        features_3d_pyramid: list[torch.Tensor],
        *,
        internal_hw: tuple[int, int],
    ) -> torch.Tensor:
        out_features = self._radial_out_features(features_3d_pyramid)
        h_out, w_out = int(out_features[-1].shape[-2]), int(out_features[-1].shape[-1])
        out_depth_features: torch.Tensor | None = None
        for i, (layer, features) in enumerate(zip(self.depth_mlp, out_features, strict=False)):
            out_depth_features = layer(features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            if i < len(self.depth_mlp) - 1:
                continue
        if out_depth_features is None:
            raise RuntimeError("UniK3D copied depth head received no radial features.")

        out_depth_features = F.interpolate(
            out_depth_features,
            size=(h_out, w_out),
            mode="bilinear",
            align_corners=True,
        )
        logradius = self.to_depth_lr(out_depth_features)
        logradius = F.interpolate(
            logradius,
            size=(int(internal_hw[0]), int(internal_hw[1])),
            mode="bilinear",
            align_corners=True,
        )
        return self.to_depth_hr(logradius)


class UnisharpFeatureModel(nn.Module):
    
    def __init__(self, config: UnisharpFeatureConfig):
        super().__init__()
        self.config = config
        
        from unisharp.utils.unik3d_adapter import load_unik3d_model
        unik3d_model = load_unik3d_model(
            backbone=config.unik3d_backbone,
            pretrained=True,
            device="cpu",
        )
        if hasattr(unik3d_model, "resolution_level"):
            unik3d_model.resolution_level = int(max(0, min(9, config.unik3d_resolution_level)))
        
        from unisharp.models.feature_gaussian_decoder import FeatureGaussianDecoderParams, create_feature_gaussian_decoder
        if config.unik3d_backbone == "vitl":
            dino_feature_dim = 1024
            decoder_params = FeatureGaussianDecoderParams(
                dims_3d_in=(128, 256, 512, 512),
                dims_3d_out=(256, 512, 1024, 1024),
                dim_2d_in=1024,
                dim_2d_out=256,
                dim_decoder_out=256,
                dim_texture_out=32,
                dim_geometry_out=32,
                stride_out=int(max(1, config.initializer_stride)),
            )
        else:
            dino_feature_dim = 768
            decoder_params = FeatureGaussianDecoderParams(
                dims_3d_in=(96, 192, 384, 384),
                dims_3d_out=(256, 512, 768, 768),
                dim_2d_in=768,
                dim_2d_out=256,
                dim_decoder_out=256,
                dim_texture_out=32,
                dim_geometry_out=32,
                stride_out=int(max(1, config.initializer_stride)),
            )

        self.feature_extractor = UniK3DFeatureExtractor(
            unik3d_model=unik3d_model,
            dino_feature_dim=dino_feature_dim,
        )
        
        self.decoder_params = decoder_params
        self.feature_decoder = create_feature_gaussian_decoder(decoder_params)
        
        params = PanoPredictorParams()
        params.initializer.stride = config.initializer_stride
        params.initializer.scale_factor = float(config.initializer_scale_factor)
        params.num_monodepth_layers = 2
        params.initializer.num_layers = 2
        
        self.init_model = PanoInitializer(params.initializer)
        self.prediction_head = DirectPredictionHead(
            feature_dim=32,
            num_layers=params.initializer.num_layers,
        )
        decoder_stride = int(getattr(self.feature_decoder, "stride", 1))
        init_stride = int(max(1, config.initializer_stride))
        if decoder_stride != init_stride:
            raise ValueError(
                "Feature decoder stride must match initializer stride so base/features/head/delta "
                "share one Gaussian grid, "
                f"got decoder_stride={decoder_stride}, initializer_stride={init_stride}"
            )
        self.gaussian_composer = PanoGaussianComposer(
            delta_factor=params.delta_factor,
            min_scale=params.min_scale,
            max_scale=params.max_scale,
            color_activation_type=params.color_activation_type,
            opacity_activation_type=params.opacity_activation_type,
            color_space="linearRGB",
            base_scale_on_predicted_mean=params.base_scale_on_predicted_mean,
            scale_factor=decoder_stride // init_stride,
            delta_rho_limit=float(getattr(config, "delta_rho_limit", 2.0)),
        )

        radial_module = self.feature_extractor.unik3d.pixel_decoder.radial_module  # type: ignore[attr-defined]
        self.second_layer_depth_head = UniK3DCopiedDepthHead(
            radial_module,
            out_channels=1,
        )

        self.params = params

    def train(self, mode: bool = True) -> "UnisharpFeatureModel":
        super().train(mode)
        return self

    def _set_initializer_circular_mode(self, circular_horizontal: bool) -> None:
        self.init_model.params.circular_horizontal = bool(circular_horizontal)

    def _initializer_grid_cell_size_override(
        self,
        *,
        camera_intrinsics: torch.Tensor | None,
        camera_params: torch.Tensor | None,
        image: torch.Tensor,
        is_spherical: bool,
    ) -> torch.Tensor | None:
        if bool(is_spherical):
            return None
        stride = float(max(1, int(self.init_model.params.stride)))
        if torch.is_tensor(camera_intrinsics):
            k = camera_intrinsics.to(device=image.device, dtype=torch.float32)
            if k.ndim == 2:
                k = k.unsqueeze(0)
            fx = k[:, 0, 0].clamp(min=1.0)
            fy = k[:, 1, 1].clamp(min=1.0)
            return torch.stack([stride / fx, stride / fy], dim=1)
        if torch.is_tensor(camera_params):
            params = camera_params.to(device=image.device, dtype=torch.float32)
            if params.ndim == 1:
                params = params.unsqueeze(0)
            if int(params.shape[-1]) == 15:
                fx = fy = params[:, 0].clamp(min=1.0)
            else:
                fx = params[:, 0].clamp(min=1.0)
                fy = params[:, 1].clamp(min=1.0)
            return torch.stack([stride / fx, stride / fy], dim=1)
        return None

    def _predict_delta(
        self,
        image_features: ImageFeatures,
    ) -> torch.Tensor:
        return self.prediction_head(image_features)

    @staticmethod
    def _strip_module_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for k, v in state.items():
            if isinstance(k, str) and k.startswith("module."):
                out[k[len("module.") :]] = v
            else:
                out[k] = v
        return out

    @staticmethod
    def _distance_layers_from_unik3d_logradius(
        *,
        logradius_layers: torch.Tensor,
        internal_rays: torch.Tensor | None,
        final_rays: torch.Tensor,
        unik3d_output: dict[str, Any],
    ) -> torch.Tensor:
        radius_layers = torch.exp(logradius_layers.clamp(min=-8.0, max=8.0) + 2.0)
        if internal_rays is None:
            return F.interpolate(
                radius_layers,
                size=final_rays.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).clamp(min=1e-4)

        padded_hw = unik3d_output.get("_unisharp_postprocess_padded_hw", None)
        paddings = unik3d_output.get("_unisharp_postprocess_paddings", None)
        interpolation_mode = unik3d_output.get("_unisharp_postprocess_interpolation_mode", None)
        if padded_hw is None or paddings is None or interpolation_mode is None:
            return F.interpolate(
                radius_layers,
                size=final_rays.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).clamp(min=1e-4)

        from unisharp.utils.unik3d_adapter import postprocess_unik3d_tensor

        bsz, num_layers, h_int, w_int = radius_layers.shape
        radius_post = postprocess_unik3d_tensor(
            radius_layers.reshape(bsz * num_layers, 1, h_int, w_int),
            padded_hw=tuple(int(x) for x in padded_hw),
            paddings=tuple(int(x) for x in paddings),
            interpolation_mode=str(interpolation_mode),
        )
        h_out, w_out = int(radius_post.shape[-2]), int(radius_post.shape[-1])
        return radius_post.reshape(bsz, num_layers, h_out, w_out).clamp(min=1e-4)

    @staticmethod
    def _compute_spherical_distortion_drop_prob_map(
        *,
        batch_size: int,
        h: int,
        w: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        yy = (torch.arange(h, device=device, dtype=dtype) + 0.5) / float(max(1, h))
        lat = (yy - 0.5) * torch.pi
        row_keep = torch.cos(lat).clamp(min=0.0)
        row_keep = row_keep / row_keep.max().clamp(min=1e-6)
        row_drop = 1.0 - row_keep
        drop_2d = row_drop[:, None].expand(h, w)
        return drop_2d.unsqueeze(0).expand(int(batch_size), -1, -1).contiguous()

    @staticmethod
    def _sample_dropout_mask(
        drop_prob_map: torch.Tensor,
        *,
        num_layers: int,
    ) -> torch.Tensor:
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        b, h, w = drop_prob_map.shape
        p = torch.zeros((b, 1, int(num_layers), h, w), device=drop_prob_map.device, dtype=drop_prob_map.dtype)
        if int(num_layers) > 1:
            p[:, :, 1] = drop_prob_map[:, None]
        rnd = torch.rand_like(p)
        return (rnd < p).to(dtype=drop_prob_map.dtype)

    @staticmethod
    def _apply_dropout_to_gaussians(
        gaussians: Gaussians3D,
        dropout_mask: torch.Tensor | None,
    ) -> Gaussians3D:
        if dropout_mask is None:
            return gaussians
        if dropout_mask.ndim != 5:
            raise ValueError(f"Expected dropout mask shape [B,1,L,H,W], got {tuple(dropout_mask.shape)}")
        mask_flat = dropout_mask[:, 0].flatten(1).to(
            device=gaussians.opacities.device,
            dtype=gaussians.opacities.dtype,
        ).clamp(0.0, 1.0)
        if mask_flat.shape != gaussians.opacities.shape:
            raise ValueError(
                "Dropout mask must match flattened Gaussian opacity shape, "
                f"got {tuple(mask_flat.shape)} vs {tuple(gaussians.opacities.shape)}"
            )
        return Gaussians3D(
            mean_vectors=gaussians.mean_vectors,
            singular_values=gaussians.singular_values,
            quaternions=gaussians.quaternions,
            colors=gaussians.colors,
            opacities=gaussians.opacities * (1.0 - mask_flat),
        )

    @staticmethod
    def _apply_flat_opacity_dropout(
        gaussians: Gaussians3D,
        dropout_mask_flat: torch.Tensor | None,
    ) -> Gaussians3D:
        if dropout_mask_flat is None:
            return gaussians
        mask = dropout_mask_flat.to(device=gaussians.opacities.device, dtype=gaussians.opacities.dtype).clamp(0.0, 1.0)
        if tuple(mask.shape) != tuple(gaussians.opacities.shape):
            raise ValueError(
                "Flat dropout mask must match Gaussian opacity shape, "
                f"got {tuple(mask.shape)} vs {tuple(gaussians.opacities.shape)}"
            )
        return Gaussians3D(
            mean_vectors=gaussians.mean_vectors,
            singular_values=gaussians.singular_values,
            quaternions=gaussians.quaternions,
            colors=gaussians.colors,
            opacities=gaussians.opacities * (1.0 - mask),
        )

    def _build_spherical_dropout_prob_map(
        self,
        base_values: Any,
        *,
        is_spherical: bool,
    ) -> torch.Tensor | None:
        rays = base_values.rays
        if rays.ndim != 5 or rays.shape[2] <= 1:
            return None
        if not bool(is_spherical):
            return None
        if not bool(self.training):
            return None
        with torch.no_grad():
            bsz = int(base_values.rays.shape[0])
            h = int(base_values.rays.shape[-2])
            w = int(base_values.rays.shape[-1])
            drop_prob_map = self._compute_spherical_distortion_drop_prob_map(
                batch_size=bsz,
                h=h,
                w=w,
                device=base_values.rays.device,
                dtype=base_values.rays.dtype,
            )
        return drop_prob_map

    @staticmethod
    def _strip_known_prefixes_for_unik3d(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for k, v in state.items():
            kk = str(k)
            if kk.startswith("module."):
                kk = kk[len("module.") :]
            if kk.startswith("unik3d."):
                kk = kk[len("unik3d.") :]
            out[kk] = v
        return out

    @staticmethod
    def _looks_like_unik3d_state_dict(state: dict[str, torch.Tensor]) -> bool:
        if len(state) == 0:
            return False
        probe = UnisharpFeatureModel._strip_known_prefixes_for_unik3d(state)
        prefixes = ("pixel_encoder.", "pixel_decoder.", "head.")
        return any(any(str(k).startswith(p) for p in prefixes) for k in probe.keys())

    def _load_unik3d_state_dict(self, state: dict[str, torch.Tensor]) -> tuple[list[str], list[str]]:
        state_norm = self._strip_known_prefixes_for_unik3d(state)
        incompatible = self.feature_extractor.unik3d.load_state_dict(state_norm, strict=False)
        missing = [f"feature_extractor.unik3d.{k}" for k in list(getattr(incompatible, "missing_keys", []))]
        unexpected = [f"feature_extractor.unik3d.{k}" for k in list(getattr(incompatible, "unexpected_keys", []))]
        return missing, unexpected

    @staticmethod
    def _center_embed_or_crop_kernel(src: torch.Tensor, dst_shape: torch.Size) -> torch.Tensor:
        out = src.new_zeros(tuple(dst_shape))
        src_h, src_w = int(src.shape[-2]), int(src.shape[-1])
        dst_h, dst_w = int(dst_shape[-2]), int(dst_shape[-1])
        copy_h = min(src_h, dst_h)
        copy_w = min(src_w, dst_w)
        src_y0 = max(0, (src_h - copy_h) // 2)
        src_x0 = max(0, (src_w - copy_w) // 2)
        dst_y0 = max(0, (dst_h - copy_h) // 2)
        dst_x0 = max(0, (dst_w - copy_w) // 2)
        out[..., dst_y0 : dst_y0 + copy_h, dst_x0 : dst_x0 + copy_w] = src[
            ..., src_y0 : src_y0 + copy_h, src_x0 : src_x0 + copy_w
        ]
        return out

    @classmethod
    def _load_module_state_shape_compat(
        cls,
        module: nn.Module,
        state: dict[str, torch.Tensor],
    ) -> tuple[list[str], list[str]]:
        current = module.state_dict()
        filtered: dict[str, torch.Tensor] = {}
        mismatched: list[str] = []
        migrated: list[str] = []
        for key, value in state.items():
            dst = current.get(key)
            if not torch.is_tensor(value) or not torch.is_tensor(dst):
                continue
            if tuple(value.shape) == tuple(dst.shape):
                filtered[key] = value.to(dtype=dst.dtype)
                continue
            if (
                str(key).endswith(".deconv.weight")
                and value.ndim == 4
                and dst.ndim == 4
                and tuple(value.shape[:2]) == tuple(dst.shape[:2])
            ):
                filtered[key] = cls._center_embed_or_crop_kernel(value, dst.shape).to(dtype=dst.dtype)
                migrated.append(f"{key}:shape{tuple(value.shape)}->{tuple(dst.shape)}")
                continue
            mismatched.append(f"{key}:shape{tuple(value.shape)}->{tuple(dst.shape)}")
        incompatible = module.load_state_dict(filtered, strict=False)
        missing = list(getattr(incompatible, "missing_keys", []))
        unexpected = list(getattr(incompatible, "unexpected_keys", [])) + mismatched
        unexpected.extend([f"migrated:{item}" for item in migrated])
        return missing, unexpected

    def _load_prediction_head_compat(
        self,
        state: dict[str, torch.Tensor],
    ) -> tuple[list[str], list[str], int]:
        current = self.prediction_head.state_dict()
        filtered: dict[str, torch.Tensor] = {}
        shape_mismatch: list[str] = []
        for key, value in state.items():
            dst = current.get(key)
            if not torch.is_tensor(value) or not torch.is_tensor(dst):
                continue
            if tuple(value.shape) != tuple(dst.shape):
                shape_mismatch.append(str(key))
                continue
            filtered[key] = value.to(dtype=dst.dtype)
        incompatible = self.prediction_head.load_state_dict(filtered, strict=False)
        missing = list(getattr(incompatible, "missing_keys", []))
        unexpected = list(getattr(incompatible, "unexpected_keys", [])) + shape_mismatch
        copied_legacy = 0
        if len(state) > 0 and len(missing) > 0:
            copied_legacy = self.prediction_head.init_from_legacy_direct_state(state)
            if copied_legacy > 0:
                missing = [
                    k
                    for k in missing
                    if k
                    not in (
                        "geometry_prediction_head.weight",
                        "geometry_prediction_head.bias",
                        "texture_prediction_head.weight",
                        "texture_prediction_head.bias",
                    )
                ]
                unexpected = [
                    k
                    for k in unexpected
                    if k
                    not in (
                        "geometry_weight",
                        "geometry_bias",
                        "texture_weight",
                        "texture_bias",
                        "geometry_prediction_head.weight",
                        "geometry_prediction_head.bias",
                        "texture_prediction_head.weight",
                        "texture_prediction_head.bias",
                        "geo_fc2.weight",
                        "geo_fc2.bias",
                        "tex_fc2.weight",
                        "tex_fc2.bias",
                    )
                ]
        return missing, unexpected, copied_legacy

    def _load_depth_head_compat(self, state: dict[str, torch.Tensor]) -> tuple[list[str], list[str]]:
        state = dict(state)
        current = self.second_layer_depth_head.state_dict()
        filtered: dict[str, torch.Tensor] = {}
        unexpected: list[str] = []
        for key, value in state.items():
            dst = current.get(key)
            if not torch.is_tensor(value) or not torch.is_tensor(dst):
                unexpected.append(str(key))
                continue
            if tuple(value.shape) != tuple(dst.shape):
                unexpected.append(str(key))
                continue
            filtered[key] = value.to(dtype=dst.dtype)
        incompatible = self.second_layer_depth_head.load_state_dict(filtered, strict=False)
        missing = list(getattr(incompatible, "missing_keys", []))
        unexpected.extend(list(getattr(incompatible, "unexpected_keys", [])))
        return missing, unexpected

    def forward(
        self,
        image: torch.Tensor,
        image_u8: torch.Tensor | None = None,
        camera_intrinsics: torch.Tensor | None = None,
        camera_params: torch.Tensor | None = None,
        camera_model: str | None = None,
        depth_gt: torch.Tensor | None = None,
        distance_init_cap_m: float | None = None,
        validity_mask: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> dict[str, Any] | Any:
        _, _, H, W = image.shape

        import numpy as np
        camera_model_name_input = str(camera_model or "").strip().lower()
        if (
            camera_model_name_input == ""
            and camera_intrinsics is None
            and camera_params is None
            and int(W) == 2 * int(H)
        ):
            camera_model = "spherical"
        if image_u8 is None:
            image_u8 = (image * 255.0).round().clamp(0, 255).to(torch.uint8)
        features_2d, features_3d_pyramid = self.feature_extractor.forward(
            rgb_u8=image_u8,
            target_h=H,
            target_w=W,
            intrinsics=camera_intrinsics,
            camera_params=camera_params,
            camera_model=camera_model,
            hfov=float(2.0 * np.pi),
            vfov=float(np.pi),
            validity_mask=validity_mask,
            use_predicted_rays=False,
        )
        unik3d_output = self.feature_extractor._unisharp_last_unik3d_output

        if unik3d_output is None:
            raise RuntimeError("Missing cached UniK3D output from feature_extractor forward.")

        rays = unik3d_output["rays"]
        distance = unik3d_output["distance"]
        try:
            from unisharp.utils.unik3d_adapter import build_unik3d_camera_rays

            _, gt_rays, _, _ = build_unik3d_camera_rays(
                image_u8,
                device=rays.device,
                intrinsics=camera_intrinsics,
                camera_params=camera_params,
                camera_model=camera_model,
                hfov=float(2.0 * np.pi),
                vfov=float(np.pi),
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to build calibrated camera rays required by the fixed gt-override geometry path."
            ) from exc
        geometry_rays = gt_rays.to(device=rays.device, dtype=rays.dtype)
        if tuple(geometry_rays.shape[-2:]) != tuple(rays.shape[-2:]):
            geometry_rays = F.interpolate(geometry_rays, size=rays.shape[-2:], mode="bilinear", align_corners=False)
            geometry_rays = geometry_rays / torch.norm(geometry_rays, dim=1, keepdim=True).clamp(min=1e-5)

        camera_model_name = str(camera_model or "").strip().lower()
        render_geometry_rays = geometry_rays.detach()

        is_spherical = bool(
            (camera_intrinsics is None)
            and (camera_params is None)
            and camera_model_name in {"spherical", "erp", "panorama"}
        )

        internal_rays_raw = unik3d_output.get("_unisharp_internal_rays", None)
        internal_rays = internal_rays_raw if torch.is_tensor(internal_rays_raw) else None
        internal_hw = (
            (int(internal_rays.shape[-2]), int(internal_rays.shape[-1]))
            if internal_rays is not None
            else (int(image.shape[-2]), int(image.shape[-1]))
        )
        logradius_extra = self.second_layer_depth_head(
            features_3d_pyramid,
            internal_hw=internal_hw,
        )
        if int(logradius_extra.shape[1]) != 1:
            raise RuntimeError(
                f"copied UniK3D depth head output channels ({int(logradius_extra.shape[1])}) must be 1"
            )
        extra_distance_layers = self._distance_layers_from_unik3d_logradius(
            logradius_layers=logradius_extra,
            internal_rays=internal_rays.detach() if torch.is_tensor(internal_rays) else None,
            final_rays=render_geometry_rays,
            unik3d_output=unik3d_output,
        )
        distance_layers = torch.cat([distance.clamp(min=1e-4), extra_distance_layers], dim=1)
        distance_ray_align_factor = None

        max_distance_m = float(getattr(self.config, "max_distance_m", DEFAULT_MAX_DEPTH_M))
        finite_cap_m = max_distance_m if max_distance_m > 0.0 else DEFAULT_MAX_DEPTH_M
        distance_layers = torch.nan_to_num(
            distance_layers,
            nan=finite_cap_m,
            posinf=finite_cap_m,
            neginf=1e-4,
        ).clamp(min=1e-4)

        circular_horizontal = bool(is_spherical)
        self._set_initializer_circular_mode(circular_horizontal=circular_horizontal)

        init_cap_m = float(distance_init_cap_m) if distance_init_cap_m is not None else max_distance_m
        if max_distance_m > 0.0 and init_cap_m > 0.0:
            init_cap_m = min(max_distance_m, init_cap_m)
        elif max_distance_m > 0.0:
            init_cap_m = max_distance_m
        distance_layers_for_supervision = distance_layers
        init_distance_layers = distance_layers.clamp(min=1e-4)
        if init_cap_m > 0.0:
            init_distance_layers = init_distance_layers.clamp(max=init_cap_m)

        init_rays = render_geometry_rays
        init_layer0_distance = init_distance_layers[:, 0:1]
        if bool(getattr(self.config, "detach_init_layer0_distance", True)):
            init_layer0_distance = init_layer0_distance.detach()
        init_distance_layers = torch.cat(
            [
                init_layer0_distance,
                init_distance_layers[:, 1:],
            ],
            dim=1,
        )
        
        feat_2d = features_2d
        feat_3d = features_3d_pyramid

        scale_cell_intrinsics = camera_intrinsics
        scale_cell_camera_params = camera_params
        scale_cell_rays = None

        init_output = self.init_model(
            image=image,
            rays=init_rays,
            distance=init_distance_layers,
            angular_cell_rays=scale_cell_rays,
            grid_cell_size_override=self._initializer_grid_cell_size_override(
                camera_intrinsics=scale_cell_intrinsics,
                camera_params=scale_cell_camera_params,
                image=image,
                is_spherical=bool(is_spherical),
            ),
            target_hw=None,
        )
        base_values = init_output.gaussian_base_values

        base_hw = (int(base_values.rays.shape[-2]), int(base_values.rays.shape[-1]))

        decoded_features = self.feature_decoder(
            feat_2d,
            feat_3d,
            circular_horizontal=bool(is_spherical),
            target_hw=base_hw,
        )
        feature_hw = (
            int(decoded_features.texture_features.shape[-2]),
            int(decoded_features.texture_features.shape[-1]),
        )
        if tuple(decoded_features.geometry_features.shape[-2:]) != feature_hw:
            raise RuntimeError(
                "Texture and geometry feature grids must match, "
                f"got texture={feature_hw} geometry={tuple(decoded_features.geometry_features.shape[-2:])}"
            )
        if feature_hw != base_hw:
            raise RuntimeError(
                "Decoded feature grid must match initializer Gaussian grid, "
                f"got features={feature_hw} base={base_hw}"
            )

        dropout_prob_map: torch.Tensor | None = None
        dropout_mask: torch.Tensor | None = None
        dropout_prob_map = self._build_spherical_dropout_prob_map(
            base_values,
            is_spherical=bool(is_spherical),
        )
        if bool(is_spherical) and (dropout_prob_map is not None):
            dropout_mask = self._sample_dropout_mask(
                dropout_prob_map,
                num_layers=int(base_values.rays.shape[2]),
            )

        image_features = decoded_features

        delta = self._predict_delta(image_features)

        gaussians = self.gaussian_composer(
            delta=delta,
            base_values=base_values,
            global_scale=init_output.global_scale,
            flatten_output=True,
        )
        gaussians = self._apply_dropout_to_gaussians(gaussians, dropout_mask)
        
        if not return_aux:
            return gaussians
        else:
            return {
                "gaussians": gaussians,
                "gaussian_base_values": init_output.gaussian_base_values,
                "gaussian_base_values_for_composer": base_values,
                "delta": delta,
                "delta_rho_applied": self.gaussian_composer.apply_delta_rho(delta[:, 2:3]),
                "scale_factor_applied": self.gaussian_composer.apply_scale_factor(
                    self.gaussian_composer._smooth_scale_delta(
                        delta[:, 3:6],
                        circular_horizontal=bool(is_spherical),
                    )
                ),
                "unik3d_rays": rays,
                "unik3d_ray_conditioning_rays": unik3d_output.get("ray_conditioning_rays", None),
                "unik3d_gt_rays": gt_rays,
                "geometry_rays": geometry_rays,
                "initializer_geometry_rays": render_geometry_rays,
                "detach_init_layer0_distance": bool(getattr(self.config, "detach_init_layer0_distance", True)),
                "unik3d_distance": distance,
                "distance_ray_align_factor": distance_ray_align_factor,
                "distance_layers": distance_layers_for_supervision,
                "init_distance_layers": init_distance_layers,
                "decoded_image_features": image_features,
                "gaussian_dropout_mask": dropout_mask,
                "unik3d_features_2d": feat_2d,
                "unik3d_features_3d": feat_3d,
                "initializer_output": init_output,
                "second_layer_dropout_prob_map": dropout_prob_map,
                "second_layer_dropout_mask": dropout_mask,
            }
    
    def get_trainable_parameters(self) -> list[torch.nn.Parameter]:
        params = []
        
        params.extend([p for p in self.feature_decoder.parameters() if p.requires_grad])
        
        params.extend([p for p in self.init_model.parameters() if p.requires_grad])
        params.extend([p for p in self.prediction_head.parameters() if p.requires_grad])
        params.extend([p for p in self.gaussian_composer.parameters() if p.requires_grad])
        params.extend([p for p in self.second_layer_depth_head.parameters() if p.requires_grad])
        
        for _, p in self.feature_extractor.named_parameters():
            if not p.requires_grad:
                continue
            params.append(p)
        
        return params
    
    def load_from_checkpoint(self, ckpt_path: str, strict: bool = False) -> tuple[list[str], list[str]]:
        try:
            payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(ckpt_path, map_location="cpu")

        def _finish(result: tuple[list[str], list[str]]) -> tuple[list[str], list[str]]:
            return result
        
        if isinstance(payload, dict):
            if "feature_extractor" in payload:
                missing_keys: list[str] = []
                unexpected_keys: list[str] = []

                def _merge_incompatible(prefix: str, incompatible: Any) -> None:
                    missing = list(getattr(incompatible, "missing_keys", []))
                    unexpected = list(getattr(incompatible, "unexpected_keys", []))
                    missing_keys.extend([f"{prefix}.{k}" for k in missing])
                    unexpected_keys.extend([f"{prefix}.{k}" for k in unexpected])

                def _missing_all(prefix: str, module: nn.Module) -> None:
                    missing_keys.extend([f"{prefix}.{k}" for k in module.state_dict().keys()])

                def _load_module_compat(
                    prefix: str,
                    module: nn.Module,
                    state: dict[str, torch.Tensor],
                    *,
                    resize_spatial_kernels: bool = False,
                ) -> None:
                    target_state = module.state_dict()
                    filtered: dict[str, torch.Tensor] = {}
                    for key, value in state.items():
                        target = target_state.get(key)
                        if target is None:
                            unexpected_keys.append(f"{prefix}.{key}")
                            continue
                        if tuple(value.shape) == tuple(target.shape):
                            filtered[key] = value
                            continue
                        can_resize = (
                            bool(resize_spatial_kernels)
                            and isinstance(value, torch.Tensor)
                            and isinstance(target, torch.Tensor)
                            and value.ndim == 4
                            and target.ndim == 4
                            and tuple(value.shape[:2]) == tuple(target.shape[:2])
                        )
                        if can_resize:
                            resized = F.interpolate(
                                value.to(dtype=torch.float32),
                                size=tuple(int(x) for x in target.shape[-2:]),
                                mode="bilinear",
                                align_corners=False,
                            ).to(dtype=target.dtype)
                            filtered[key] = resized
                        else:
                            unexpected_keys.append(f"{prefix}.{key}")
                    _merge_incompatible(prefix, module.load_state_dict(filtered, strict=False))

                if "feature_extractor" in payload and isinstance(payload["feature_extractor"], dict):
                    state = self._strip_module_prefix(payload["feature_extractor"])
                    _merge_incompatible(
                        "feature_extractor",
                        self.feature_extractor.load_state_dict(state, strict=False),
                    )
                else:
                    _missing_all("feature_extractor", self.feature_extractor)

                if "feature_decoder" in payload and isinstance(payload["feature_decoder"], dict):
                    state = self._strip_module_prefix(payload["feature_decoder"])
                    _load_module_compat(
                        "feature_decoder",
                        self.feature_decoder,
                        state,
                        resize_spatial_kernels=True,
                    )
                else:
                    _missing_all("feature_decoder", self.feature_decoder)

                if "init_model" in payload and isinstance(payload["init_model"], dict):
                    state = self._strip_module_prefix(payload["init_model"])
                    _merge_incompatible(
                        "init_model",
                        self.init_model.load_state_dict(state, strict=False),
                    )
                else:
                    _missing_all("init_model", self.init_model)

                if "prediction_head" in payload and isinstance(payload["prediction_head"], dict):
                    state = self._strip_module_prefix(payload["prediction_head"])
                    miss, unexp, _ = self._load_prediction_head_compat(state)
                    missing_keys.extend([f"prediction_head.{k}" for k in miss])
                    unexpected_keys.extend([f"prediction_head.{k}" for k in unexp])
                else:
                    _missing_all("prediction_head", self.prediction_head)

                if "gaussian_composer" in payload and isinstance(payload["gaussian_composer"], dict):
                    state = self._strip_module_prefix(payload["gaussian_composer"])
                    _merge_incompatible(
                        "gaussian_composer",
                        self.gaussian_composer.load_state_dict(state, strict=False),
                    )
                else:
                    _missing_all("gaussian_composer", self.gaussian_composer)

                if "second_layer_depth_head" in payload and isinstance(payload["second_layer_depth_head"], dict):
                    state = self._strip_module_prefix(payload["second_layer_depth_head"])
                    miss, unexp = self._load_depth_head_compat(state)
                    missing_keys.extend([f"second_layer_depth_head.{k}" for k in miss])
                    unexpected_keys.extend([f"second_layer_depth_head.{k}" for k in unexp])
                else:
                    _missing_all("second_layer_depth_head", self.second_layer_depth_head)

                expected_top = {
                    "step",
                    "feature_extractor",
                    "feature_decoder",
                    "init_model",
                    "prediction_head",
                    "gaussian_composer",
                    "second_layer_depth_head",
                    "config",
                    "decoder_params",
                    "optimizer",
                    "use_feature_only",
                    "unik3d_backbone",
                }
                for k in payload.keys():
                    if k not in expected_top:
                        unexpected_keys.append(f"payload.{k}")
                if strict and (missing_keys or unexpected_keys):
                    raise RuntimeError(
                        "Feature-only checkpoint is incompatible: "
                        f"missing={missing_keys[:20]} unexpected={unexpected_keys[:20]}"
                    )
                return _finish((missing_keys, unexpected_keys))
            
            elif "model" in payload:
                state = payload["model"]
                return _finish(self._load_unisharp_checkpoint(state, strict=strict))
            elif "state_dict" in payload and isinstance(payload["state_dict"], dict):
                state_dict = self._strip_module_prefix(payload["state_dict"])
                if self._looks_like_unik3d_state_dict(state_dict):
                    return _finish(self._load_unik3d_state_dict(state_dict))
                return _finish(self.load_state_dict(state_dict, strict=strict))
            else:
                raw_state = self._strip_module_prefix(payload)
                if self._looks_like_unik3d_state_dict(raw_state):
                    return _finish(self._load_unik3d_state_dict(raw_state))
                return _finish(self.load_state_dict(payload, strict=strict))
        else:
            return _finish(self.load_state_dict(payload, strict=strict))
    
    def _load_unisharp_checkpoint(self, state: dict, strict: bool = False) -> tuple[list[str], list[str]]:
        state = self._strip_module_prefix(state)
        init_state = {k.replace("init_model.", ""): v for k, v in state.items() if k.startswith("init_model.")}
        comp_state = {k.replace("gaussian_composer.", ""): v for k, v in state.items() if k.startswith("gaussian_composer.")}
        depth_head_state = {
            k.replace("second_layer_depth_head.", ""): v
            for k, v in state.items()
            if k.startswith("second_layer_depth_head.")
        }
        missing_keys = []
        unexpected_keys = []
        
        if init_state:
            m, u = self.init_model.load_state_dict(init_state, strict=False)
            missing_keys.extend(m)
            unexpected_keys.extend(u)
        
        
        if comp_state:
            m, u = self.gaussian_composer.load_state_dict(comp_state, strict=False)
            missing_keys.extend(m)
            unexpected_keys.extend(u)

        if depth_head_state:
            m, u = self._load_depth_head_compat(depth_head_state)
            missing_keys.extend(m)
            unexpected_keys.extend(u)

        known_prefixes = (
            "init_model.",
            "prediction_head.",
            "gaussian_composer.",
            "second_layer_depth_head.",
        )
        for k in state.keys():
            if not any(str(k).startswith(pref) for pref in known_prefixes):
                unexpected_keys.append(str(k))
        return (missing_keys, unexpected_keys)
    
    def save_checkpoint(self, path: str, step: int, optimizer: torch.optim.Optimizer | None = None) -> None:
        from dataclasses import asdict
        
        ckpt = {
            "step": step,
            "feature_extractor": self.feature_extractor.state_dict(),
            "feature_decoder": self.feature_decoder.state_dict(),
            "init_model": self.init_model.state_dict(),
            "prediction_head": self.prediction_head.state_dict(),
            "gaussian_composer": self.gaussian_composer.state_dict(),
            "second_layer_depth_head": self.second_layer_depth_head.state_dict(),
            "config": self.config.__dict__,
            "decoder_params": asdict(self.decoder_params),
            "use_feature_only": True,
            "unik3d_backbone": self.config.unik3d_backbone,
        }
        if optimizer is not None:
            ckpt["optimizer"] = optimizer.state_dict()
        
        torch.save(ckpt, path)

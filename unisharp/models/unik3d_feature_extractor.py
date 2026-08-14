from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn


LOGGER = logging.getLogger(__name__)


def _enable_unik3d_encoder_feature_capture(model: torch.nn.Module) -> None:
    try:
        encoder = model.pixel_encoder  # type: ignore[attr-defined]
    except Exception:
        return
    
    if getattr(encoder, "_unisharp_encoder_wrapped", False):
        return
    
    import types
    
    orig_forward = encoder.forward
    
    def wrapped_forward(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        output = orig_forward(*args, **kwargs)
        self._unisharp_last_encoder_output = output
        return output
    
    encoder.forward = types.MethodType(wrapped_forward, encoder)  # type: ignore[method-assign]
    encoder._unisharp_encoder_wrapped = True


def extract_unik3d_2d_feature_layers(
    encoder_output: Any,
    target_h: int,
    target_w: int,
) -> list[torch.Tensor]:
    def _to_bchw(feat_in: torch.Tensor) -> torch.Tensor:
        if feat_in.ndim == 4:
            if feat_in.shape[-1] > feat_in.shape[1]:
                return feat_in.permute(0, 3, 1, 2).contiguous()
            return feat_in
        if feat_in.ndim == 3:
            B, N, C = feat_in.shape
            import math

            ratio = target_h / max(target_w, 1)
            pH = max(1, int(math.sqrt(N * ratio) + 0.5))
            pW = max(1, N // pH)
            while pH * pW < N:
                pW += 1
            return feat_in[:, : pH * pW, :].transpose(1, 2).reshape(B, C, pH, pW)
        raise TypeError(f"Unsupported spatial feature ndim={feat_in.ndim}")

    def _resize(feat_in: torch.Tensor) -> torch.Tensor:
        if feat_in.shape[-2:] != (target_h, target_w):
            feat_in = torch.nn.functional.interpolate(
                feat_in,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )
        return feat_in

    feats: list[torch.Tensor] = []

    if isinstance(encoder_output, (list, tuple)) and len(encoder_output) == 2:
        spatial_or_cls, _ = encoder_output
        if isinstance(spatial_or_cls, (list, tuple)) and len(spatial_or_cls) > 0:
            spatial_candidates = [x for x in spatial_or_cls if isinstance(x, torch.Tensor)]
            if len(spatial_candidates) > 0:
                n = len(spatial_candidates)
                if n <= 4:
                    idxs = list(range(n))
                else:
                    idxs = sorted({n // 4, n // 2, (3 * n) // 4, n - 1})
                feats = [_resize(_to_bchw(spatial_candidates[i])) for i in idxs]
                channels = {int(f.shape[1]) for f in feats}
                if len(channels) != 1:
                    raise ValueError(
                        f"Selected DINO spatial features must share channels, got {sorted(channels)}"
                    )
                return feats

    if len(feats) == 0:
        def _find_spatial(x: Any, depth: int = 0) -> torch.Tensor | None:
            if isinstance(x, torch.Tensor):
                if x.ndim == 4:
                    return x
                if x.ndim == 3 and x.shape[1] > 2:
                    return x
                return None
            if isinstance(x, (list, tuple)) and depth < 3:
                for elem in x:
                    result = _find_spatial(elem, depth + 1)
                    if result is not None:
                        return result
            return None

        feat = _find_spatial(encoder_output)
    else:
        feat = None

    if feat is None or not isinstance(feat, torch.Tensor):
        raise TypeError(
            f"Cannot extract spatial 2D features from encoder_output of type {type(encoder_output)}. "
            "Expected (spatial_list, cls_list) from DINOv2 encoder."
        )
    return [_resize(_to_bchw(feat))]


def extract_unik3d_2d_features(
    encoder_output: Any,
    target_h: int,
    target_w: int,
) -> torch.Tensor:
    feats = extract_unik3d_2d_feature_layers(encoder_output, target_h, target_w)
    return torch.stack(feats, dim=0).mean(dim=0)


class DINOFeatureLayerFusion(nn.Module):

    def __init__(self, dim: int, max_layers: int = 4) -> None:
        super().__init__()
        self.dim = int(dim)
        self.max_layers = int(max_layers)
        self.proj = nn.ModuleList(
            [nn.Conv2d(self.dim, self.dim, kernel_size=1, bias=False) for _ in range(self.max_layers)]
        )
        self.layer_logits = nn.Parameter(torch.zeros(self.max_layers))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for proj in self.proj:
            nn.init.zeros_(proj.weight)
            eye = torch.eye(self.dim, dtype=proj.weight.dtype).view(self.dim, self.dim, 1, 1)
            with torch.no_grad():
                proj.weight.copy_(eye)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        if not features:
            raise ValueError("Expected at least one DINO feature layer.")
        if len(features) > self.max_layers:
            raise ValueError(f"Expected at most {self.max_layers} feature layers, got {len(features)}")
        channels = {int(f.shape[1]) for f in features}
        if channels != {self.dim}:
            raise ValueError(f"Expected DINO feature channels {self.dim}, got {sorted(channels)}")
        weights = torch.softmax(self.layer_logits[: len(features)], dim=0)
        fused = None
        for i, feat in enumerate(features):
            projected = self.proj[i](feat)
            weighted = projected * weights[i].to(dtype=projected.dtype, device=projected.device)
            fused = weighted if fused is None else fused + weighted
        if fused is None:
            raise RuntimeError("DINO feature fusion produced no output.")
        return fused


class UniK3DFeatureExtractor(nn.Module):
    
    def __init__(
        self,
        unik3d_model: nn.Module,
        dino_feature_dim: int = 1024,
    ):
        super().__init__()
        
        self.unik3d = unik3d_model
        self.dino_layer_fusion = DINOFeatureLayerFusion(dim=int(dino_feature_dim), max_layers=4)
        
        from unisharp.utils.unik3d_adapter import _enable_unik3d_decoder_feature_capture
        _enable_unik3d_decoder_feature_capture(self.unik3d)
        _enable_unik3d_encoder_feature_capture(self.unik3d)
        try:
            self.unik3d.pixel_decoder.radial_module._unisharp_detach_rays_embeddings = True  # type: ignore[attr-defined]
        except Exception:
            pass
        self._unisharp_last_unik3d_output: dict[str, torch.Tensor] | None = None
        
    def train(self, mode: bool = True) -> "UniK3DFeatureExtractor":
        super().train(mode)
        return self

    def _extract_features_from_output(
        self,
        output: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        self._unisharp_last_unik3d_output = output

        features_3d_pyramid = output.get("pyramid_features")
        if features_3d_pyramid is None:
            raise RuntimeError(
                "Failed to capture pyramid_features from UniK3D. "
                "Ensure _enable_unik3d_decoder_feature_capture() is called."
            )

        highest_res_3d = features_3d_pyramid[-1]
        actual_h, actual_w = highest_res_3d.shape[-2:]

        try:
            encoder = self.unik3d.pixel_encoder  # type: ignore[attr-defined]
            encoder_output = getattr(encoder, "_unisharp_last_encoder_output", None)
            if encoder_output is None:
                raise RuntimeError("Failed to capture encoder output from UniK3D.")
            dino_layers = extract_unik3d_2d_feature_layers(
                encoder_output,
                target_h=actual_h,
                target_w=actual_w,
            )
            features_2d = self.dino_layer_fusion(dino_layers)
        except Exception as e:
            LOGGER.error(f"Failed to extract 2D features: {e}")
            raise RuntimeError(
                "Failed to extract DINO 2D features. The Gaussian decoder expects "
                "DINO-channel features here; lowres UniK3D decoder features are not "
                "a safe fallback because their channels do not match Feature2DEncoder."
            ) from e

        return features_2d, features_3d_pyramid

    def forward(
        self,
        rgb_u8: torch.Tensor,
        target_h: int,
        target_w: int,
        intrinsics: torch.Tensor | None = None,
        camera_params: torch.Tensor | None = None,
        camera_model: str | None = None,
        hfov: float | None = None,
        vfov: float | None = None,
        validity_mask: torch.Tensor | None = None,
        use_predicted_rays: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        from unisharp.utils.unik3d_adapter import (
            build_unik3d_camera_rays,
            forward_unik3d_camera_rays,
            forward_unik3d_fisheye624,
            forward_unik3d_pinhole,
        )

        def _forward() -> dict[str, torch.Tensor]:
            if bool(use_predicted_rays):
                return forward_unik3d_camera_rays(
                    self.unik3d,
                    rgb_u8,
                    normalize=True,
                    validity_mask=validity_mask,
                )
            if torch.is_tensor(intrinsics):
                return forward_unik3d_pinhole(
                    self.unik3d,
                    rgb_u8,
                    intrinsics=intrinsics,
                    normalize=True,
                )
            if torch.is_tensor(camera_params):
                return forward_unik3d_fisheye624(
                    self.unik3d,
                    rgb_u8,
                    camera_params=camera_params,
                    normalize=True,
                    validity_mask=validity_mask,
                )
            _, rays, _, _ = build_unik3d_camera_rays(
                rgb_u8,
                device=next(self.unik3d.parameters()).device,
                camera_model=camera_model,
                hfov=float(2.0 * torch.pi) if hfov is None else float(hfov),
                vfov=float(torch.pi) if vfov is None else float(vfov),
            )
            return forward_unik3d_camera_rays(
                self.unik3d,
                rgb_u8,
                normalize=True,
                validity_mask=validity_mask,
                rays=rays,
            )

        output = _forward()
        return self._extract_features_from_output(output)

from __future__ import annotations

import torch
from torch import nn

from .feature_gaussian_decoder import ImageFeatures


class DirectPredictionHead(nn.Module):

    def __init__(self, feature_dim: int, num_layers: int) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_layers = int(num_layers)

        self.geometry_prediction_head = nn.Conv2d(self.feature_dim, 10 * self.num_layers, kernel_size=1)
        self.texture_prediction_head = nn.Conv2d(self.feature_dim, 4 * self.num_layers, kernel_size=1)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.geometry_prediction_head.weight.zero_()
            self.geometry_prediction_head.bias.zero_()
            self.texture_prediction_head.weight.zero_()
            self.texture_prediction_head.bias.zero_()

    def _copy_conv_weight_prefix(self, dst: nn.Conv2d, src: torch.Tensor) -> bool:
        if not isinstance(src, torch.Tensor):
            return False
        src_4d = src.reshape(src.shape[0], src.shape[1], 1, 1) if src.ndim == 2 else src
        if src_4d.ndim != 4 or src_4d.shape[0] != dst.weight.shape[0]:
            return False
        channels = min(int(src_4d.shape[1]), self.feature_dim, int(dst.weight.shape[1]))
        if channels <= 0 or tuple(src_4d.shape[2:]) != (1, 1):
            return False
        dst.weight[:, :channels].copy_(src_4d[:, :channels].to(device=dst.weight.device, dtype=dst.weight.dtype))
        return True

    def _copy_conv_weight_range(
        self,
        dst: nn.Conv2d,
        src: torch.Tensor,
        *,
        src_start: int,
        dst_start: int,
        count: int,
    ) -> bool:
        if not isinstance(src, torch.Tensor) or int(count) <= 0:
            return False
        src_4d = src.reshape(src.shape[0], src.shape[1], 1, 1) if src.ndim == 2 else src
        if src_4d.ndim != 4 or tuple(src_4d.shape[2:]) != (1, 1):
            return False
        src_end = int(src_start) + int(count)
        dst_end = int(dst_start) + int(count)
        if src_start < 0 or dst_start < 0 or src_end > int(src_4d.shape[0]) or dst_end > int(dst.weight.shape[0]):
            return False
        channels = min(int(src_4d.shape[1]), self.feature_dim, int(dst.weight.shape[1]))
        if channels <= 0:
            return False
        dst.weight[dst_start:dst_end, :channels].copy_(
            src_4d[src_start:src_end, :channels].to(device=dst.weight.device, dtype=dst.weight.dtype)
        )
        return True

    @staticmethod
    def _copy_bias_range(
        dst: nn.Parameter,
        src: torch.Tensor,
        *,
        src_start: int,
        dst_start: int,
        count: int,
    ) -> bool:
        if not isinstance(src, torch.Tensor) or src.ndim != 1 or int(count) <= 0:
            return False
        src_end = int(src_start) + int(count)
        dst_end = int(dst_start) + int(count)
        if src_start < 0 or dst_start < 0 or src_end > int(src.shape[0]) or dst_end > int(dst.shape[0]):
            return False
        dst[dst_start:dst_end].copy_(src[src_start:src_end].to(device=dst.device, dtype=dst.dtype))
        return True

    def init_from_legacy_direct_state(self, state: dict[str, torch.Tensor]) -> int:
        copied = 0
        with torch.no_grad():
            geo_w = state.get(
                "geometry_weight",
                state.get("geometry_prediction_head.weight", state.get("geo_fc2.weight")),
            )
            geo_b = state.get(
                "geometry_bias",
                state.get("geometry_prediction_head.bias", state.get("geo_fc2.bias")),
            )
            tex_w = state.get(
                "texture_weight",
                state.get("texture_prediction_head.weight", state.get("tex_fc2.weight")),
            )
            tex_b = state.get(
                "texture_bias",
                state.get("texture_prediction_head.bias", state.get("tex_fc2.bias")),
            )

            l = self.num_layers
            if isinstance(geo_w, torch.Tensor):
                if int(geo_w.shape[0]) == 10 * l and self._copy_conv_weight_prefix(self.geometry_prediction_head, geo_w):
                    copied += 1
                elif self._copy_conv_weight_range(
                    self.geometry_prediction_head,
                    geo_w,
                    src_start=0,
                    dst_start=0,
                    count=3 * l,
                ):
                    copied += 1
            if isinstance(geo_b, torch.Tensor):
                if tuple(geo_b.shape) == (10 * l,):
                    self.geometry_prediction_head.bias.copy_(
                        geo_b.to(device=self.geometry_prediction_head.bias.device, dtype=self.geometry_prediction_head.bias.dtype)
                    )
                    copied += 1
                elif self._copy_bias_range(self.geometry_prediction_head.bias, geo_b, src_start=0, dst_start=0, count=3 * l):
                    copied += 1
            if isinstance(tex_w, torch.Tensor):
                if int(tex_w.shape[0]) == 4 * l and self._copy_conv_weight_prefix(self.texture_prediction_head, tex_w):
                    copied += 1
                elif int(tex_w.shape[0]) == 11 * l:
                    if self._copy_conv_weight_range(self.geometry_prediction_head, tex_w, src_start=0, dst_start=3 * l, count=7 * l):
                        copied += 1
                    if self._copy_conv_weight_range(self.texture_prediction_head, tex_w, src_start=7 * l, dst_start=0, count=4 * l):
                        copied += 1
            if isinstance(tex_b, torch.Tensor):
                if tuple(tex_b.shape) == (4 * l,):
                    self.texture_prediction_head.bias.copy_(
                        tex_b.to(device=self.texture_prediction_head.bias.device, dtype=self.texture_prediction_head.bias.dtype)
                    )
                    copied += 1
                elif tuple(tex_b.shape) == (11 * l,):
                    if self._copy_bias_range(self.geometry_prediction_head.bias, tex_b, src_start=0, dst_start=3 * l, count=7 * l):
                        copied += 1
                    if self._copy_bias_range(self.texture_prediction_head.bias, tex_b, src_start=7 * l, dst_start=0, count=4 * l):
                        copied += 1
        return copied

    def forward(self, image_features: ImageFeatures) -> torch.Tensor:
        delta_geo = self.geometry_prediction_head(image_features.geometry_features)
        delta_texture = self.texture_prediction_head(image_features.texture_features)
        delta_geo = delta_geo.unflatten(1, (10, self.num_layers))
        delta_texture = delta_texture.unflatten(1, (4, self.num_layers))
        return torch.cat([delta_geo, delta_texture], dim=1)


GaussianConditionedPredictionHead = DirectPredictionHead

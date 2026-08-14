from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from unisharp.models.blocks import FeatureFusionBlock2d, NormLayerName, residual_block_2d
from unisharp.models.decoder import MultiresConvDecoder


LOGGER = logging.getLogger(__name__)


class ImageFeatures(NamedTuple):

    texture_features: torch.Tensor
    geometry_features: torch.Tensor


class CircularAwareConv2d(nn.Conv2d):

    circular_horizontal: bool

    @classmethod
    def from_conv2d(cls, conv: nn.Conv2d) -> "CircularAwareConv2d":
        out = cls(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=conv.bias is not None,
            padding_mode=conv.padding_mode,
            device=conv.weight.device,
            dtype=conv.weight.dtype,
        )
        with torch.no_grad():
            out.weight.copy_(conv.weight)
            if conv.bias is not None and out.bias is not None:
                out.bias.copy_(conv.bias)
        return out

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self.circular_horizontal = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not bool(self.circular_horizontal):
            return super().forward(input)
        pad_h, pad_w = self._padding_hw()
        if pad_h == 0 and pad_w == 0:
            return F.conv2d(
                input,
                self.weight,
                self.bias,
                self.stride,
                0,
                self.dilation,
                self.groups,
            )
        x = input
        if pad_w > 0:
            x = F.pad(x, (pad_w, pad_w, 0, 0), mode="circular")
        if pad_h > 0:
            x = F.pad(x, (0, 0, pad_h, pad_h), mode="constant", value=0.0)
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            0,
            self.dilation,
            self.groups,
        )

    def _padding_hw(self) -> tuple[int, int]:
        if isinstance(self.padding, tuple):
            if len(self.padding) == 2:
                return int(self.padding[0]), int(self.padding[1])
            return int(self.padding[0]), int(self.padding[0])
        return int(self.padding), int(self.padding)


def _convert_conv2d_modules_to_circular_aware(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d) and not isinstance(child, CircularAwareConv2d):
            setattr(module, name, CircularAwareConv2d.from_conv2d(child))
        else:
            _convert_conv2d_modules_to_circular_aware(child)


def _set_circular_horizontal(module: nn.Module, enabled: bool) -> None:
    for child in module.modules():
        if hasattr(child, "circular_horizontal"):
            child.circular_horizontal = bool(enabled)


@dataclass
class FeatureGaussianDecoderParams:

    dims_3d_in: tuple[int, int, int, int] = (128, 256, 512, 512)
    dims_3d_out: tuple[int, int, int, int] = (256, 512, 1024, 1024)
    
    dim_2d_in: int = 1024
    dim_2d_out: int = 256
    
    dim_decoder_out: int = 256
    
    dim_texture_out: int = 32
    dim_geometry_out: int = 32
    norm_type: NormLayerName = "group_norm"
    norm_num_groups: int = 8
    
    stride_out: int = 2

    use_learned_upsampling: bool = False
    target_resolution: tuple[int, int] | None = None


class Feature2DEncoder(nn.Module):
    
    def __init__(
        self,
        dim_in: int = 1024,
        dim_out: int = 256,
    ):
        super().__init__()
        
        self.process = nn.Sequential(
            nn.Conv2d(dim_in, 512, kernel_size=3, padding=1),
            nn.GroupNorm(8, 512),
            nn.GELU(),
            
            nn.Conv2d(512, dim_out, kernel_size=3, padding=1),
            nn.GroupNorm(8, dim_out),
            nn.GELU(),
        )
        
        self.dim_out = dim_out
    
    def forward(
        self,
        x: torch.Tensor,
        target_h: int,
        target_w: int,
    ) -> torch.Tensor:
        x = self.process(x)
        x = torch.nn.functional.interpolate(
            x,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )
        
        return x


class Feature3DProjector(nn.Module):
    
    def __init__(
        self,
        dims_in: list[int],
        dims_out: list[int],
    ):
        super().__init__()
        
        if len(dims_in) != len(dims_out):
            raise ValueError(
                f"dims_in and dims_out must have same length, "
                f"got {len(dims_in)} vs {len(dims_out)}"
            )
        
        self.projectors = nn.ModuleList([
            nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=False)
            for dim_in, dim_out in zip(dims_in, dims_out)
        ])
        
        self.dims_out = dims_out
        self.num_levels = len(dims_in)
    
    def forward(self, pyramid_features: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(pyramid_features) != self.num_levels:
            raise ValueError(
                f"Expected {self.num_levels} pyramid features, got {len(pyramid_features)}"
            )
        
        return [proj(feat) for proj, feat in zip(self.projectors, pyramid_features)]


class LearnedUpsampler(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.GroupNorm(8, in_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.GroupNorm(8, out_channels)
        
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        x_upsampled = torch.nn.functional.interpolate(
            x, size=(target_h, target_w), mode="bilinear", align_corners=False
        )
        
        identity = self.skip(x_upsampled)
        
        out = self.conv1(x_upsampled)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + identity
        out = self.relu(out)
        
        return out


def _create_project_upsample_block(dim_in: int, dim_out: int, upsample_layers: int) -> nn.Module:
    blocks: list[nn.Module] = [
        nn.Conv2d(
            in_channels=dim_in,
            out_channels=dim_out,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
    ]
    blocks.extend(
        nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            CircularAwareConv2d(
                in_channels=dim_out,
                out_channels=dim_out,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )
        for _ in range(int(upsample_layers))
    )
    return nn.Sequential(*blocks)


class FeatureGaussianDecoder(nn.Module):
    
    def __init__(
        self,
        params: FeatureGaussianDecoderParams,
    ):
        super().__init__()
        
        self.params = params
        self.stride_out = params.stride_out
        self.norm_type = params.norm_type
        self.norm_num_groups = int(params.norm_num_groups)
        if int(self.stride_out) not in (1, 2):
            raise ValueError(f"FeatureGaussianDecoder only supports stride_out 1 or 2, got {self.stride_out}")
        
        self.feature_3d_projector = Feature3DProjector(
            dims_in=list(params.dims_3d_in),
            dims_out=list(params.dims_3d_out),
        )
        
        self.decoder = MultiresConvDecoder(
            dims_encoder=list(params.dims_3d_out),
            dims_decoder=params.dim_decoder_out,
        )
        if int(self.stride_out) == 1:
            self.upsample = _create_project_upsample_block(
                params.dim_decoder_out,
                params.dim_decoder_out,
                upsample_layers=1,
            )
        else:
            self.upsample = nn.Identity()
        
        self.feature_2d_encoder = Feature2DEncoder(
            dim_in=params.dim_2d_in,
            dim_out=params.dim_2d_out,
        )
        
        self.fusion = FeatureFusionBlock2d(
            params.dim_decoder_out,
            params.dim_2d_out,
        )
        
        self.texture_head = self._create_head(
            params.dim_decoder_out,
            params.dim_texture_out,
        )
        
        self.geometry_head = self._create_head(
            params.dim_decoder_out,
            params.dim_geometry_out,
        )

        if int(params.dim_2d_out) != int(params.dim_decoder_out):
            raise ValueError(
                "FeatureFusionBlock2d requires 2D skip channels to match decoder channels, "
                f"got dim_2d_out={params.dim_2d_out}, dim_decoder_out={params.dim_decoder_out}"
            )

        self.dim_out = params.dim_texture_out
        
        self.fused_upsampler = None

        _convert_conv2d_modules_to_circular_aware(self)
        
    def _create_head(self, dim_in: int, dim_out: int) -> nn.Module:
        return nn.Sequential(
            residual_block_2d(
                dim_in=dim_in,
                dim_out=dim_in,
                dim_hidden=dim_in // 2,
                norm_type=self.norm_type,
                norm_num_groups=self.norm_num_groups,
            ),
            residual_block_2d(
                dim_in=dim_in,
                dim_hidden=dim_in // 2,
                dim_out=dim_in,
                norm_type=self.norm_type,
                norm_num_groups=self.norm_num_groups,
            ),
            nn.ReLU(),
            nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1),
            nn.ReLU(),
        )
    
    def forward(
        self,
        features_2d: torch.Tensor,
        features_3d_pyramid: list[torch.Tensor],
        *,
        circular_horizontal: bool = False,
        target_hw: tuple[int, int] | None = None,
    ) -> ImageFeatures:
        _set_circular_horizontal(self, bool(circular_horizontal))
        features_3d_sorted = sorted(
            features_3d_pyramid,
            key=lambda t: int(t.shape[-2]) * int(t.shape[-1]),
            reverse=True
        )
        
        pyramid_projected = self.feature_3d_projector(features_3d_sorted)
        
        decoder_out = self.decoder(pyramid_projected).contiguous()
        decoder_out = self.upsample(decoder_out).contiguous()
        if target_hw is not None:
            target_h, target_w = int(target_hw[0]), int(target_hw[1])
            if target_h <= 0 or target_w <= 0:
                raise ValueError(f"target_hw must be positive, got {target_hw}")
            if tuple(decoder_out.shape[-2:]) != (target_h, target_w):
                decoder_out = F.interpolate(
                    decoder_out,
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                ).contiguous()
        
        target_h, target_w = decoder_out.shape[-2:]
        features_2d_proj = self.feature_2d_encoder(
            features_2d,
            target_h=target_h,
            target_w=target_w,
        )
        
        fused = self.fusion(decoder_out, features_2d_proj)
        
        if target_hw is not None:
            target_h, target_w = int(target_hw[0]), int(target_hw[1])
            if tuple(fused.shape[-2:]) != (target_h, target_w):
                raise RuntimeError(
                    "Feature decoder grid must match base Gaussian grid before heads, "
                    f"got fused={tuple(fused.shape[-2:])} target={(target_h, target_w)}. "
                    "Only high-channel decoder features may be adapted before fusion."
                )

        texture_features = self.texture_head(fused)
        geometry_features = self.geometry_head(fused)
        
        return ImageFeatures(
            texture_features=texture_features,
            geometry_features=geometry_features,
        )
    
    @property
    def stride(self) -> int:
        return self.stride_out


def create_feature_gaussian_decoder(
    params: FeatureGaussianDecoderParams | None = None,
) -> FeatureGaussianDecoder:
    if params is None:
        params = FeatureGaussianDecoderParams()
    
    return FeatureGaussianDecoder(params)

from __future__ import annotations

import dataclasses
from typing import Literal

import unisharp.utils.math as math_utils


@dataclasses.dataclass
class DeltaFactor:

    xy: float = 0.001
    z: float = 0.001
    color: float = 0.1
    opacity: float = 1.0
    scale: float = 1.0
    quaternion: float = 1.0


@dataclasses.dataclass
class PanoInitializerParams:

    stride: int = 2
    num_layers: int = 2
    scale_factor: float = 1.0
    opacity_init: float = 0.5
    normalize_distance: bool = True
    circular_horizontal: bool = True

    first_layer_depth_option: Literal["surface_min", "surface_max"] = "surface_min"
    rest_layer_depth_option: Literal["surface_min", "surface_max"] = "surface_min"

@dataclasses.dataclass
class PanoPredictorParams:

    initializer: PanoInitializerParams = dataclasses.field(default_factory=PanoInitializerParams)

    delta_factor: DeltaFactor = dataclasses.field(default_factory=DeltaFactor)
    color_activation_type: math_utils.ActivationType = "sigmoid"
    opacity_activation_type: math_utils.ActivationType = "sigmoid"
    max_scale: float = 10.0
    min_scale: float = 0.0
    base_scale_on_predicted_mean: bool = True

    unik3d_backbone: str = "vitl"
    unik3d_pretrained: bool = True
    num_monodepth_layers: int = 2

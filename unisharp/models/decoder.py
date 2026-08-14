from __future__ import annotations

import abc
from typing import Iterable

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from unisharp.models.blocks import FeatureFusionBlock2d, UpsamplingMode


class BaseDecoder(nn.Module, abc.ABC):
    dim_out: int

    @abc.abstractmethod
    def forward(self, encodings: list[torch.Tensor]) -> torch.Tensor:
        pass


class MultiresConvDecoder(BaseDecoder):
    def __init__(
        self,
        dims_encoder: Iterable[int],
        dims_decoder: Iterable[int] | int,
        grad_checkpointing: bool = False,
        upsampling_mode: UpsamplingMode = "transposed_conv",
    ):
        super().__init__()
        self.dims_encoder = list(dims_encoder)

        if isinstance(dims_decoder, int):
            self.dims_decoder = [dims_decoder] * len(self.dims_encoder)
        else:
            self.dims_decoder = list(dims_decoder)

        if len(self.dims_decoder) != len(self.dims_encoder):
            raise ValueError("Received dims_encoder and dims_decoder of different sizes.")

        self.dim_out = self.dims_decoder[0]
        num_encoders = len(self.dims_encoder)
        conv0 = (
            nn.Conv2d(self.dims_encoder[0], self.dims_decoder[0], kernel_size=1, bias=False)
            if self.dims_encoder[0] != self.dims_decoder[0]
            else nn.Identity()
        )

        convs = [conv0]
        for i in range(1, num_encoders):
            convs.append(
                nn.Conv2d(
                    self.dims_encoder[i],
                    self.dims_decoder[i],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                )
            )
        self.convs = nn.ModuleList(convs)

        fusions = []
        for i in range(num_encoders):
            fusions.append(
                FeatureFusionBlock2d(
                    dim_in=self.dims_decoder[i],
                    dim_out=self.dims_decoder[i - 1] if i != 0 else self.dim_out,
                    upsampling_mode=upsampling_mode if i != 0 else None,
                    batch_norm=False,
                )
            )
        self.fusions = nn.ModuleList(fusions)
        self.grad_checkpointing = grad_checkpointing

    @torch.jit.ignore
    def set_grad_checkpointing(self, is_enabled=True):
        self.grad_checkpointing = is_enabled

    def _checkpoint(self, fn, *args):
        if self.grad_checkpointing:
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def forward(self, encodings: list[torch.Tensor]) -> torch.Tensor:
        num_levels = len(encodings)
        num_encoders = len(self.dims_encoder)
        if num_levels != num_encoders:
            raise ValueError(
                f"Encoder output levels={num_levels} at runtime "
                f"mismatch with expected levels={num_encoders}."
            )

        features = self.convs[-1](encodings[-1])
        features = self._checkpoint(self.fusions[-1], features)
        for i in range(num_levels - 2, -1, -1):
            features_i = self.convs[i](encodings[i])
            features = self._checkpoint(self.fusions[i], features, features_i)
        return features

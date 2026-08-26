"""Target-camera set conditioning for target-aware Gaussian prediction."""

from __future__ import annotations

import torch
from torch import nn


class TargetRigEncoder(nn.Module):
    """Encode a set of target cameras expressed in the source-camera frame.

    Each target pose must be a rigid ``source_w2c -> target_w2c`` transform.
    The encoder is permutation-invariant over target views, allowing a model
    trained with one target view per sample to be fine-tuned later with a
    fixed multi-camera rig without changing the decoder interface.
    """

    def __init__(self, embedding_dim: int, *, translation_scale: float = 1.0) -> None:
        super().__init__()
        if int(embedding_dim) <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        if float(translation_scale) <= 0.0:
            raise ValueError("translation_scale must be positive")
        self.embedding_dim = int(embedding_dim)
        self.translation_scale = float(translation_scale)
        # R6D (two rotation rows), normalized translation, and normalized K.
        self.view_mlp = nn.Sequential(
            nn.Linear(13, self.embedding_dim),
            nn.SiLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.SiLU(),
        )
        self.output_norm = nn.LayerNorm(self.embedding_dim)

    @staticmethod
    def _as_bv44(poses: torch.Tensor) -> torch.Tensor:
        if poses.ndim == 3:
            poses = poses.unsqueeze(1)
        if poses.ndim != 4 or tuple(poses.shape[-2:]) != (4, 4):
            raise ValueError(f"Expected target poses shape (B,V,4,4), got {tuple(poses.shape)}")
        return poses

    @staticmethod
    def _as_bv33(intrinsics: torch.Tensor, *, batch_size: int, views: int) -> torch.Tensor:
        if intrinsics.ndim == 3:
            intrinsics = intrinsics.unsqueeze(1)
        if intrinsics.ndim != 4 or tuple(intrinsics.shape[-2:]) != (3, 3):
            raise ValueError(f"Expected target intrinsics shape (B,V,3,3), got {tuple(intrinsics.shape)}")
        if int(intrinsics.shape[0]) != int(batch_size) or int(intrinsics.shape[1]) != int(views):
            raise ValueError(
                "Target pose/intrinsic batch mismatch: "
                f"poses={(batch_size, views)} intrinsics={tuple(intrinsics.shape[:2])}"
            )
        return intrinsics

    def forward(
        self,
        target_w2c: torch.Tensor | None,
        target_intrinsics: torch.Tensor | None,
        *,
        image_h: int,
        image_w: int,
        valid_mask: torch.Tensor | None = None,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        """Return one embedding per source image.

        A missing rig produces a zero embedding, so non-pinhole and legacy
        call sites remain valid.  Pixel intrinsics are normalized by the
        source image shape before encoding.
        """
        if target_w2c is None:
            if batch_size is None:
                raise ValueError("batch_size is required when target_w2c is None")
            device = next(self.parameters()).device
            return torch.zeros((int(batch_size), self.embedding_dim), device=device, dtype=torch.float32)

        pose = self._as_bv44(target_w2c).to(dtype=torch.float32)
        batch, views = int(pose.shape[0]), int(pose.shape[1])
        if target_intrinsics is None:
            intr = torch.zeros((batch, views, 4), device=pose.device, dtype=pose.dtype)
        else:
            k = self._as_bv33(target_intrinsics, batch_size=batch, views=views).to(device=pose.device, dtype=pose.dtype)
            intr = torch.stack(
                (
                    k[..., 0, 0] / max(float(image_w), 1.0),
                    k[..., 1, 1] / max(float(image_h), 1.0),
                    k[..., 0, 2] / max(float(image_w), 1.0),
                    k[..., 1, 2] / max(float(image_h), 1.0),
                ),
                dim=-1,
            )

        rotation_6d = pose[..., :2, :3].reshape(batch, views, 6)
        translation = torch.tanh(pose[..., :3, 3] / self.translation_scale)
        per_view = self.view_mlp(torch.cat((rotation_6d, translation, intr), dim=-1))

        if valid_mask is None:
            weights = torch.ones((batch, views, 1), device=pose.device, dtype=per_view.dtype)
        else:
            valid = valid_mask
            if valid.ndim == 1:
                valid = valid.unsqueeze(1)
            if tuple(valid.shape) != (batch, views):
                raise ValueError(f"Expected target valid mask shape {(batch, views)}, got {tuple(valid.shape)}")
            weights = valid.to(device=pose.device, dtype=per_view.dtype).unsqueeze(-1)
        pooled = (per_view * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        return self.output_norm(pooled)

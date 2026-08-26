"""NPU-friendly operations for homogeneous rigid camera transforms."""

from __future__ import annotations

import torch


def invert_rigid_transform(transform: torch.Tensor) -> torch.Tensor:
    """Invert batched or unbatched 4x4 rigid transforms without ``linalg.inv``.

    Camera poses in the training path have the homogeneous form ``[R | t]``
    with an orthonormal 3x3 rotation ``R``.  Their inverse is exactly
    ``[R.T | -R.T @ t]``.  Using this expression avoids ``linalg_inv_ex``,
    which is not implemented by some torch_npu/CANN combinations and can
    otherwise trigger a costly CPU fallback.

    ``transform`` may have arbitrary leading batch dimensions but must end in
    ``(4, 4)``.  The function preserves the input device and dtype.
    """
    if transform.ndim < 2 or tuple(transform.shape[-2:]) != (4, 4):
        raise ValueError(f"Expected transform shape (..., 4, 4), got {tuple(transform.shape)}")

    rotation_t = transform[..., :3, :3].transpose(-1, -2)
    translation_inv = -(rotation_t @ transform[..., :3, 3:4])

    inverse = torch.zeros_like(transform)
    inverse[..., :3, :3] = rotation_t
    inverse[..., :3, 3:4] = translation_inv
    inverse[..., 3, 3] = 1.0
    return inverse

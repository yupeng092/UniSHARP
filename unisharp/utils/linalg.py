
from __future__ import annotations

import torch
from scipy.spatial.transform import Rotation


def rotation_matrices_from_quaternions(quaternions: torch.Tensor) -> torch.Tensor:
    device = quaternions.device
    shape = quaternions.shape[:-1]

    quaternions = quaternions / torch.linalg.norm(quaternions, dim=-1, keepdim=True)
    real_part = quaternions[..., 0]
    vector_part = quaternions[..., 1:]

    vector_cross = get_cross_product_matrix(vector_part)
    real_part = real_part[..., None, None]

    matrix_outer = vector_part[..., :, None] * vector_part[..., None, :]
    matrix_diag = real_part.square() * eyes(3, shape=shape, device=device)
    matrix_cross_1 = 2 * real_part * vector_cross
    matrix_cross_2 = vector_cross @ vector_cross

    return matrix_outer + matrix_diag + matrix_cross_1 + matrix_cross_2


def quaternions_from_rotation_matrices(matrices: torch.Tensor) -> torch.Tensor:
    if not matrices.shape[-2:] == (3, 3):
        raise ValueError(f"matrices have invalid shape {matrices.shape}")
    matrices_np = matrices.detach().cpu().numpy()
    quaternions_np = Rotation.from_matrix(matrices_np.reshape(-1, 3, 3)).as_quat()
    quaternions_np = quaternions_np[:, [3, 0, 1, 2]]
    quaternions_np = quaternions_np.reshape(matrices_np.shape[:-2] + (4,))
    return torch.as_tensor(quaternions_np, device=matrices.device, dtype=matrices.dtype)


def get_cross_product_matrix(vectors: torch.Tensor) -> torch.Tensor:
    if not vectors.shape[-1] == 3:
        raise ValueError("Only 3-dimensional vectors are supported")
    device = vectors.device
    shape = vectors.shape[:-1]
    unit_basis = eyes(3, shape=shape, device=device)
    return torch.cross(vectors[..., :, None], unit_basis, dim=-2)


def eyes(
    dim: int, shape: tuple[int, ...], device: torch.device | str | None = None
) -> torch.Tensor:
    return torch.eye(dim, device=device).broadcast_to(shape + (dim, dim)).clone()



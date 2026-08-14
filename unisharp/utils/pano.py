from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


CubemapFace = Literal["F", "R", "B", "L", "U", "D"]
_FACE_ORDER: list[CubemapFace] = ["U", "B", "L", "F", "R", "D"]


def get_pinhole_intrinsics_4x4(face_w: int, fov_degrees: float = 90.0) -> torch.Tensor:
    fov = math.radians(fov_degrees)
    f_px = (face_w - 1) / 2.0 / math.tan(fov / 2.0)
    cx = (face_w - 1) / 2.0
    cy = (face_w - 1) / 2.0
    intr = torch.tensor(
        [
            [f_px, 0.0, cx, 0.0],
            [0.0, f_px, cy, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    return intr


def _rotation_world_to_cam(face: CubemapFace) -> torch.Tensor:
    if face == "F":
        x_cam = torch.tensor([1.0, 0.0, 0.0])
        y_cam = torch.tensor([0.0, 1.0, 0.0])
        z_cam = torch.tensor([0.0, 0.0, 1.0])
    elif face == "R":
        x_cam = torch.tensor([0.0, 0.0, -1.0])
        y_cam = torch.tensor([0.0, 1.0, 0.0])
        z_cam = torch.tensor([1.0, 0.0, 0.0])
    elif face == "B":
        x_cam = torch.tensor([-1.0, 0.0, 0.0])
        y_cam = torch.tensor([0.0, 1.0, 0.0])
        z_cam = torch.tensor([0.0, 0.0, -1.0])
    elif face == "L":
        x_cam = torch.tensor([0.0, 0.0, 1.0])
        y_cam = torch.tensor([0.0, 1.0, 0.0])
        z_cam = torch.tensor([-1.0, 0.0, 0.0])
    elif face == "U":
        x_cam = torch.tensor([-1.0, 0.0, 0.0])
        y_cam = torch.tensor([0.0, 0.0, -1.0])
        z_cam = torch.tensor([0.0, -1.0, 0.0])
    elif face == "D":
        x_cam = torch.tensor([-1.0, 0.0, 0.0])
        y_cam = torch.tensor([0.0, 0.0, 1.0])
        z_cam = torch.tensor([0.0, 1.0, 0.0])
    else:
        raise ValueError(f"Unsupported face: {face}")

    return torch.stack([x_cam, y_cam, z_cam], dim=0)


def get_cubemap_extrinsics_4x4(
    device: torch.device,
    yaw_degrees: float = 0.0,
    faces: list[CubemapFace] | None = None,
) -> torch.Tensor:
    if faces is None:
        faces = _FACE_ORDER

    yaw = math.radians(yaw_degrees)
    cy, sy = math.cos(yaw), math.sin(yaw)
    r_yaw = torch.tensor(
        [
            [cy, 0.0, sy],
            [0.0, 1.0, 0.0],
            [-sy, 0.0, cy],
        ],
        dtype=torch.float32,
        device=device,
    )

    mats = []
    for face in faces:
        r_wc0 = _rotation_world_to_cam(face).to(device=device)
        r_wc = r_wc0 @ r_yaw.T
        ext = torch.eye(4, dtype=torch.float32, device=device)
        ext[:3, :3] = r_wc
        mats.append(ext)
    return torch.stack(mats, dim=0)


class Cube2Equirec(nn.Module):

    def __init__(self, face_w: int, equ_h: int, equ_w: int) -> None:
        super().__init__()
        self.face_w = face_w
        self.equ_h = equ_h
        self.equ_w = equ_w

        tp, sample_grid = self._build_sample_grid()
        self.register_buffer("tp", tp, persistent=False)
        self.register_buffer("sample_grid", sample_grid, persistent=False)

    def _build_sample_grid(self) -> tuple[torch.Tensor, torch.Tensor]:
        equ_h, equ_w = self.equ_h, self.equ_w
        device = torch.device("cpu")

        tp = np.roll(
            np.arange(4).repeat(equ_w // 4)[None, :].repeat(equ_h, 0), 3 * equ_w // 8, 1
        )

        mask = np.zeros((equ_h, equ_w // 4), np.bool_)
        idx = np.linspace(-np.pi, np.pi, equ_w // 4) / 4
        idx = equ_h // 2 - np.round(np.arctan(np.cos(idx)) * equ_h / np.pi).astype(int)
        for i, j in enumerate(idx):
            mask[:j, i] = 1
        mask = np.roll(np.concatenate([mask] * 4, 1), 3 * equ_w // 8, 1)
        tp[mask] = 4
        tp[np.flip(mask, 0)] = 5

        lon = ((np.linspace(0, equ_w - 1, num=equ_w, dtype=np.float32) + 0.5) / equ_w - 0.5) * 2 * np.pi
        lat = -((np.linspace(0, equ_h - 1, num=equ_h, dtype=np.float32) + 0.5) / equ_h - 0.5) * np.pi
        lon, lat = np.meshgrid(lon, lat)
        coor_u = np.zeros((equ_h, equ_w), dtype=np.float32)
        coor_v = np.zeros((equ_h, equ_w), dtype=np.float32)
        for i in range(4):
            m = tp == i
            coor_u[m] = 0.5 * np.tan(lon[m] - np.pi * i / 2)
            coor_v[m] = -0.5 * np.tan(lat[m]) / np.cos(lon[m] - np.pi * i / 2)
        m = tp == 4
        c = 0.5 * np.tan(np.pi / 2 - lat[m])
        coor_u[m] = c * np.sin(lon[m])
        coor_v[m] = c * np.cos(lon[m])
        m = tp == 5
        c = 0.5 * np.tan(np.pi / 2 - np.abs(lat[m]))
        coor_u[m] = c * np.sin(lon[m])
        coor_v[m] = -c * np.cos(lon[m])

        coor_u = (np.clip(coor_u, -0.5, 0.5)) * 2
        coor_v = (np.clip(coor_v, -0.5, 0.5)) * 2

        tp_t = torch.from_numpy(tp.astype(np.float32) / 2.5 - 1.0).to(device)
        u_t = torch.from_numpy(coor_u).to(device)
        v_t = torch.from_numpy(coor_v).to(device)

        sample_grid = torch.stack([u_t, v_t, tp_t], dim=-1).view(1, 1, equ_h, equ_w, 3)
        return tp_t, sample_grid

    def forward(self, cube_feat: torch.Tensor) -> torch.Tensor:
        bs = cube_feat.shape[0]
        cube_feat = cube_feat[:, :, [3, 4, 1, 2, 0, 5], :, :]
        cube_feat[:, :, 4:] = torch.flip(cube_feat[:, :, 4:], [3, 4])
        sample_grid = torch.cat([self.sample_grid.to(cube_feat.device)] * bs, dim=0)
        equi_feat = F.grid_sample(
            cube_feat,
            sample_grid,
            padding_mode="border",
            align_corners=True,
        )
        return equi_feat.squeeze(2)




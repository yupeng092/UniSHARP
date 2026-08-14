
from __future__ import annotations

import torch
import torch.nn.functional as F
from unisharp.utils.pixel_convention import integer_pixel_center_grid, scale_intrinsics_align_corners_false


def _pixel_grid(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    xx, yy = integer_pixel_center_grid(int(height), int(width), device=device, dtype=dtype)
    return xx.reshape(-1), yy.reshape(-1)


def _solve_linear_2param(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a.to(dtype=torch.float32)
    b = b.to(device=a.device, dtype=a.dtype)
    with torch.autocast(device_type=a.device.type, enabled=False):
        ata = a.transpose(0, 1) @ a
        ridge = torch.eye(2, device=a.device, dtype=a.dtype) * 1e-6
        atb = a.transpose(0, 1) @ b
        return torch.linalg.solve(ata + ridge, atb)


def _solve_linear(a: torch.Tensor, b: torch.Tensor, *, ridge: float = 1e-6) -> torch.Tensor:
    a = a.to(dtype=torch.float32)
    b = b.to(device=a.device, dtype=a.dtype)
    with torch.autocast(device_type=a.device.type, enabled=False):
        ata = a.transpose(0, 1) @ a
        reg = torch.eye(int(a.shape[1]), device=a.device, dtype=a.dtype) * float(ridge)
        atb = a.transpose(0, 1) @ b
        return torch.linalg.solve(ata + reg, atb)


def fit_pinhole_intrinsics_from_rays(
    rays: torch.Tensor,
    *,
    min_focal_px: float = 1.0,
    max_samples: int = 65536,
) -> torch.Tensor:
    if rays.ndim != 4 or int(rays.shape[1]) != 3:
        raise ValueError(f"Expected rays shape (B,3,H,W), got {tuple(rays.shape)}")
    with torch.no_grad():
        rays_f = rays.detach().to(dtype=torch.float32)
        bsz, _, height, width = rays_f.shape
        uu, vv = _pixel_grid(height, width, device=rays_f.device, dtype=rays_f.dtype)
        stride = max(1, int((height * width + int(max_samples) - 1) // int(max_samples)))
        uu_s = uu[::stride]
        vv_s = vv[::stride]
        out = torch.zeros((bsz, 3, 3), device=rays_f.device, dtype=rays_f.dtype)
        out[:, 2, 2] = 1.0
        for b in range(int(bsz)):
            rb = rays_f[b].reshape(3, -1)[:, ::stride]
            x, y, z = rb.unbind(dim=0)
            valid = torch.isfinite(rb).all(dim=0) & (z > 1e-4)
            if int(valid.sum().item()) < 16:
                fx = fy = float(max(height, width))
                cx = (float(width) - 1.0) * 0.5
                cy = (float(height) - 1.0) * 0.5
            else:
                z_valid = z[valid]
                xz = x[valid] / z_valid
                yz = y[valid] / z_valid
                ones = torch.ones_like(xz)
                sol_x = _solve_linear_2param(torch.stack([xz, ones], dim=1), uu_s[valid])
                sol_y = _solve_linear_2param(torch.stack([yz, ones], dim=1), vv_s[valid])
                fx = float(torch.clamp(sol_x[0], min=float(min_focal_px)).item())
                fy = float(torch.clamp(sol_y[0], min=float(min_focal_px)).item())
                cx = float(sol_x[1].item())
                cy = float(sol_y[1].item())
            out[b, 0, 0] = fx
            out[b, 1, 1] = fy
            out[b, 0, 2] = cx
            out[b, 1, 2] = cy
        return out


def scale_pinhole_intrinsics(
    intrinsics: torch.Tensor,
    *,
    src_hw: tuple[int, int],
    dst_hw: tuple[int, int],
) -> torch.Tensor:
    if tuple(int(x) for x in src_hw) == tuple(int(x) for x in dst_hw):
        return intrinsics
    src_h, src_w = int(src_hw[0]), int(src_hw[1])
    dst_h, dst_w = int(dst_hw[0]), int(dst_hw[1])
    return scale_intrinsics_align_corners_false(
        intrinsics,
        sx=float(dst_w) / float(max(src_w, 1)),
        sy=float(dst_h) / float(max(src_h, 1)),
    )


def fit_fisheye624_params_from_rays(
    rays: torch.Tensor,
    *,
    min_focal_px: float = 1.0,
    max_samples: int = 65536,
) -> torch.Tensor:
    if rays.ndim != 4 or int(rays.shape[1]) != 3:
        raise ValueError(f"Expected rays shape (B,3,H,W), got {tuple(rays.shape)}")
    with torch.no_grad():
        rays_f = F.normalize(rays.detach().to(dtype=torch.float32), dim=1, eps=1e-6)
        bsz, _, height, width = rays_f.shape
        uu, vv = _pixel_grid(height, width, device=rays_f.device, dtype=rays_f.dtype)
        stride = max(1, int((height * width + int(max_samples) - 1) // int(max_samples)))
        uu_s = uu[::stride]
        vv_s = vv[::stride]
        out = torch.zeros((bsz, 16), device=rays_f.device, dtype=rays_f.dtype)
        for b in range(int(bsz)):
            rb = rays_f[b].reshape(3, -1)[:, ::stride]
            x, y, z = rb.unbind(dim=0)
            xy_norm = torch.sqrt(x.square() + y.square()).clamp(min=1e-8)
            theta = torch.atan2(xy_norm, z.clamp(min=1e-8))
            dir_x = x / xy_norm
            dir_y = y / xy_norm
            xd = theta * dir_x
            yd = theta * dir_y
            valid = torch.isfinite(rb).all(dim=0) & torch.isfinite(xd) & torch.isfinite(yd) & (z > 1e-4)
            if int(valid.sum().item()) < 16:
                fx = fy = float(max(height, width)) * 0.5
                cx = (float(width) - 1.0) * 0.5
                cy = (float(height) - 1.0) * 0.5
                coeffs = torch.zeros(4, device=rays_f.device, dtype=rays_f.dtype)
            else:
                ones = torch.ones_like(xd[valid])
                sol_x = _solve_linear_2param(torch.stack([xd[valid], ones], dim=1), uu_s[valid])
                sol_y = _solve_linear_2param(torch.stack([yd[valid], ones], dim=1), vv_s[valid])
                fx_t = torch.clamp(sol_x[0], min=float(min_focal_px))
                fy_t = torch.clamp(sol_y[0], min=float(min_focal_px))
                cx_t = sol_x[1]
                cy_t = sol_y[1]

                x_img = (uu_s[valid] - cx_t) / fx_t
                y_img = (vv_s[valid] - cy_t) / fy_t
                rho_obs = x_img * dir_x[valid] + y_img * dir_y[valid]
                theta_v = theta[valid]
                coeff_basis = torch.stack([theta_v.pow(3 + i * 2) for i in range(4)], dim=1)
                coeff_valid = torch.isfinite(rho_obs) & torch.isfinite(coeff_basis).all(dim=1) & (theta_v > 1e-6)
                if int(coeff_valid.sum().item()) >= 16:
                    coeffs = _solve_linear(
                        coeff_basis[coeff_valid],
                        (rho_obs - theta_v)[coeff_valid],
                        ridge=1e-4,
                    ).clamp(min=-10.0, max=10.0)
                else:
                    coeffs = torch.zeros(4, device=rays_f.device, dtype=rays_f.dtype)

                theta_dist = theta + sum(coeffs[i] * theta.pow(3 + i * 2) for i in range(4))
                xd_refit = theta_dist * dir_x
                yd_refit = theta_dist * dir_y
                refit_valid = valid & torch.isfinite(xd_refit) & torch.isfinite(yd_refit)
                if int(refit_valid.sum().item()) >= 16:
                    ones_refit = torch.ones_like(xd_refit[refit_valid])
                    sol_x = _solve_linear_2param(
                        torch.stack([xd_refit[refit_valid], ones_refit], dim=1),
                        uu_s[refit_valid],
                    )
                    sol_y = _solve_linear_2param(
                        torch.stack([yd_refit[refit_valid], ones_refit], dim=1),
                        vv_s[refit_valid],
                    )
                    fx_t = torch.clamp(sol_x[0], min=float(min_focal_px))
                    fy_t = torch.clamp(sol_y[0], min=float(min_focal_px))
                    cx_t = sol_x[1]
                    cy_t = sol_y[1]
                fx = float(fx_t.item())
                fy = float(fy_t.item())
                cx = float(cx_t.item())
                cy = float(cy_t.item())
            out[b, 0] = fx
            out[b, 1] = fy
            out[b, 2] = cx
            out[b, 3] = cy
            out[b, 4:8] = coeffs
        return out


def scale_fisheye624_params(
    params: torch.Tensor,
    *,
    src_hw: tuple[int, int],
    dst_hw: tuple[int, int],
) -> torch.Tensor:
    if tuple(int(x) for x in src_hw) == tuple(int(x) for x in dst_hw):
        return params
    src_h, src_w = int(src_hw[0]), int(src_hw[1])
    dst_h, dst_w = int(dst_hw[0]), int(dst_hw[1])
    out = params.clone()
    sx = float(dst_w) / float(max(src_w, 1))
    sy = float(dst_h) / float(max(src_h, 1))
    out[:, 0] *= sx
    out[:, 1] *= sy
    out[:, 2] = (out[:, 2] + 0.5) * sx - 0.5
    out[:, 3] = (out[:, 3] + 0.5) * sy - 0.5
    return out

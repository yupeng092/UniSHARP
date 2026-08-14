from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from unisharp.utils.color_space import linearRGB2sRGB
from unisharp.utils.io import save_image
from unisharp.utils.vis import colorize_alpha, colorize_scalar_map


def _to_u8_hwc(img_chw: torch.Tensor) -> np.ndarray:
    if img_chw.dtype == torch.uint8:
        return img_chw.permute(1, 2, 0).detach().cpu().numpy()
    x = img_chw.detach().to(torch.float32).clamp(0, 1)
    return (x * 255.0).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()


def _concat_grid(rows: list[list[np.ndarray]], pad: int = 6, pad_value: int = 0) -> np.ndarray:
    row_imgs = []
    for r in rows:
        padded = []
        for i, im in enumerate(r):
            padded.append(im)
            if i != len(r) - 1 and pad > 0:
                padded.append(np.full((im.shape[0], pad, 3), pad_value, dtype=np.uint8))
        row_imgs.append(np.concatenate(padded, axis=1))
    padded_rows = []
    for i, im in enumerate(row_imgs):
        padded_rows.append(im)
        if i != len(row_imgs) - 1 and pad > 0:
            padded_rows.append(np.full((pad, im.shape[1], 3), pad_value, dtype=np.uint8))
    return np.concatenate(padded_rows, axis=0)


def _pose_to_text(pose_w2c: torch.Tensor | None) -> str:
    if pose_w2c is None:
        return "None"
    p = pose_w2c.detach().to(torch.float32).cpu()
    if p.ndim == 3:
        p = p[0]
    p = p[:3, :4]
    rows = []
    for r in range(3):
        vals = [f"{float(v):+.3f}" for v in p[r].tolist()]
        rows.append("[" + ",".join(vals) + "]")
    return " ".join(rows)


def _append_text_header(image: np.ndarray, lines: list[str]) -> np.ndarray:
    if len(lines) == 0:
        return image
    h, w = image.shape[:2]
    line_h = 16
    header_h = 6 + line_h * len(lines)
    canvas = np.zeros((h + header_h, w, 3), dtype=np.uint8)
    canvas[header_h:, :, :] = image
    pil_img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_img)
    for i, txt in enumerate(lines):
        draw.text((6, 3 + i * line_h), txt, fill=(255, 255, 255))
    return np.asarray(pil_img)


def _range_from(depth_list: list[torch.Tensor | None]) -> tuple[float, float]:
    vals = []
    for d in depth_list:
        if d is None:
            continue
        valid = d[torch.isfinite(d) & (d > 0.0)]
        if valid.numel() > 8:
            vals.append(valid)
    if len(vals) == 0:
        return (0.0, 10.0)
    vv = torch.cat(vals, dim=0)
    vmin = float(torch.quantile(vv, 0.01).item())
    vmax = float(torch.quantile(vv, 0.99).item())
    vmin = max(0.0, vmin)
    vmax = max(vmin + 1e-3, vmax)
    return (vmin, vmax)


def _depth_u8_or_blank(
    depth: torch.Tensor | None,
    val_min: float,
    val_max: float,
    blank: np.ndarray,
    *,
    mask_invalid_black: bool,
) -> np.ndarray:
    if depth is None:
        return blank
    valid = torch.isfinite(depth) & (depth > 0.0)
    if int(valid.sum().item()) < 8:
        return blank
    valid_vals = depth[valid]
    fill_val = float(torch.quantile(valid_vals, 0.5).item()) if valid_vals.numel() > 0 else float(val_min)
    depth_clean = torch.where(valid, depth, torch.full_like(depth, fill_val))
    depth_clean = depth_clean.clamp(min=float(val_min), max=float(val_max))
    depth_u8 = _to_u8_hwc(colorize_scalar_map(depth_clean[0, 0], val_min=val_min, val_max=val_max, color_map="turbo"))
    if mask_invalid_black:
        valid_2d = valid[0, 0].detach().cpu().numpy()
        depth_u8[~valid_2d] = 0
    return depth_u8


def _to_face_u8_list(cube_img: torch.Tensor, face_count: int = 6) -> list[np.ndarray]:
    x = cube_img
    if x.ndim == 5 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 4:
        return []
    faces = []
    if x.shape[0] == face_count and x.shape[1] == 3:
        for i in range(face_count):
            faces.append(_to_u8_hwc(x[i]))
    elif x.shape[0] == face_count and x.shape[-1] == 3:
        for i in range(face_count):
            xi = x[i].permute(2, 0, 1).contiguous()
            faces.append(_to_u8_hwc(xi))
    return faces


def _make_cube_rows(
    src_cube_gt_u8: torch.Tensor | None,
    src_cube_pred_linear: torch.Tensor | None,
    src_cube_alpha: torch.Tensor | None,
    tgt_cube_gt_u8: torch.Tensor | None,
    tgt_cube_pred_linear: torch.Tensor | None,
    tgt_cube_alpha: torch.Tensor | None,
) -> list[list[np.ndarray]] | None:
    if src_cube_gt_u8 is None or src_cube_pred_linear is None or src_cube_alpha is None:
        return None
    if tgt_cube_gt_u8 is None or tgt_cube_pred_linear is None or tgt_cube_alpha is None:
        return None

    src_gt_faces = _to_face_u8_list(src_cube_gt_u8)
    tgt_gt_faces = _to_face_u8_list(tgt_cube_gt_u8)
    if len(src_gt_faces) != 6 or len(tgt_gt_faces) != 6:
        return None

    src_pred = linearRGB2sRGB(
        (src_cube_pred_linear / src_cube_alpha.clamp(min=1e-4)).clamp(0.0, 1.0)
    ).clamp(0.0, 1.0)
    tgt_pred = linearRGB2sRGB(
        (tgt_cube_pred_linear / tgt_cube_alpha.clamp(min=1e-4)).clamp(0.0, 1.0)
    ).clamp(0.0, 1.0)
    src_pred_faces = _to_face_u8_list(src_pred)
    tgt_pred_faces = _to_face_u8_list(tgt_pred)
    if len(src_pred_faces) != 6 or len(tgt_pred_faces) != 6:
        return None

    src_gt_f = torch.stack(
        [torch.from_numpy(x).permute(2, 0, 1).to(torch.float32) / 255.0 for x in src_gt_faces],
        dim=0,
    )
    tgt_gt_f = torch.stack(
        [torch.from_numpy(x).permute(2, 0, 1).to(torch.float32) / 255.0 for x in tgt_gt_faces],
        dim=0,
    )
    src_err = (src_pred.detach().cpu() - src_gt_f).abs().mean(dim=1, keepdim=True)
    tgt_err = (tgt_pred.detach().cpu() - tgt_gt_f).abs().mean(dim=1, keepdim=True)
    vmax = float(
        max(
            1e-3,
            min(float(torch.quantile(torch.cat([src_err.flatten(), tgt_err.flatten()]), 0.99).item()), 0.5),
        )
    )
    src_err_faces = [_to_u8_hwc(colorize_scalar_map(src_err[i, 0], val_min=0.0, val_max=vmax, color_map="turbo")) for i in range(6)]
    tgt_err_faces = [_to_u8_hwc(colorize_scalar_map(tgt_err[i, 0], val_min=0.0, val_max=vmax, color_map="turbo")) for i in range(6)]

    return [
        src_gt_faces,
        src_pred_faces,
        src_err_faces,
        tgt_gt_faces,
        tgt_pred_faces,
        tgt_err_faces,
    ]


def save_pair_visualization(
    out_file: Path,
    *,
    src_gt: torch.Tensor,
    src_pred: torch.Tensor,
    src_alpha: torch.Tensor,
    tgt_gt: torch.Tensor,
    tgt_pred: torch.Tensor,
    tgt_alpha: torch.Tensor,
    src_gt_depth: torch.Tensor | None = None,
    tgt_gt_depth: torch.Tensor | None = None,
    src_pred_depth: torch.Tensor | None = None,
    tgt_pred_depth: torch.Tensor | None = None,
    src_unik3d_depth: torch.Tensor | None = None,
    tgt_unik3d_depth: torch.Tensor | None = None,
    dataset_name: str | None = None,
    scene: str | None = None,
    step: int | None = None,
    src_idx: int | None = None,
    tgt_idx: int | None = None,
    src_pose_w2c: torch.Tensor | None = None,
    tgt_pose_w2c: torch.Tensor | None = None,
    src_cube_gt_u8: torch.Tensor | None = None,
    src_cube_pred_linear: torch.Tensor | None = None,
    src_cube_alpha: torch.Tensor | None = None,
    tgt_cube_gt_u8: torch.Tensor | None = None,
    tgt_cube_pred_linear: torch.Tensor | None = None,
    tgt_cube_alpha: torch.Tensor | None = None,
) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)

    src_a = src_alpha.clamp(0.0, 1.0)
    tgt_a = tgt_alpha.clamp(0.0, 1.0)
    src_vis_lin = (src_pred / src_a.clamp(min=1e-4)).clamp(0.0, 1.0)
    tgt_vis_lin = (tgt_pred / tgt_a.clamp(min=1e-4)).clamp(0.0, 1.0)
    src_vis = linearRGB2sRGB(src_vis_lin).clamp(0.0, 1.0)
    tgt_vis = linearRGB2sRGB(tgt_vis_lin).clamp(0.0, 1.0)
    src_err = (src_vis - src_gt).abs().mean(dim=1, keepdim=True)
    tgt_err = (tgt_vis - tgt_gt).abs().mean(dim=1, keepdim=True)
    vmax = float(
        max(
            1e-3,
            min(float(torch.quantile(torch.cat([src_err.flatten(), tgt_err.flatten()]), 0.99).item()), 0.5),
        )
    )
    src_err_u8 = colorize_scalar_map(src_err[0, 0], val_min=0.0, val_max=vmax, color_map="turbo")
    tgt_err_u8 = colorize_scalar_map(tgt_err[0, 0], val_min=0.0, val_max=vmax, color_map="turbo")
    src_alpha_u8 = colorize_alpha(src_alpha)[0]
    tgt_alpha_u8 = colorize_alpha(tgt_alpha)[0]

    has_gt_depth = (src_gt_depth is not None) and (tgt_gt_depth is not None)
    has_render_depth = (src_pred_depth is not None) and (tgt_pred_depth is not None)
    has_unik3d_depth = (src_unik3d_depth is not None) or (tgt_unik3d_depth is not None)

    base_hwc = _to_u8_hwc(src_gt[0])
    blank = np.zeros_like(base_hwc)

    if has_gt_depth:
        gt_min, gt_max = _range_from([src_gt_depth, tgt_gt_depth])
        render_min, render_max = gt_min, gt_max
        unik_min, unik_max = gt_min, gt_max
    else:
        gt_min, gt_max = (0.0, 10.0)
        shared_min, shared_max = _range_from(
            [src_pred_depth, tgt_pred_depth, src_unik3d_depth, tgt_unik3d_depth]
        )
        render_min, render_max = shared_min, shared_max
        unik_min, unik_max = shared_min, shared_max

    src_cols = [_to_u8_hwc(src_gt[0]), _to_u8_hwc(src_vis[0]), _to_u8_hwc(src_err_u8), _to_u8_hwc(src_alpha_u8)]
    tgt_cols = [_to_u8_hwc(tgt_gt[0]), _to_u8_hwc(tgt_vis[0]), _to_u8_hwc(tgt_err_u8), _to_u8_hwc(tgt_alpha_u8)]
    if has_gt_depth:
        src_cols.append(_depth_u8_or_blank(src_gt_depth, gt_min, gt_max, blank, mask_invalid_black=True))
        tgt_cols.append(_depth_u8_or_blank(tgt_gt_depth, gt_min, gt_max, blank, mask_invalid_black=True))
    if has_render_depth:
        src_cols.append(_depth_u8_or_blank(src_pred_depth, render_min, render_max, blank, mask_invalid_black=True))
        tgt_cols.append(_depth_u8_or_blank(tgt_pred_depth, render_min, render_max, blank, mask_invalid_black=True))
    if has_unik3d_depth:
        src_cols.append(_depth_u8_or_blank(src_unik3d_depth, unik_min, unik_max, blank, mask_invalid_black=False))
        tgt_cols.append(_depth_u8_or_blank(tgt_unik3d_depth, unik_min, unik_max, blank, mask_invalid_black=False))

    erp_grid = _concat_grid(rows=[src_cols, tgt_cols], pad=6, pad_value=0)
    lines = [
        f"dataset={str(dataset_name) if dataset_name is not None else 'unknown'} scene={str(scene) if scene is not None else 'unknown'} step={int(step) if step is not None else -1}",
        f"src_idx={int(src_idx) if src_idx is not None else -1} tgt_idx={int(tgt_idx) if tgt_idx is not None else -1}",
        f"src_w2c={_pose_to_text(src_pose_w2c)}",
        f"tgt_w2c={_pose_to_text(tgt_pose_w2c)}",
    ]
    grid = _append_text_header(erp_grid, lines)

    cube_rows = _make_cube_rows(
        src_cube_gt_u8=src_cube_gt_u8,
        src_cube_pred_linear=src_cube_pred_linear,
        src_cube_alpha=src_cube_alpha,
        tgt_cube_gt_u8=tgt_cube_gt_u8,
        tgt_cube_pred_linear=tgt_cube_pred_linear,
        tgt_cube_alpha=tgt_cube_alpha,
    )
    save_image(grid, out_file)
    if cube_rows is not None:
        cube_grid = _concat_grid(rows=cube_rows, pad=6, pad_value=0)
        cube_lines = lines + ["cubemap_rows=src_gt/src_pred/src_err/tgt_gt/tgt_pred/tgt_err"]
        cube_grid = _append_text_header(cube_grid, cube_lines)
        cube_file = out_file.with_name(f"{out_file.stem}_cubemap{out_file.suffix}")
        save_image(cube_grid, cube_file)


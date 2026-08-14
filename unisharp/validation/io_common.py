from __future__ import annotations

import io
import re
import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from unisharp.utils.pixel_convention import scale_intrinsics_align_corners_false


def resize_k3_align_corners_false(k3: torch.Tensor, *, sx: float, sy: float) -> torch.Tensor:
    return scale_intrinsics_align_corners_false(k3, sx=float(sx), sy=float(sy))


def resize_rgb_depth_k_fit_box_no_pad(
    rgb_u8_bchw: torch.Tensor,
    k3_b33: torch.Tensor,
    *,
    max_h: int,
    max_w: int,
    depth_b1hw: torch.Tensor | None = None,
    size_divisor: int = 14,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    old_h, old_w = int(rgb_u8_bchw.shape[-2]), int(rgb_u8_bchw.shape[-1])
    if old_h <= int(max_h) and old_w <= int(max_w):
        return rgb_u8_bchw, k3_b33, depth_b1hw
    max_h = max(1, int(max_h))
    max_w = max(1, int(max_w))
    size_divisor = max(1, int(size_divisor))
    scale_h = float(max_h) / float(old_h)
    scale_w = float(max_w) / float(old_w)
    if scale_h <= scale_w:
        new_h = max(size_divisor, (max_h // size_divisor) * size_divisor)
        new_w_raw = int(round(float(old_w) * float(new_h) / float(old_h)))
        new_w = max(size_divisor, min(max_w, (new_w_raw // size_divisor) * size_divisor))
    else:
        new_w = max(size_divisor, (max_w // size_divisor) * size_divisor)
        new_h_raw = int(round(float(old_h) * float(new_w) / float(old_w)))
        new_h = max(size_divisor, min(max_h, (new_h_raw // size_divisor) * size_divisor))
    new_h = max(1, min(max_h, new_h))
    new_w = max(1, min(max_w, new_w))
    if new_h == old_h and new_w == old_w:
        return rgb_u8_bchw, k3_b33, depth_b1hw
    rgb_resized = F.interpolate(
        rgb_u8_bchw.float(),
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
    ).round().clamp(0, 255).to(torch.uint8)
    sx = float(new_w) / float(old_w)
    sy = float(new_h) / float(old_h)
    k3_out = resize_k3_align_corners_false(k3_b33, sx=sx, sy=sy)
    depth_out = None if depth_b1hw is None else F.interpolate(depth_b1hw, size=(new_h, new_w), mode="nearest")
    return rgb_resized, k3_out, depth_out


def decode_rgb_u8(jpeg_bytes_tensor: torch.Tensor) -> torch.Tensor:
    img_bytes = jpeg_bytes_tensor.detach().cpu().numpy().tobytes()
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return torch.from_numpy(np.array(pil)).permute(2, 0, 1).contiguous().to(torch.uint8)


def torch_load_any(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def pseudo_depth_safe_key(value: Any) -> str:
    text = str(value).strip().replace("\\", "__").replace("/", "__")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text[:180] if text else "unknown"


def validation_pseudo_depth_path(root: Path, dataset: str, scene: Any, frame_idx: Any) -> Path:
    if isinstance(frame_idx, int):
        frame_key = f"{int(frame_idx):05d}"
    elif torch.is_tensor(frame_idx) and frame_idx.numel() == 1:
        frame_key = f"{int(frame_idx.item()):05d}"
    else:
        frame_key = pseudo_depth_safe_key(frame_idx)
    return Path(root) / str(dataset) / pseudo_depth_safe_key(scene) / f"{frame_key}.npz"


def normalize_depth_kind(value: Any, *, default: str = "distance") -> str:
    text = str(value).strip().lower().replace("-", "_")
    if text in {"zdepth", "z_depth", "z"}:
        return "zdepth"
    if text in {"distance", "dist", "radial", "ray_distance"}:
        return "distance"
    return str(default)


def distance_to_z_depth_pinhole(distance_1hw: torch.Tensor, intrinsics_k3: torch.Tensor) -> torch.Tensor:
    if distance_1hw.ndim == 2:
        distance_1hw = distance_1hw.unsqueeze(0)
    if distance_1hw.ndim != 3 or int(distance_1hw.shape[0]) != 1:
        raise ValueError(f"Expected distance shape (1,H,W), got {tuple(distance_1hw.shape)}")
    d = distance_1hw.to(torch.float32)
    h, w = int(d.shape[-2]), int(d.shape[-1])
    k = intrinsics_k3.to(dtype=torch.float32, device=d.device)
    ys = torch.arange(h, device=d.device, dtype=torch.float32)
    xs = torch.arange(w, device=d.device, dtype=torch.float32)
    vv, uu = torch.meshgrid(ys, xs, indexing="ij")
    x = (uu - k[0, 2]) / k[0, 0].clamp(min=1e-6)
    y = (vv - k[1, 2]) / k[1, 1].clamp(min=1e-6)
    ray_z = 1.0 / torch.sqrt(x * x + y * y + 1.0).clamp(min=1e-8)
    return (d[0] * ray_z).unsqueeze(0)


def load_validation_pseudo_depth(
    root: Path | None,
    dataset: str,
    scene: Any,
    frame_idx: Any,
    intrinsics_k3: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if root is None:
        return None
    path = validation_pseudo_depth_path(Path(root), dataset, scene, frame_idx)
    if not path.exists():
        legacy_path = path.with_suffix(".pt")
        if not legacy_path.exists():
            return None
        path = legacy_path
    try:
        if path.suffix.lower() == ".npz":
            with np.load(path, allow_pickle=False) as payload_np:
                if "z_depth_m" in payload_np:
                    depth = torch.from_numpy(np.array(payload_np["z_depth_m"], copy=True))
                    depth_kind = "zdepth"
                elif "depth_m" in payload_np:
                    depth = torch.from_numpy(np.array(payload_np["depth_m"], copy=True))
                    depth_kind_raw = payload_np["depth_kind"] if "depth_kind" in payload_np else "distance"
                    depth_kind = normalize_depth_kind(
                        depth_kind_raw.tolist() if hasattr(depth_kind_raw, "tolist") else depth_kind_raw,
                        default="distance",
                    )
                elif "distance_m" in payload_np:
                    depth = torch.from_numpy(np.array(payload_np["distance_m"], copy=True))
                    depth_kind = "distance"
                else:
                    return None
        else:
            payload = torch_load_any(path)
            if isinstance(payload, dict):
                depth = payload.get("z_depth_m", None)
                if not torch.is_tensor(depth):
                    depth = payload.get("depth_m", None)
                    depth_kind = normalize_depth_kind(payload.get("depth_kind", "distance"), default="distance")
                else:
                    depth_kind = "zdepth"
            else:
                depth = payload
                depth_kind = "distance"
        if torch.is_tensor(depth):
            pass
        elif isinstance(depth, np.ndarray):
            depth = torch.from_numpy(depth)
        else:
            return None
        if depth.ndim == 2:
            depth = depth.unsqueeze(0)
        if depth.ndim == 3 and int(depth.shape[0]) == 1:
            depth = depth.to(torch.float32)
        else:
            return None
        if normalize_depth_kind(depth_kind, default="distance") != "zdepth":
            if intrinsics_k3 is None:
                return None
            depth = distance_to_z_depth_pinhole(depth, intrinsics_k3=intrinsics_k3)
        valid = torch.isfinite(depth) & (depth > 0.0)
        if int(valid.sum().item()) <= 0:
            return None
        return torch.where(valid, depth, torch.zeros_like(depth))
    except Exception:
        return None


def load_validation_pseudo_distance(
    root: Path | None,
    dataset: str,
    scene: Any,
    frame_idx: Any,
) -> torch.Tensor | None:
    if root is None:
        return None
    path = validation_pseudo_depth_path(Path(root), dataset, scene, frame_idx)
    if not path.exists():
        legacy_path = path.with_suffix(".pt")
        if not legacy_path.exists():
            return None
        path = legacy_path
    try:
        if path.suffix.lower() == ".npz":
            with np.load(path, allow_pickle=False) as payload_np:
                if "distance_m" in payload_np:
                    depth = torch.from_numpy(np.array(payload_np["distance_m"], copy=True))
                elif "depth_m" in payload_np:
                    depth_kind_raw = payload_np["depth_kind"] if "depth_kind" in payload_np else "distance"
                    depth_kind = normalize_depth_kind(
                        depth_kind_raw.tolist() if hasattr(depth_kind_raw, "tolist") else depth_kind_raw,
                        default="distance",
                    )
                    if depth_kind != "distance":
                        return None
                    depth = torch.from_numpy(np.array(payload_np["depth_m"], copy=True))
                else:
                    return None
        else:
            payload = torch_load_any(path)
            if isinstance(payload, dict):
                depth = payload.get("distance_m", None)
                if not torch.is_tensor(depth):
                    depth = payload.get("depth_m", None)
                    if normalize_depth_kind(payload.get("depth_kind", "distance"), default="distance") != "distance":
                        return None
            else:
                depth = payload
        if isinstance(depth, np.ndarray):
            depth = torch.from_numpy(depth)
        if not torch.is_tensor(depth):
            return None
        if depth.ndim == 2:
            depth = depth.unsqueeze(0)
        if depth.ndim != 3 or int(depth.shape[0]) != 1:
            return None
        depth = depth.to(torch.float32)
        valid = torch.isfinite(depth) & (depth > 0.0)
        if int(valid.sum().item()) <= 0:
            return None
        return torch.where(valid, depth, torch.zeros_like(depth))
    except Exception:
        return None


def load_png_rgb_u8(path: Path) -> torch.Tensor:
    arr = np.array(Image.open(path))
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected RGB image at {path}, got shape={arr.shape}")
    return torch.from_numpy(arr.astype(np.uint8)).permute(2, 0, 1).contiguous()


def load_png_depth_m(path: Path) -> torch.Tensor:
    dep = torch.from_numpy(np.array(Image.open(path))).to(torch.float32)
    if float(dep.max().item()) > 200.0:
        dep = dep / 1000.0
    return dep.unsqueeze(0)


def resolve_replica_test_root(root: Path) -> Path:
    root = Path(root)

    def _looks_like_replica_test_dir(path: Path) -> bool:
        if not path.exists() or not path.is_dir():
            return False
        for child in path.iterdir():
            if child.is_dir() and (child / "pano").exists() and (
                (child / "rotation.npy").exists() or (child / "meta.pt").exists()
            ):
                return True
        return False

    candidates = [
        root / "replica_dataset" / "test",
        root / "test",
        root,
    ]
    for candidate in candidates:
        if _looks_like_replica_test_dir(candidate):
            return candidate
    return candidates[0] if candidates[0].exists() else root


def read_manifest_lines(path: Path | None, max_lines: int = 0) -> list[str]:
    if path is None:
        return []
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return lines[: int(max_lines)] if int(max_lines) > 0 else lines


def colmap_scene_roots(root: Path) -> list[Path]:
    if (root / "images_4").exists() or (root / "images").exists() or (root / "poses_bounds.npy").exists() or (root / "sparse" / "0").exists():
        return [root]
    return sorted(
        [
            p
            for p in root.iterdir()
            if p.is_dir() and ((p / "images_4").exists() or (p / "images").exists() or (p / "poses_bounds.npy").exists() or (p / "sparse" / "0").exists())
        ]
    ) if root.exists() else []


def colmap_image_dir(scene_root: Path) -> Path:
    if (scene_root / "images_4").exists():
        return scene_root / "images_4"
    if (scene_root / "images").exists():
        return scene_root / "images"
    return scene_root


def load_hm3d_pose(scene_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    rot_path = scene_dir / "rotation.npy"
    trans_path = scene_dir / "translation.npy"
    if rot_path.exists() and trans_path.exists():
        return np.load(rot_path).astype(np.float32), np.load(trans_path).astype(np.float32)
    meta = torch_load_any(scene_dir / "meta.pt")
    if isinstance(meta, dict) and ("R" in meta) and ("t" in meta):
        return np.asarray(meta["R"], dtype=np.float32), np.asarray(meta["t"], dtype=np.float32)
    cams = meta["cameras"].to(torch.float32)
    return cams[:, :3, :3].cpu().numpy(), cams[:, :3, 3].cpu().numpy()


def nerf_c2w_to_opencv_c2w(c2w_in: torch.Tensor | np.ndarray) -> torch.Tensor:
    c2w = torch.as_tensor(c2w_in, dtype=torch.float32).clone()
    if tuple(c2w.shape) != (4, 4):
        raise ValueError(f"Expected c2w shape (4,4), got {tuple(c2w.shape)}")
    c2w[:3, 1:3] *= -1.0
    return c2w


def colmap_pose_scale_from_bounds(root: Path, bd_factor: float = 0.75) -> float:
    poses_bounds_path = root / "poses_bounds.npy"
    if not poses_bounds_path.exists():
        return 1.0
    poses_bounds = np.load(poses_bounds_path)
    if poses_bounds.ndim != 2 or poses_bounds.shape[1] < 17:
        return 1.0
    bounds = np.asarray(poses_bounds[:, -2:], dtype=np.float32)
    finite = bounds[np.isfinite(bounds)]
    positive = finite[finite > 0.0]
    if positive.size == 0:
        return 1.0
    min_bound = float(positive.min())
    return 1.0 if min_bound <= 0.0 else 1.0 / max(min_bound * float(bd_factor), 1e-6)


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    q = np.asarray(qvec, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError(f"Expected qvec shape (4,), got {tuple(q.shape)}")
    w, x, y, z = q.tolist()
    return np.array(
        [
            [1.0 - 2.0 * y * y - 2.0 * z * z, 2.0 * x * y - 2.0 * w * z, 2.0 * x * z + 2.0 * w * y],
            [2.0 * x * y + 2.0 * w * z, 1.0 - 2.0 * x * x - 2.0 * z * z, 2.0 * y * z - 2.0 * w * x],
            [2.0 * x * z - 2.0 * w * y, 2.0 * y * z + 2.0 * w * x, 1.0 - 2.0 * x * x - 2.0 * y * y],
        ],
        dtype=np.float32,
    )


_COLMAP_CAMERA_MODEL_NUM_PARAMS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


def _colmap_camera_to_k(camera: dict[str, float | int | str]) -> torch.Tensor:
    model = str(camera["model"])
    params = list(camera["params"])  # type: ignore[arg-type]
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):
        fx = fy = float(params[0])
        cx = float(params[1])
        cy = float(params[2])
    elif model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "FOV", "THIN_PRISM_FISHEYE"):
        fx = float(params[0])
        fy = float(params[1])
        cx = float(params[2])
        cy = float(params[3])
    else:
        raise ValueError(f"Unsupported COLMAP camera model: {model}")
    k = torch.eye(3, dtype=torch.float32)
    k[0, 0] = fx
    k[1, 1] = fy
    k[0, 2] = cx
    k[1, 2] = cy
    return k


def _read_cameras_txt(cameras_txt: Path) -> dict[int, dict[str, float | int | str | list[float]]]:
    cameras: dict[int, dict[str, float | int | str | list[float]]] = {}
    for raw in cameras_txt.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if (not line) or line.startswith("#"):
            continue
        parts = line.split()
        camera_id = int(parts[0])
        cameras[camera_id] = {
            "model": str(parts[1]),
            "width": int(parts[2]),
            "height": int(parts[3]),
            "params": [float(x) for x in parts[4:]],
        }
    return cameras


def _read_cameras_bin(cameras_bin: Path) -> dict[int, dict[str, float | int | str | list[float]]]:
    cameras: dict[int, dict[str, float | int | str | list[float]]] = {}
    data = cameras_bin.read_bytes()
    offset = 0
    (num_cameras,) = struct.unpack_from("<Q", data, offset)
    offset += 8
    for _ in range(int(num_cameras)):
        camera_id, model_id, width, height = struct.unpack_from("<iiQQ", data, offset)
        offset += struct.calcsize("<iiQQ")
        model_name, num_params = _COLMAP_CAMERA_MODEL_NUM_PARAMS[int(model_id)]
        params = list(struct.unpack_from("<" + "d" * int(num_params), data, offset))
        offset += 8 * int(num_params)
        cameras[int(camera_id)] = {
            "model": model_name,
            "width": int(width),
            "height": int(height),
            "params": [float(x) for x in params],
        }
    return cameras


def _colmap_entry_for_image(
    *,
    image_id: int,
    qvec: np.ndarray,
    tvec: np.ndarray,
    camera_id: int,
    camera: dict[str, float | int | str | list[float]],
    pose_scale: float,
) -> dict[str, torch.Tensor | int | str | list[float]]:
    w2c = torch.eye(4, dtype=torch.float32)
    w2c[:3, :3] = torch.from_numpy(qvec2rotmat(qvec))
    w2c[:3, 3] = torch.from_numpy(tvec.astype(np.float32) * float(pose_scale))
    return {
        "image_id": int(image_id),
        "camera_id": int(camera_id),
        "camera_model": str(camera["model"]),
        "camera_params": [float(x) for x in camera["params"]],  # type: ignore[index]
        "w2c": w2c,
        "k": _colmap_camera_to_k(camera),
        "width": int(camera["width"]),
        "height": int(camera["height"]),
    }


def _read_images_txt(
    images_txt: Path,
    cameras: dict[int, dict[str, float | int | str | list[float]]],
    pose_scale: float,
) -> dict[str, dict[str, torch.Tensor | int | str | list[float]]]:
    entries: dict[str, dict[str, torch.Tensor | int | str | list[float]]] = {}
    lines = images_txt.read_text(encoding="utf-8").splitlines()
    line_idx = 0
    while line_idx < len(lines):
        line = lines[line_idx].strip()
        line_idx += 1
        if (not line) or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        qvec = np.array([float(x) for x in parts[1:5]], dtype=np.float64)
        tvec = np.array([float(x) for x in parts[5:8]], dtype=np.float32)
        camera_id = int(parts[8])
        image_name = str(parts[9])
        entries[image_name] = _colmap_entry_for_image(
            image_id=-1,
            qvec=qvec,
            tvec=tvec,
            camera_id=camera_id,
            camera=cameras[camera_id],
            pose_scale=float(pose_scale),
        )
        if line_idx < len(lines):
            line_idx += 1
    return entries


def _read_images_bin(
    images_bin: Path,
    cameras: dict[int, dict[str, float | int | str | list[float]]],
    pose_scale: float,
) -> dict[str, dict[str, torch.Tensor | int | str | list[float]]]:
    entries: dict[str, dict[str, torch.Tensor | int | str | list[float]]] = {}
    data = images_bin.read_bytes()
    offset = 0
    (num_images,) = struct.unpack_from("<Q", data, offset)
    offset += 8
    for _ in range(int(num_images)):
        image_id = struct.unpack_from("<i", data, offset)[0]
        offset += 4
        qvec = np.array(struct.unpack_from("<dddd", data, offset), dtype=np.float64)
        offset += 32
        tvec = np.array(struct.unpack_from("<ddd", data, offset), dtype=np.float32)
        offset += 24
        camera_id = struct.unpack_from("<i", data, offset)[0]
        offset += 4
        name_end = data.index(b"\x00", offset)
        image_name = data[offset:name_end].decode("utf-8")
        offset = name_end + 1
        (num_points2d,) = struct.unpack_from("<Q", data, offset)
        offset += 8 + int(num_points2d) * struct.calcsize("<ddq")
        entries[image_name] = _colmap_entry_for_image(
            image_id=int(image_id),
            qvec=qvec,
            tvec=tvec,
            camera_id=int(camera_id),
            camera=cameras[int(camera_id)],
            pose_scale=float(pose_scale),
        )
    return entries


def load_colmap_entries(root: Path, *, pose_scale: float = 1.0) -> dict[str, dict[str, torch.Tensor | int | str | list[float]]] | None:
    sparse_dir = root / "sparse" / "0"
    cameras_txt = sparse_dir / "cameras.txt"
    images_txt = sparse_dir / "images.txt"
    cameras_bin = sparse_dir / "cameras.bin"
    images_bin = sparse_dir / "images.bin"
    if cameras_txt.exists() and images_txt.exists():
        cameras = _read_cameras_txt(cameras_txt)
        return _read_images_txt(images_txt, cameras, pose_scale=float(pose_scale))
    if cameras_bin.exists() and images_bin.exists():
        cameras = _read_cameras_bin(cameras_bin)
        return _read_images_bin(images_bin, cameras, pose_scale=float(pose_scale))
    return None


def load_scaled_colmap_entries(root: Path) -> dict[str, dict[str, torch.Tensor | int | str | list[float]]] | None:
    return load_colmap_entries(root, pose_scale=float(colmap_pose_scale_from_bounds(root)))


def wild_validation_roots(data_root: Path) -> list[Path]:
    if data_root.is_file():
        return [Path(line.strip()) for line in data_root.read_text(encoding="utf-8").splitlines() if line.strip()]
    if (data_root / "scenes").is_dir():
        return [data_root]
    roots: list[Path] = []
    if data_root.is_dir():
        for child in sorted([p for p in data_root.iterdir() if p.is_dir()]):
            if (child / "scenes").is_dir():
                roots.append(child)
            elif (child / child.name / "scenes").is_dir():
                roots.append(child / child.name)
    if roots:
        return roots
    return [data_root]

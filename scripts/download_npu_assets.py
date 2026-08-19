from __future__ import annotations

"""Prepare public UniSHARP / UniK3D assets on a Linux Ascend host.

This downloader intentionally has a small default scope (UniK3D weights and
UniSHARP manifests). Large training datasets are explicit options. RE10K and
DL3DV are gated by their providers, while WildRGB-D has a provider-maintained
downloader. The script invokes those official sources after access has been
granted; it never mirrors, scrapes, or bypasses their access controls.
"""

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
UNIK3D_CACHE = REPO_ROOT / "UniK3D" / "checkpoints" / "huggingface"
MANIFEST_ROOT = REPO_ROOT / "dataset_manifests"
DATA_ROOT = REPO_ROOT / "datasets"

UNIK3D_REPO_PREFIX = "lpiccinelli/unik3d-"
UNISHARP_REPO = "Insta360-Research/Unisharp"
OMNIROOMS_REPO = "Insta360-Research/OmniRooms"
RE10K_REPO = "RE10K/RealEstate10K"
WILDRGBD_DOWNLOADER_URL = "https://raw.githubusercontent.com/wildrgbd/wildrgbd/main/download.py"
DL3DV_DOWNLOADER_URL = "https://raw.githubusercontent.com/DL3DV-10K/Dataset/main/scripts/download.py"
MANIFEST_REPO = OMNIROOMS_REPO
MANIFEST_PATTERNS = ("manifests/train/*", "manifests/validation/*")


def _require_huggingface_hub():
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required. On the NPU host run: "
            "pip install -r requirements_npu.txt"
        ) from exc
    return hf_hub_download, snapshot_download


def _token(value: str | None) -> str | None:
    return value or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _download_unik3d(
    backbones: Iterable[str],
    *,
    token: str | None,
    dry_run: bool,
) -> None:
    if dry_run:
        for backbone in backbones:
            print(f"would cache {UNIK3D_REPO_PREFIX}{backbone}/pytorch_model.bin in {UNIK3D_CACHE}")
        return
    hf_hub_download, _ = _require_huggingface_hub()
    UNIK3D_CACHE.mkdir(parents=True, exist_ok=True)
    for backbone in backbones:
        repo_id = f"{UNIK3D_REPO_PREFIX}{backbone}"
        print(f"Downloading {repo_id} …")
        path = hf_hub_download(
            repo_id=repo_id,
            filename="pytorch_model.bin",
            repo_type="model",
            cache_dir=str(UNIK3D_CACHE),
            token=token,
        )
        print(f"Cached: {path}")


def _copy_training_manifests(download_root: Path, manifest_root: Path) -> None:
    """Make both the released names and the training CLI's expected aliases available."""
    source = download_root / "manifests" / "train"
    if not source.is_dir():
        print("Warning: Hugging Face snapshot contains no manifests/train directory.")
        return
    manifest_root.mkdir(parents=True, exist_ok=True)
    for file_path in source.iterdir():
        if file_path.is_file():
            shutil.copy2(file_path, manifest_root / file_path.name)

    # The public card calls this file omnirooms.txt; this training CLI calls it
    # sim_train_scenes.txt. Keep an ordinary copy so either name can be used.
    released_name = manifest_root / "omnirooms.txt"
    cli_name = manifest_root / "sim_train_scenes.txt"
    if released_name.is_file() and not cli_name.exists():
        shutil.copy2(released_name, cli_name)


def _download_manifests(
    *,
    token: str | None,
    manifest_root: Path,
    dry_run: bool,
) -> None:
    download_root = manifest_root / ".hf_source"
    if dry_run:
        print(f"would download {MANIFEST_REPO}:{', '.join(MANIFEST_PATTERNS)} into {manifest_root}")
        return
    _, snapshot_download = _require_huggingface_hub()
    print(f"Downloading UniSHARP manifests from {MANIFEST_REPO} …")
    snapshot_download(
        repo_id=MANIFEST_REPO,
        repo_type="dataset",
        allow_patterns=list(MANIFEST_PATTERNS),
        local_dir=str(download_root),
        token=token,
    )
    _copy_training_manifests(download_root, manifest_root)
    print(f"Manifests prepared: {manifest_root}")


def _download_unisharp_checkpoints(
    *,
    token: str | None,
    target: Path,
    dry_run: bool,
) -> None:
    if dry_run:
        print(f"would download model files from {UNISHARP_REPO} into {target}")
        return
    _, snapshot_download = _require_huggingface_hub()
    print(f"Downloading released UniSHARP checkpoints from {UNISHARP_REPO} …")
    snapshot_download(
        repo_id=UNISHARP_REPO,
        repo_type="model",
        local_dir=str(target),
        token=token,
    )
    print(f"Checkpoints prepared: {target}")


def _download_omnirooms(
    *,
    token: str | None,
    target: Path,
    dry_run: bool,
) -> None:
    if dry_run:
        print(f"would download the full {OMNIROOMS_REPO} dataset into {target} (public release is about 541 GB)")
        return
    _, snapshot_download = _require_huggingface_hub()
    print(f"Downloading full OmniRooms dataset from {OMNIROOMS_REPO} …")
    snapshot_download(
        repo_id=OMNIROOMS_REPO,
        repo_type="dataset",
        local_dir=str(target),
        token=token,
    )
    print(f"Dataset prepared: {target}")


def _download_re10k(
    *,
    token: str | None,
    target: Path,
    dry_run: bool,
) -> None:
    """Download the official gated RE10K repository after terms are accepted."""
    if dry_run:
        print(f"would download gated {RE10K_REPO} into {target}; accept its Hugging Face terms first")
        return
    _, snapshot_download = _require_huggingface_hub()
    print(f"Downloading gated RE10K data from {RE10K_REPO} …")
    snapshot_download(
        repo_id=RE10K_REPO,
        repo_type="dataset",
        local_dir=str(target),
        token=token,
    )
    chunks = list((target / "train").glob("*.torch")) if (target / "train").is_dir() else []
    if not chunks:
        print(
            "Warning: downloaded RE10K is not in this repository's pre-packed train/*.torch layout. "
            "Convert it to UniSHARP RE10K chunks and update re10k_train_chunks.txt before mixed training."
        )
    else:
        print(f"RE10K chunks prepared: {target}")


def _download_official_python(
    *,
    url: str,
    target: Path,
    filename: str,
    command: list[str],
    dry_run: bool,
) -> None:
    if dry_run:
        print(f"would fetch {url} and run: {' '.join(command)}")
        return
    target.mkdir(parents=True, exist_ok=True)
    script_path = target / filename
    print(f"Fetching provider downloader: {url}")
    urllib.request.urlretrieve(url, script_path)
    subprocess.run(command, cwd=str(target), check=True)


def _write_wildrgbd_manifests(root: Path, manifest_root: Path) -> None:
    scene_dirs: list[Path] = []
    roots: set[Path] = set()
    for scenes_dir in sorted(root.rglob("scenes")):
        if not scenes_dir.is_dir():
            continue
        children = sorted(path for path in scenes_dir.iterdir() if path.is_dir())
        if not children:
            continue
        roots.add(scenes_dir.parent.resolve())
        scene_dirs.extend(child.resolve() for child in children)
    if not scene_dirs:
        print(f"Warning: no WildRGB-D scene folders found under {root}; no manifest was written.")
        return
    manifest_root.mkdir(parents=True, exist_ok=True)
    (manifest_root / "wildrgbd_roots.txt").write_text(
        "\n".join(str(path) for path in sorted(roots)) + "\n", encoding="utf-8"
    )
    (manifest_root / "wildrgbd_train_scenes.txt").write_text(
        "\n".join(str(path) for path in scene_dirs) + "\n", encoding="utf-8"
    )
    print(f"WildRGB-D manifests prepared: {manifest_root}")


def _download_wildrgbd(
    *,
    target: Path,
    manifest_root: Path,
    category: str | None,
    dry_run: bool,
) -> None:
    if not category:
        raise SystemExit("--assets wildrgbd requires --wildrgbd-category <name|all>.")
    script_path = target / "download_wildrgbd_official.py"
    command = [sys.executable, str(script_path), "--cat", str(category)]
    _download_official_python(
        url=WILDRGBD_DOWNLOADER_URL,
        target=target,
        filename=script_path.name,
        command=command,
        dry_run=dry_run,
    )
    if not dry_run:
        _write_wildrgbd_manifests(target, manifest_root)


def _write_dl3dv_manifest(rgb_root: Path, depth_root: Path, manifest_root: Path) -> None:
    """Write the exact scene-spec format consumed by ``DL3DVDataset``."""
    if not depth_root.is_dir():
        print(
            f"Warning: DL3DV depth root is missing: {depth_root}. "
            "No DL3DV training manifest was written; provide precomputed per-image depth first."
        )
        return
    rows: list[str] = []
    for bucket_dir in sorted(path for path in rgb_root.iterdir() if path.is_dir()):
        for scene_stub in sorted(path for path in bucket_dir.iterdir() if path.is_dir()):
            inner_dirs = [path for path in scene_stub.iterdir() if path.is_dir()]
            scene_dir = inner_dirs[0] if inner_dirs else scene_stub
            depth_dir = depth_root / bucket_dir.name / scene_stub.name / "exports" / "mini_npz" / "per_image"
            if (scene_dir / "transforms.json").is_file() and (scene_dir / "images_4").is_dir() and depth_dir.is_dir():
                rows.append(f"{bucket_dir.name}/{scene_stub.name}|{scene_dir.resolve()}|{depth_dir.resolve()}")
    if not rows:
        print(f"Warning: no DL3DV RGB/depth scene pairs found under {rgb_root} and {depth_root}.")
        return
    manifest_root.mkdir(parents=True, exist_ok=True)
    (manifest_root / "dl3dv_train_scenes.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"DL3DV manifest prepared: {manifest_root / 'dl3dv_train_scenes.txt'}")


def _download_dl3dv(
    *,
    target: Path,
    depth_root: Path,
    manifest_root: Path,
    subset: str | None,
    resolution: str,
    dry_run: bool,
) -> None:
    if not subset:
        raise SystemExit("--assets dl3dv requires --dl3dv-subset (for example 1K or 10K).")
    script_path = target / "download_dl3dv_official.py"
    command = [
        sys.executable,
        str(script_path),
        "--odir",
        str(target),
        "--subset",
        str(subset),
        "--resolution",
        str(resolution),
        "--file_type",
        "images+poses",
    ]
    _download_official_python(
        url=DL3DV_DOWNLOADER_URL,
        target=target,
        filename=script_path.name,
        command=command,
        dry_run=dry_run,
    )
    if not dry_run:
        _write_dl3dv_manifest(target, depth_root, manifest_root)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assets",
        nargs="+",
        choices=("unik3d", "manifests", "unisharp-checkpoints", "omnirooms", "re10k", "wildrgbd", "dl3dv"),
        default=("unik3d", "manifests"),
        help="Assets to download. Large datasets are always explicit options.",
    )
    parser.add_argument(
        "--backbones",
        nargs="+",
        choices=("vits", "vitb", "vitl"),
        default=("vits",),
        help="UniK3D pretrained backbones to cache.",
    )
    parser.add_argument("--manifest-root", type=Path, default=MANIFEST_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=REPO_ROOT / "checkpoints" / "released")
    parser.add_argument("--re10k-root", type=Path, default=DATA_ROOT / "re10k")
    parser.add_argument("--wildrgbd-root", type=Path, default=DATA_ROOT / "wildrgbd")
    parser.add_argument("--wildrgbd-category", default=None, help="Official WildRGB-D category, or all (requires about 4 TB extracted).")
    parser.add_argument("--dl3dv-root", type=Path, default=DATA_ROOT / "dl3dv")
    parser.add_argument("--dl3dv-depth-root", type=Path, default=DATA_ROOT / "dl3dv_depth", help="Existing prepared DL3DV per-image depth tree; provider RGB download does not include it.")
    parser.add_argument("--dl3dv-subset", choices=("1K", "2K", "3K", "4K", "5K", "6K", "7K", "8K", "9K", "10K"), default=None)
    parser.add_argument("--dl3dv-resolution", choices=("480P", "960P", "2K", "4K"), default="960P")
    parser.add_argument("--token", default=None, help="Hugging Face token; defaults to HF_TOKEN if set.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned downloads without network access.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if sys.platform == "win32":
        print("Warning: this script is intended to be executed on the Linux NPU host.")
    assets = set(args.assets)
    token = _token(args.token)
    if "unik3d" in assets:
        _download_unik3d(args.backbones, token=token, dry_run=bool(args.dry_run))
    if "manifests" in assets:
        _download_manifests(token=token, manifest_root=Path(args.manifest_root), dry_run=bool(args.dry_run))
    if "unisharp-checkpoints" in assets:
        _download_unisharp_checkpoints(
            token=token,
            target=Path(args.checkpoint_root),
            dry_run=bool(args.dry_run),
        )
    if "omnirooms" in assets:
        _download_omnirooms(
            token=token,
            target=Path(args.dataset_root) / "omnirooms",
            dry_run=bool(args.dry_run),
        )
    if "re10k" in assets:
        _download_re10k(token=token, target=Path(args.re10k_root), dry_run=bool(args.dry_run))
    if "wildrgbd" in assets:
        _download_wildrgbd(
            target=Path(args.wildrgbd_root),
            manifest_root=Path(args.manifest_root),
            category=args.wildrgbd_category,
            dry_run=bool(args.dry_run),
        )
    if "dl3dv" in assets:
        _download_dl3dv(
            target=Path(args.dl3dv_root),
            depth_root=Path(args.dl3dv_depth_root),
            manifest_root=Path(args.manifest_root),
            subset=args.dl3dv_subset,
            resolution=args.dl3dv_resolution,
            dry_run=bool(args.dry_run),
        )


if __name__ == "__main__":
    main()

from __future__ import annotations

"""Prepare public UniSHARP / UniK3D assets on a Linux Ascend host.

This downloader intentionally has a small default scope (UniK3D weights and
UniSHARP manifests). Large training datasets are explicit options. RE10K and
DL3DV are gated by their providers, while WildRGB-D has a provider-maintained
downloader. The script invokes those official sources after access has been
granted; it never mirrors, scrapes, or bypasses their access controls.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
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
COCO_ANNOTATIONS_URL = "https://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_IMAGE_URL_TEMPLATE = "https://images.cocodataset.org/{split}/{filename}"
WIDER_FACE_REPO = "CUHK-CSE/wider_face"
WIDER_FACE_PATTERNS = ("data/WIDER_train.zip", "data/WIDER_val.zip", "data/wider_face_split.zip")
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


def _download_url(url: str, target: Path, *, dry_run: bool) -> None:
    if dry_run:
        print(f"would download {url} -> {target}")
        return
    if target.is_file() and target.stat().st_size > 0:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} …")
    urllib.request.urlretrieve(url, target)


def _download_coco_person(
    *,
    target: Path,
    manifest_root: Path,
    split: str,
    max_images: int,
    dry_run: bool,
) -> None:
    """Download a bounded COCO-2017 person-image subset and its box manifest.

    This is an appearance/augmentation corpus, not a drop-in geometric
    UniSHARP dataset: COCO does not supply paired target views or metric depth.
    The JSONL manifest retains the official person boxes for a later crop or
    monocular-augmentation stage.
    """
    archive = target / ".archives" / "annotations_trainval2017.zip"
    annotation = target / "annotations" / f"instances_{split}.json"
    if dry_run:
        count_text = "all person images" if int(max_images) == 0 else f"up to {int(max_images)} person images"
        print(f"would download COCO {split} annotations and {count_text} into {target}")
        return
    _download_url(COCO_ANNOTATIONS_URL, archive, dry_run=False)
    if not annotation.is_file():
        with zipfile.ZipFile(archive) as package:
            member = f"annotations/instances_{split}.json"
            if member not in package.namelist():
                raise RuntimeError(f"Official COCO annotation archive does not contain {member}")
            annotation.parent.mkdir(parents=True, exist_ok=True)
            with package.open(member) as source, annotation.open("wb") as destination:
                shutil.copyfileobj(source, destination)

    payload = json.loads(annotation.read_text(encoding="utf-8"))
    person_category = next((item["id"] for item in payload["categories"] if item["name"] == "person"), None)
    if person_category is None:
        raise RuntimeError("COCO annotations contain no 'person' category")
    boxes_by_image: dict[int, list[list[float]]] = {}
    for item in payload["annotations"]:
        if int(item["category_id"]) == int(person_category) and not bool(item.get("iscrowd", 0)):
            boxes_by_image.setdefault(int(item["image_id"]), []).append([float(value) for value in item["bbox"]])
    records = [item for item in payload["images"] if int(item["id"]) in boxes_by_image]
    records.sort(key=lambda item: int(item["id"]))
    if int(max_images) > 0:
        records = records[: int(max_images)]
    image_root = target / "images" / split
    manifest_root.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    rows: list[str] = []
    for index, item in enumerate(records, start=1):
        filename = str(item["file_name"])
        image_path = image_root / filename
        _download_url(COCO_IMAGE_URL_TEMPLATE.format(split=split, filename=filename), image_path, dry_run=False)
        paths.append(str(image_path.resolve()))
        rows.append(json.dumps({"image": str(image_path.resolve()), "person_boxes_xywh": boxes_by_image[int(item["id"])]}))
        if index % 100 == 0 or index == len(records):
            print(f"COCO person images: {index}/{len(records)}")
    (manifest_root / f"coco_person_{split}_images.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
    (manifest_root / f"coco_person_{split}_boxes.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"COCO person subset prepared: {target} ({len(records)} images)")


def _read_widerface_boxes(annotation: Path) -> list[tuple[str, list[list[float]]]]:
    """Parse the released WIDER FACE bounding-box text format."""
    lines = annotation.read_text(encoding="utf-8").splitlines()
    records: list[tuple[str, list[list[float]]]] = []
    index = 0
    while index < len(lines):
        relative = lines[index].strip()
        index += 1
        if not relative:
            continue
        if index >= len(lines):
            raise RuntimeError(f"Malformed WIDER FACE annotation near {relative}")
        count = int(lines[index].strip())
        index += 1
        boxes: list[list[float]] = []
        for _ in range(count):
            values = lines[index].split()
            index += 1
            if len(values) < 4:
                raise RuntimeError(f"Malformed WIDER FACE box for {relative}")
            boxes.append([float(value) for value in values[:4]])
        if boxes:
            records.append((relative, boxes))
    return records


def _download_widerface(
    *,
    token: str | None,
    target: Path,
    manifest_root: Path,
    max_images: int,
    include_val: bool,
    dry_run: bool,
) -> None:
    """Fetch WIDER FACE archives and create a bounded face-box manifest.

    WIDER FACE labels faces but provides neither metric depth nor target-view
    poses.  It is therefore intentionally exported as an appearance/crop
    augmentation corpus rather than a UniSHARP geometric training manifest.
    """
    selected_patterns = [WIDER_FACE_PATTERNS[0], WIDER_FACE_PATTERNS[2]]
    if include_val:
        selected_patterns.append(WIDER_FACE_PATTERNS[1])
    if dry_run:
        count_text = "all annotated images" if int(max_images) == 0 else f"up to {int(max_images)} annotated images"
        print(f"would download {WIDER_FACE_REPO} ({', '.join(selected_patterns)}) and prepare {count_text} in {target}")
        return
    _, snapshot_download = _require_huggingface_hub()
    source = target / ".hf_source"
    print(f"Downloading WIDER FACE archives from {WIDER_FACE_REPO} …")
    snapshot_download(
        repo_id=WIDER_FACE_REPO,
        repo_type="dataset",
        allow_patterns=selected_patterns,
        local_dir=str(source),
        token=token,
    )
    for name in ("WIDER_train.zip", "WIDER_val.zip", "wider_face_split.zip"):
        archive = source / "data" / name
        if archive.is_file():
            marker = target / ".extracted" / f"{name}.ok"
            if not marker.is_file():
                print(f"Extracting {name} …")
                with zipfile.ZipFile(archive) as package:
                    package.extractall(target)
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text("ok\n", encoding="utf-8")

    candidates: list[tuple[Path, Path]] = [
        (target / "WIDER_train" / "images", target / "wider_face_split" / "wider_face_train_bbx_gt.txt"),
    ]
    if include_val:
        candidates.append((target / "WIDER_val" / "images", target / "wider_face_split" / "wider_face_val_bbx_gt.txt"))
    records: list[tuple[Path, list[list[float]]]] = []
    for image_root, annotation in candidates:
        if not image_root.is_dir() or not annotation.is_file():
            raise RuntimeError(f"WIDER FACE extraction is incomplete: expected {image_root} and {annotation}")
        for relative, boxes in _read_widerface_boxes(annotation):
            image_path = image_root / relative
            if image_path.is_file():
                records.append((image_path.resolve(), boxes))
    records.sort(key=lambda item: str(item[0]))
    if int(max_images) > 0:
        records = records[: int(max_images)]
    manifest_root.mkdir(parents=True, exist_ok=True)
    image_manifest = manifest_root / "widerface_images.txt"
    box_manifest = manifest_root / "widerface_boxes.jsonl"
    image_manifest.write_text("\n".join(str(path) for path, _ in records) + "\n", encoding="utf-8")
    box_manifest.write_text(
        "\n".join(json.dumps({"image": str(path), "face_boxes_xywh": boxes}) for path, boxes in records) + "\n",
        encoding="utf-8",
    )
    print(f"WIDER FACE subset prepared: {target} ({len(records)} images)")


def _prepare_calibrated_human_dataset(
    *, name: str, slug: str, target: Path, manifest_root: Path, source_url: str, preparation: str, dry_run: bool
) -> None:
    """Prepare a local integration point for licensed calibrated human data.

    This intentionally never guesses archive URLs or bypasses a dataset's
    registration, research-use, or non-commercial terms.  It records the
    common JSONL contract used by the mixed UniSHARP loader instead.
    """
    if dry_run:
        print(f"would prepare {name} integration under {target}; get it from {source_url}")
        return
    target.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)
    template = manifest_root / f"{slug}_train.template.jsonl"
    if not template.exists():
        template.write_text(
            '{"scene":"subject/expression","frames":[{"image":"/absolute/path/frame0.jpg","intrinsics":[[1000,0,512],[0,1000,512],[0,0,1]],"w2c":[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]},{"image":"/absolute/path/frame1.jpg","intrinsics":[[1000,0,512],[0,1000,512],[0,0,1]],"w2c":[[1,0,0,0.1],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}]}\n',
            encoding="utf-8",
        )
    print(
        f"{name}: get the approved release from {source_url}, place it under {target}, then {preparation} "
        f"Convert the resulting RGB frames and OpenCV camera metadata to {template.name}; save the completed "
        f"manifest as {slug}_train.jsonl."
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assets",
        nargs="+",
        choices=("unik3d", "manifests", "unisharp-checkpoints", "omnirooms", "re10k", "wildrgbd", "dl3dv", "coco-person", "widerface", "humbi", "nersemble", "aistpp", "thuman"),
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
    parser.add_argument("--coco-person-root", type=Path, default=DATA_ROOT / "coco_person")
    parser.add_argument("--coco-person-split", choices=("train2017", "val2017"), default="train2017")
    parser.add_argument("--coco-person-max-images", type=int, default=5000, help="0 downloads every COCO image containing a non-crowd person instance.")
    parser.add_argument("--widerface-root", type=Path, default=DATA_ROOT / "widerface")
    parser.add_argument("--widerface-max-images", type=int, default=10000, help="0 keeps every WIDER FACE train image with a labeled face.")
    parser.add_argument("--widerface-include-val", action="store_true", help="Also download WIDER FACE validation images and labels.")
    parser.add_argument("--humbi-root", type=Path, default=DATA_ROOT / "humbi")
    parser.add_argument("--nersemble-root", type=Path, default=DATA_ROOT / "nersemble")
    parser.add_argument("--aistpp-root", type=Path, default=DATA_ROOT / "aistpp")
    parser.add_argument("--thuman-root", type=Path, default=DATA_ROOT / "thuman")
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
    if "coco-person" in assets:
        if int(args.coco_person_max_images) < 0:
            raise SystemExit("--coco-person-max-images must be >= 0.")
        _download_coco_person(
            target=Path(args.coco_person_root),
            manifest_root=Path(args.manifest_root),
            split=str(args.coco_person_split),
            max_images=int(args.coco_person_max_images),
            dry_run=bool(args.dry_run),
        )
    if "widerface" in assets:
        if int(args.widerface_max_images) < 0:
            raise SystemExit("--widerface-max-images must be >= 0.")
        _download_widerface(
            token=token,
            target=Path(args.widerface_root),
            manifest_root=Path(args.manifest_root),
            max_images=int(args.widerface_max_images),
            include_val=bool(args.widerface_include_val),
            dry_run=bool(args.dry_run),
        )
    if "humbi" in assets:
        _prepare_calibrated_human_dataset(
            name="HUMBI", slug="humbi", target=Path(args.humbi_root), manifest_root=Path(args.manifest_root),
            source_url="https://github.com/zhixuany/HUMBI",
            preparation="use its provided intrinsic/extrinsic calibration files for synchronized camera views.",
            dry_run=bool(args.dry_run),
        )
    if "nersemble" in assets:
        _prepare_calibrated_human_dataset(
            name="NeRSemble", slug="nersemble", target=Path(args.nersemble_root), manifest_root=Path(args.manifest_root),
            source_url="https://github.com/tobias-kirschstein/nersemble",
            preparation="request access and retain the release's calibrated synchronized head-camera views.",
            dry_run=bool(args.dry_run),
        )
    if "aistpp" in assets:
        _prepare_calibrated_human_dataset(
            name="AIST++", slug="aistpp", target=Path(args.aistpp_root), manifest_root=Path(args.manifest_root),
            source_url="https://google.github.io/aistplusplus_dataset/download.html",
            preparation="run the official downloader, then pair synchronized camera views using the published calibration.",
            dry_run=bool(args.dry_run),
        )
    if "thuman" in assets:
        _prepare_calibrated_human_dataset(
            name="THuman", slug="thuman", target=Path(args.thuman_root), manifest_root=Path(args.manifest_root),
            source_url="https://github.com/ytrock/THuman2.0-Dataset",
            preparation="render each approved scan into at least two views and write the renderer's OpenCV w2c matrices and pixel intrinsics.",
            dry_run=bool(args.dry_run),
        )


if __name__ == "__main__":
    main()

from __future__ import annotations

"""Create an appearance-training manifest from a local image directory."""

import argparse
from pathlib import Path


DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", type=Path, nargs="+", required=True,
        help="One or more local photo directories; every subdirectory is included.",
    )
    parser.add_argument("--manifest", type=Path, required=True, help="Output image-list manifest.")
    parser.add_argument("--max-images", type=int, default=0, help="0 keeps every supported image; otherwise keeps the first N sorted paths.")
    parser.add_argument("--extensions", nargs="+", default=DEFAULT_EXTENSIONS, help="Image file suffixes to include.")
    args = parser.parse_args()
    if int(args.max_images) < 0:
        raise SystemExit("--max-images must be >= 0")
    roots = [root.resolve() for root in args.source_root]
    missing = [root for root in roots if not root.is_dir()]
    if missing:
        raise SystemExit(f"Local image root does not exist: {missing[0]}")
    extensions = {str(value).lower() if str(value).startswith(".") else f".{str(value).lower()}" for value in args.extensions}
    images = sorted(
        path.resolve()
        for root in roots
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )
    if int(args.max_images) > 0:
        images = images[: int(args.max_images)]
    if not images:
        raise SystemExit("No supported images under the supplied local image roots.")
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text("\n".join(str(path) for path in images) + "\n", encoding="utf-8")
    print(f"Wrote {len(images)} local-image records: {args.manifest}")


if __name__ == "__main__":
    main()

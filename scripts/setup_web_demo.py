#!/usr/bin/env python3
"""Prepare and optionally start the local UniSHARP browser demo.

This helper deliberately downloads model weights from their public upstream
Hugging Face repositories instead of placing them in the Git repository.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse explicit setup and optional server-start settings."""
    parser = argparse.ArgumentParser(
        description="Install dependencies and download public weights for the UniSHARP web demo."
    )
    parser.add_argument(
        "--skip-install", action="store_true", help="Do not install Python packages from requirements.txt."
    )
    parser.add_argument(
        "--skip-weights", action="store_true", help="Do not download the public UniSHARP and UniK3D weights."
    )
    parser.add_argument("--start", action="store_true", help="Start the local web demo after setup completes.")
    parser.add_argument("--host", default="127.0.0.1", help="Host passed to the web demo when using --start.")
    parser.add_argument("--port", type=int, default=5000, help="Port passed to the web demo when using --start.")
    parser.add_argument(
        "--allow-network", action="store_true", help="Pass through the explicit network-binding opt-in when using --start."
    )
    return parser.parse_args(argv)


def run(command: list[str | Path]) -> None:
    """Run one setup command in the repository and stop on an error."""
    subprocess.run([str(value) for value in command], cwd=REPO_ROOT, check=True)


def main(argv: list[str] | None = None) -> None:
    """Install runtime dependencies, cache public model weights, and optionally serve."""
    args = parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535")

    if not args.skip_install:
        run([sys.executable, "-m", "pip", "install", "-r", REPO_ROOT / "requirements.txt"])
    if not args.skip_weights:
        run([
            sys.executable,
            REPO_ROOT / "scripts" / "download_npu_assets.py",
            "--assets",
            "unik3d",
            "unisharp-checkpoints",
            "--backbones",
            "vitl",
        ])

    if args.start:
        command: list[str | Path] = [
            sys.executable,
            REPO_ROOT / "scripts" / "run_web_demo.py",
            "--host",
            args.host,
            "--port",
            str(args.port),
        ]
        if args.allow_network:
            command.append("--allow-network")
        run(command)
        return

    print("Web demo prerequisites are ready. Start it with: python scripts/run_web_demo.py")


if __name__ == "__main__":
    main()

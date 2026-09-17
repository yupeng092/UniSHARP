#!/usr/bin/env python3
"""Start the localhost-only UniSHARP upload and multiview demo."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}
_CHECKPOINT_RELATIVE_PATH = Path("checkpoints") / "released" / "pretained_model.pt"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse local-safe server settings, requiring opt-in for network binding."""
    parser = argparse.ArgumentParser(description="Run the local UniSHARP browser demonstration.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host; defaults to local loopback only.")
    parser.add_argument("--port", type=int, default=5000, help="Bind port; defaults to 5000.")
    parser.add_argument("--allow-network", action="store_true", help="Required when binding to a non-loopback host.")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.host not in _LOOPBACK_HOSTS and not args.allow_network:
        parser.error("non-loopback --host requires --allow-network")
    return args


def missing_prerequisites(repo_root: Path = REPO_ROOT) -> list[str]:
    """Return actionable setup problems before the server accepts uploads."""
    problems: list[str] = []
    if not (repo_root / _CHECKPOINT_RELATIVE_PATH).is_file():
        problems.append(f"missing model checkpoint: {_CHECKPOINT_RELATIVE_PATH}")
    try:
        import flask  # noqa: F401
    except ModuleNotFoundError:
        problems.append("missing Python dependency: Flask")
    return problems


def main() -> None:
    """Create the app and start Flask without debugger or reloader processes."""
    args = parse_args()
    problems = missing_prerequisites()
    if problems:
        joined = "\n- ".join(problems)
        raise SystemExit(
            "Cannot start the UniSHARP web demo:\n- "
            f"{joined}\n\n"
            "Run this once from the repository root:\n"
            "python scripts/setup_web_demo.py --start"
        )
    from web_demo import create_app

    app = create_app({"BIND_HOST": args.host, "PORT": args.port})
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

"""Flash3D-compatible CPU multiview renderer for UniSHARP Gaussian exports.

This entry point forwards Flash3D's complete CPU-renderer interface rather
than maintaining a second, subtly different implementation. UniSHARP
``gaussians.pt`` files are supported by Flash3D's loader.
"""

import importlib.util
import os
import sys
from pathlib import Path


DEFAULT_FLASH3D_ROOT = Path(os.environ.get("FLASH3D_ROOT", r"D:\PythonFiles\flash3d-main"))


def _split_root_argument(argv: list[str]) -> tuple[Path, list[str]]:
    """Consume only this wrapper option; forward all Flash3D options unchanged."""
    root = DEFAULT_FLASH3D_ROOT
    forwarded: list[str] = []
    index = 0
    while index < len(argv):
        value = argv[index]
        if value == "--flash3d-root":
            if index + 1 >= len(argv):
                raise ValueError("--flash3d-root requires a directory path.")
            root = Path(argv[index + 1])
            index += 2
            continue
        forwarded.append(value)
        index += 1
    return root, forwarded


def main() -> None:
    flash3d_root, forwarded = _split_root_argument(sys.argv[1:])
    renderer_script = flash3d_root / "render_cpu_multiview.py"
    alpha_script = flash3d_root / "render_cpu_alpha.py"
    if not renderer_script.is_file() or not alpha_script.is_file():
        raise FileNotFoundError(
            "Flash3D renderer files not found: "
            f"{renderer_script} and {alpha_script}. Set --flash3d-root or FLASH3D_ROOT."
        )
    root_text = str(flash3d_root.resolve())
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    module_spec = importlib.util.spec_from_file_location("_unisharp_flash3d_renderer", renderer_script)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"Unable to load {renderer_script}")
    renderer_module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = renderer_module
    module_spec.loader.exec_module(renderer_module)
    old_argv = sys.argv
    try:
        sys.argv = [str(renderer_script), *forwarded]
        renderer_module.render(renderer_module.parse_args())
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()

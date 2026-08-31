#!/usr/bin/env python3
"""Fail fast before starting UniSHARP's Ascend training or rendering path."""

from __future__ import annotations

import json
import os
import argparse

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-meta-gauss-render",
        action="store_true",
        help="also require the CANN fused 3DGS custom-op wheel used by renderer-backend=ascend_fused",
    )
    args = parser.parse_args()
    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise SystemExit("torch_npu is unavailable; install the CANN-matched wheel first.") from exc
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise SystemExit("No usable Ascend NPU is visible. Check CANN and ASCEND_RT_VISIBLE_DEVICES.")
    fused_renderer = False
    if args.require_meta_gauss_render:
        try:
            import meta_gauss_render  # noqa: F401
            required = (
                "projection_three_dims_gaussian_fused",
                "flash_gaussian_build_mask",
                "gaussian_sort",
                "calc_render",
                "get_render_schedule",
            )
            missing = [name for name in required if not hasattr(meta_gauss_render, name)]
            if missing:
                raise ImportError(f"missing APIs: {', '.join(missing)}")
            fused_renderer = True
        except Exception as exc:
            raise SystemExit(
                "CANN fused renderer is unavailable. Build/install meta_gauss_render first with "
                "`bash scripts/install_meta_gauss_render.sh --source /path/to/"
                "cann-recipes-embodied-ai`. To use the slow reference backend instead, set "
                "NPU_RENDERER_BACKEND=portable."
            ) from exc
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    x = torch.randn(256, 256, device=device, dtype=torch.bfloat16, requires_grad=True)
    y = (x @ x).float().mean()
    y.backward()
    torch.npu.synchronize(device)
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "torch_npu": getattr(torch_npu, "__version__", "unknown"),
                "npu_count": torch.npu.device_count(),
                "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "all"),
                "meta_gauss_render": fused_renderer,
                "smoke_loss": float(y.detach().cpu()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fail fast before starting UniSHARP's portable Ascend training path."""

from __future__ import annotations

import json
import os

import torch


def main() -> None:
    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise SystemExit("torch_npu is unavailable; install the CANN-matched wheel first.") from exc
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise SystemExit("No usable Ascend NPU is visible. Check CANN and ASCEND_RT_VISIBLE_DEVICES.")
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
                "smoke_loss": float(y.detach().cpu()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

"""Numerically compare the portable classic renderer against CUDA gsplat.

Run this on a CUDA host after installing the same gsplat version used by the
native UniSHARP renderer. It deliberately executes both renderers with the
same float32 inputs and reports RGB, alpha and depth errors. It is a semantic
parity test for the NPU reference path; a cross-device bitwise test requires
an Ascend custom rasterization operator.
"""

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from unisharp.utils.gaussians import Gaussians3D


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--num-gaussians", type=int, default=256)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--eps2d", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=3e-4)
    parser.add_argument("--rtol", type=float, default=2e-3)
    return parser


def _max_error(lhs: torch.Tensor, rhs: torch.Tensor) -> tuple[float, float]:
    delta = (lhs - rhs).abs()
    rel = delta / rhs.abs().clamp_min(1e-6)
    return float(delta.max().item()), float(rel.max().item())


def main() -> None:
    args = _parser().parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("This parity validator requires CUDA and the native gsplat package.")
    if int(args.width) < 1 or int(args.height) < 1 or int(args.num_gaussians) < 1:
        raise SystemExit("--width, --height and --num-gaussians must be positive.")
    from unisharp.utils.gsplat import GSplatRenderer
    from unisharp.utils.portable_renderer import PortableGaussianRenderer

    torch.manual_seed(int(args.seed))
    device = torch.device("cuda")
    n = int(args.num_gaussians)
    means = torch.randn((1, n, 3), device=device, dtype=torch.float32)
    means[..., 0:2] *= 0.7
    means[..., 2] = torch.rand((1, n), device=device) * 4.0 + 2.0
    gaussians = Gaussians3D(
        mean_vectors=means,
        singular_values=torch.rand((1, n, 3), device=device) * 0.08 + 0.01,
        quaternions=torch.randn((1, n, 4), device=device),
        colors=torch.rand((1, n, 3), device=device),
        opacities=torch.rand((1, n, 1), device=device) * 0.9 + 0.05,
    )
    extrinsics = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0)
    intrinsics = torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0)
    intrinsics[:, 0, 0] = float(args.width) * 0.9
    intrinsics[:, 1, 1] = float(args.height) * 0.9
    intrinsics[:, 0, 2] = (float(args.width) - 1.0) * 0.5
    intrinsics[:, 1, 2] = (float(args.height) - 1.0) * 0.5
    native = GSplatRenderer(low_pass_filter_eps=float(args.eps2d)).to(device)
    portable = PortableGaussianRenderer(low_pass_filter_eps=float(args.eps2d)).to(device)
    with torch.no_grad():
        expected = native(gaussians, extrinsics, intrinsics, int(args.width), int(args.height))
        actual = portable(gaussians, extrinsics, intrinsics, int(args.width), int(args.height))
    reports = {
        "color": _max_error(actual.color, expected.color),
        "alpha": _max_error(actual.alpha, expected.alpha),
        "depth": _max_error(actual.depth, expected.depth),
    }
    for name, (absolute, relative) in reports.items():
        print(f"{name}: max_abs={absolute:.8g} max_rel={relative:.8g}")
    checks = [
        torch.allclose(actual.color, expected.color, atol=float(args.atol), rtol=float(args.rtol)),
        torch.allclose(actual.alpha, expected.alpha, atol=float(args.atol), rtol=float(args.rtol)),
        torch.allclose(actual.depth, expected.depth, atol=float(args.atol), rtol=float(args.rtol)),
    ]
    if not all(checks):
        raise SystemExit("Parity tolerance exceeded; inspect the reported component and gsplat version.")
    print("PASS: portable reference is within the requested CUDA gsplat tolerance.")


if __name__ == "__main__":
    main()

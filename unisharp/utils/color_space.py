
from __future__ import annotations

import logging
from typing import Callable
from typing import Literal

import torch

LOGGER = logging.getLogger(__name__)

ColorSpace = Literal["sRGB", "linearRGB"]


def robust_where(
    condition: torch.Tensor,
    input: torch.Tensor,
    branch_true_func: Callable[[torch.Tensor], torch.Tensor],
    branch_false_func: Callable[[torch.Tensor], torch.Tensor],
    branch_true_safe_value: float | None = None,
    branch_false_safe_value: float | None = None,
) -> torch.Tensor:
    input_1 = input
    input_2 = input
    if branch_true_safe_value is not None:
        input_1 = torch.where(condition, input_1, branch_true_safe_value)
    if branch_false_safe_value is not None:
        input_2 = torch.where(~condition, input_2, branch_false_safe_value)
    return torch.where(
        condition,
        branch_true_func(input_1),
        branch_false_func(input_2),
    )


def encode_color_space(color_space: ColorSpace) -> int:
    return 0 if color_space == "sRGB" else 1


def decode_color_space(color_space_index: int) -> ColorSpace:
    return "sRGB" if color_space_index == 0 else "linearRGB"


def sRGB2linearRGB(sRGB: torch.Tensor) -> torch.Tensor:
    THRESHOLD = 0.04045

    def branch_true_func(x):
        return x / 12.92

    def branch_false_func(x):
        return ((x + 0.055) / 1.055) ** 2.4

    return robust_where(
        sRGB <= THRESHOLD,
        sRGB,
        branch_true_func,
        branch_false_func,
        branch_false_safe_value=THRESHOLD,
    )


def linearRGB2sRGB(linearRGB: torch.Tensor) -> torch.Tensor:
    THRESHOLD = 0.0031308

    def branch_true_func(x):
        return x * 12.92

    def branch_false_func(x):
        return 1.055 * (x ** (1 / 2.4)) - 0.055

    return robust_where(
        linearRGB <= THRESHOLD,
        linearRGB,
        branch_true_func,
        branch_false_func,
        branch_false_safe_value=THRESHOLD,
    )

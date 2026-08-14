
from __future__ import annotations

from typing import Any, Callable, Literal, NamedTuple, Union

import torch
from torch import autograd

ActivationType = Literal[
    "linear",
    "exp",
    "sigmoid",
    "softplus",
    "relu_with_pushback",
    "hard_sigmoid_with_pushback",
]
ActivationFunction = Callable[[torch.Tensor], torch.Tensor]


class ActivationPair(NamedTuple):

    forward: ActivationFunction
    inverse: ActivationFunction


def create_activation_pair(activation_type: ActivationType) -> ActivationPair:
    if activation_type == "linear":
        return ActivationPair(lambda x: x, lambda x: x)
    elif activation_type == "exp":
        return ActivationPair(torch.exp, torch.log)
    elif activation_type == "sigmoid":
        return ActivationPair(torch.sigmoid, inverse_sigmoid)
    elif activation_type == "softplus":
        return ActivationPair(torch.nn.functional.softplus, inverse_softplus)
    elif activation_type == "relu_with_pushback":
        return ActivationPair(relu_with_pushback, lambda x: x)
    elif activation_type == "hard_sigmoid_with_pushback":
        return ActivationPair(hard_sigmoid_with_pushback, lambda x: 6.0 * x - 3.0)
    else:
        raise ValueError(f"Unsupported activation function: {activation_type}.")


def inverse_sigmoid(tensor: torch.Tensor) -> torch.Tensor:
    return torch.log(tensor / (1.0 - tensor))


def inverse_softplus(tensor: torch.Tensor, eps: float = 1e-06) -> torch.Tensor:
    tensor = tensor.clamp_min(eps)
    sigmoid = torch.sigmoid(-tensor)
    exp = sigmoid / (1.0 - sigmoid)
    return tensor + torch.log(-exp + 1.0)


class ClampWithPushback(autograd.Function):

    @staticmethod
    def forward(
        ctx: Any,
        tensor: torch.Tensor,
        min: float | None,
        max: float | None,
        pushback: float,
    ) -> torch.Tensor:
        if min is not None and max is not None and min >= max:
            raise ValueError("Only min < max is supported.")

        ctx.save_for_backward(tensor)
        ctx.min = min
        ctx.max = max
        ctx.pushback = pushback
        return torch.clamp(tensor, min=min, max=max)

    @staticmethod
    def backward(  # type: ignore[override] # Deal with buggy torch annotations.
        ctx: Any, grad_in: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None]:
        grad_out = grad_in.clone()
        (tensor,) = ctx.saved_tensors

        if ctx.min is not None:
            mask_min = tensor < ctx.min
            grad_out[mask_min] = -ctx.pushback

        if ctx.max is not None:
            mask_max = tensor > ctx.max
            grad_out[mask_max] = ctx.pushback

        return grad_out, None, None, None


def clamp_with_pushback(
    tensor: torch.Tensor,
    min: float | None = None,
    max: float | None = None,
    pushback: float = 1e-2,
) -> torch.Tensor:
    output = ClampWithPushback.apply(tensor, min, max, pushback)
    assert isinstance(output, torch.Tensor)
    return output


def hard_sigmoid_with_pushback(x: torch.Tensor, slope: float = 1.0 / 6.0) -> torch.Tensor:
    return clamp_with_pushback(slope * x + 0.5, min=0.0, max=1.0)


def relu_with_pushback(x: torch.Tensor) -> torch.Tensor:
    return clamp_with_pushback(x, min=0.0)

"""Testing utilities for MiniTorch tensors and scalars."""

from __future__ import annotations

from typing import Callable, Iterable, List, Tuple

from . import operators
from .scalar import Scalar
from .tensor import Tensor
from .tensor_functions import grad_check


class MathTestVariable:
    """Collection of mathematical functions used by tensor tests."""

    @classmethod
    def _comp_testing(
        cls,
    ) -> Tuple[
        List[Tuple[str, Callable[[float], float], Callable[[Tensor], Tensor]]],
        List[
            Tuple[
                str,
                Callable[[float, float], float],
                Callable[[Tensor, Tensor], Tensor],
            ]
        ],
        List[
            Tuple[
                str,
                Callable[[Iterable[float]], float],
                Callable[[Tensor], Tensor],
            ]
        ],
    ]:
        """Return scalar/tensor function pairs used by the tensor tests."""

        one_arg = [
            (
                "neg",
                operators.neg,
                lambda x: -x,
            ),
            (
                "sigmoid",
                operators.sigmoid,
                lambda x: x.sigmoid(),
            ),
            (
                "relu",
                operators.relu,
                lambda x: x.relu(),
            ),
            (
                "log",
                operators.log,
                lambda x: x.log(),
            ),
            (
                "exp",
                operators.exp,
                lambda x: x.exp(),
            ),
            (
                "inv",
                operators.inv,
                lambda x: x.inv(),
            ),
        ]

        two_arg = [
            (
                "add",
                operators.add,
                lambda x, y: x + y,
            ),
            (
                "mul",
                operators.mul,
                lambda x, y: x * y,
            ),
            (
                "lt",
                operators.lt,
                lambda x, y: x < y,
            ),
            (
                "eq",
                operators.eq,
                lambda x, y: x == y,
            ),
            (
                "is_close",
                operators.is_close,
                lambda x, y: x.is_close(y),
            ),
        ]

        def reduce_sum(values: Iterable[float]) -> float:
            result = 0.0
            for value in values:
                result = operators.add(result, value)
            return result

        red_arg = [
            (
                "sum",
                reduce_sum,
                lambda x: x.sum(),
            ),
        ]

        return one_arg, two_arg, red_arg


def assert_close(a: float, b: float) -> None:
    """Check that two scalar values are numerically close."""
    assert abs(a - b) < 1e-6


def test_function(x_val: float, y_val: float) -> None:
    """Small scalar autodiff check used for manual testing."""

    import math
    from .autodiff import central_difference

    x = Scalar(x_val)
    x.requires_grad_(True)

    y = Scalar(y_val)
    y.requires_grad_(True)

    z = x * y + x.log()
    z.backward()

    def f_for_x(value: float) -> float:
        return value * y_val + math.log(value)

    def f_for_y(value: float) -> float:
        return x_val * value + math.log(x_val)

    numerical_dx = central_difference(f_for_x, x_val)
    numerical_dy = central_difference(f_for_y, y_val)

    print(
        f"Autodiff: dx={x.derivative:.6f}, "
        f"dy={y.derivative:.6f}"
    )
    print(
        f"Numerical: dx={numerical_dx:.6f}, "
        f"dy={numerical_dy:.6f}"
    )


if __name__ == "__main__":
    test_function(3.0, 4.0)
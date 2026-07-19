from .scalar import Scalar
from .autodiff import central_difference
import math

def test_function(x_val, y_val):
    x = Scalar(x_val)
    x.requires_grad_(True)
    y = Scalar(y_val)
    y.requires_grad_(True)

    # Compute : z = (x * y) + x.log()
    z = x * y + x.log()
    z.backward()

    #compare with numerical derivatives
    def f_for_x(x_val):
        return x_val * y_val + math.log(x_val)
    def f_for_y(y_val):
        return x_val * y_val + math.log(x_val)
    
    numerical_dx = central_difference(f_for_x, x_val)
    numerical_dy = central_difference(f_for_y, y_val)

    print(f"Autodiff: dx={x.derivative:.6f}, dy={y.derivative:.6f}")
    print(f"Numerical: dx={numerical_dx:.6f}, dy={numerical_dy:.6f}")





test_function(3.0, 4.0)

def assert_close(a: float, b: float) -> None:
    assert abs(a - b) < 1e-6
"""
Mathemtical operators for Minitorch. these from the foundation of all neural network operations.
"""

import math
from typing import Callable , Iterable

#Basic Arithmetic Operators

def add(x:float, y:float) -> float:
    """Adds two numbers together"""
    return x + y

def mul(x:float, y:float) -> float:
    """Multiplies two numbers together"""
    return x * y

def id(x:float) -> float:
    """Identity function - return the input unchaged."""
    return x

def neg(x:float) -> float:
    """Negates a number"""
    return -x

# Comparison Operations (return float for differentiability)

def lt(x:float, y:float) -> float:
    """Returns 1.0 if x < y else 0.0"""
    return 1.0 if x < y else 0.0

def eq(x:float, y:float) -> float:
    """Returns 1.0 if x == y else 0.0"""
    return 1.0 if x == y else 0.0

def max(x:float, y:float) -> float:
    """Returns the maximum of x and y"""
    return x if x > y else y

def is_close(x:float, y:float, tol:float=1e-9) -> float:
    """Returns 1.0 if x and y are within tol of each other else 0.0"""
    return 1.0 if abs(x - y) < tol else 0.0

# Activation Functions

def sigmoid(x: float) -> float:
    if x >= 0:
        result = 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        result = exp_x / (1.0 + exp_x)
    
    eps = 1e-12
    if result == 1.0:
        return 1.0 - eps
    if result == 0.0:
        return eps
    return result


def relu(x:float) -> float:
    """ReLU activation function"""
    return x if x > 0 else 0.0

def log(x:float) -> float:
    """Natural logarithm function"""
    return math.log(x)

def exp(x: float) -> float:
    return math.exp(x)

def inv(x: float) -> float:
    return 1.0 / x

# Gradient Helper Functions 

def log_back(x:float, grad:float) ->float:
    """
    Gradient of log(x) times incoming gradient.
    Derivative of log(x) is 1/x.
    """    
    return grad / x

def inv_back(x:float, grad:float) -> float:
    """
    Gradient of 1/x times incoming gradient .
    Derivative of 1/x is -1/x^2.
    """
    return -grad / (x * x)

def relu_back(x:float, grad:float) -> float:
    """
    Gradient of ReLU times incoming gradient.
    Derivative of ReLU is 1 if x > 0 else 0.
    """
    return grad if x > 0 else 0.0





#Higher-Order Functions

def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]],Iterable[float]]:
    
    """
    Higher-order map function . Returns a function that applies fn to each element .
    """
    def mapped(ls:Iterable[float]) -> Iterable[float]:
        return [fn(x) for x in ls]
    

    return mapped


def zipWith(fn: Callable[[float, float], float]) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    def zipped(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        return [fn(x, y) for x, y in zip(ls1, ls2)]
    return zipped


def reduce(fn: Callable[[float, float], float], start: float) -> Callable[[Iterable[float]], float]:
    def reduced(ls: Iterable[float]) -> float:
        result = start
        for x in ls:
            result = fn(result, x)
        return result
    return reduced


# Composed Functions Using Higher-Order Functions

def negList(ls: Iterable[float]) -> Iterable[float]:
    """Negate each element in the list."""
    return map(neg)(ls) 

def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Element-wise addition of two lists."""
    return zipWith(add)(ls1, ls2)

def sum(ls: Iterable[float]) -> float:
    """Sum all elements in the list."""
    return reduce(add, 0.0)(ls)

def prod(ls: Iterable[float]) -> float:
    """Product of all elements in the list."""
    return reduce(mul, 1.0)(ls) 
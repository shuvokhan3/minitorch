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

def sigmoid(x:float) -> float:
    """Sigmoid activation function
    uses numerical stable implementation.
    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)

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
    Dericative of log(x) is 1/x.
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






# def naive_sigmoid(x:float) -> float:
#     return 1.0 / (1.0 + math.exp(-x))

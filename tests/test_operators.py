import math

from minitorch.operators import(add, mul, id, neg, lt, eq, max, is_close, sigmoid, relu, log,exp, inv, inv_back, naive_sigmoid)   

def test_add():
    assert add(3, 3) == 6
    assert add(-1, 1) == 0


def test_mul():
    assert mul(3, 3) == 9
    assert mul(-1, 1) == -1

def test_id():
    assert id(3) == 3
    assert id(-1) == -1

def test_neg():
    assert neg(3) == -3
    assert neg(-1) == 1

def test_lt():
    assert lt(3, 4) == 1.0
    assert lt(4, 3) == 0.0

def test_eq():
    assert eq(3, 3) == 1.0
    assert eq(3, 4) == 0.0

def test_max():
    assert max(3, 4) == 4
    assert max(4, 3) == 4

def test_is_close():
    assert is_close(1.000000001, 1.000000002) == 1.0
    assert is_close(1.0, 2.0) == 0.0

def test_sigmoid():
    result = sigmoid(0)
    assert abs(result - 0.5) < 1e-6

def test_relu():
    assert relu(3) == 3
    assert relu(-1) == 0

def test_log():
    assert log(1) == 0
    assert log(math.e) == 1

def test_exp():
    assert abs(exp(1) - 2.718281828) < 1e-6


def test_inv():
    assert inv(2) == 0.5

def test_log_back():
    assert log(1) == 0
    assert log(math.e) == 1


def test_inv_back():
    assert inv_back(2, 4) == -1.0

def test_relu_back():
    assert relu(3) == 3
    assert relu(-1) == 0   




# def test_naive_sigmoid():
#     result = naive_sigmoid(-1000)
#     assert abs(result - 0.5) < 1e-6 

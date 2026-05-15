import pytest
import math
from hypothesis import given
from hypothesis.strategies import floats
from hypothesis.strategies import lists


from minitorch.operators import(add, mul, id, neg, lt, eq, max, is_close, sigmoid, relu, log,exp, inv, inv_back , map)   
from minitorch.testing import assert_close

small_floats = floats(min_value=-100, max_value=100) 

small_float_lists = lists(
    floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
    max_size=20,
)

@pytest.mark.task0_2
@given(small_floats, small_floats,small_floats)
def test_distributive(x, y, z):
    """Test distributive property: z * (x + y) = z*x + z*y"""
    if all(math.isfinite(v) for v in [x, y, z]):
        #Left side: z * (x + y)
        left_side = mul(z, add(x, y))
        #right side: z*x + z*y
        right_side = add(mul(z, x), mul(z, y))
        assert_close(left_side, right_side)


@pytest.mark.task0_2
@given(small_floats, small_floats)
def test_symmetry(x , y):
    """Test that multipilication is commutative."""
    if math.isfinite(x) and math.isfinite(y):
        assert_close(mul(x, y), mul(y, x))



def test_id():
    assert id(3) == 3
    assert id(-1) == -1


@pytest.mark.task0_2
@given(small_floats)
def test_other(a):
    """Test that negation is the inverse of itself: neg(neg(a)) = a."""
    if math.isfinite(a):
        assert_close(neg(neg(a)), a)




@pytest.mark.task0_2
@given(small_floats, small_floats, small_floats)
def test_transitive(a, b, c):
    """Test transitive property of less-than operator: if a < b and b < c, than a < c."""
    if all(math.isfinite(v) for v in [a, b, c]):
        #check if a < b and b < c
        if(lt(a, b) == 1.0 and lt(b, c) ==1.0):
            assert lt(a, c) == 1.0


def test_eq():
    assert eq(3, 3) == 1.0
    assert eq(3, 4) == 0.0

def test_max():
    assert max(3, 4) == 4
    assert max(4, 3) == 4

def test_is_close():
    assert is_close(1.000000001, 1.000000002) == 1.0
    assert is_close(1.0, 2.0) == 0.0

#-------------------
## Property-based testing with Hypothesis

@pytest.mark.task0_2  # Labels this test under "task0_2" for grouping/filtering
@given(small_floats) # Hypothesis: auto-generates many random float inputs
def test_sigmoid_properties(a):
    "Test mathematical properties of sigmoid function"

    if(math.isfinite(a)):
        sig_a = sigmoid(a)

        #Property 1 : Output bounded between 0 and 1
        assert 0.0 <sig_a< 1.0

        #Property 2 : Sigmoid(0) = 0.5
        if is_close(a, 0.0) == 1.0:
            assert is_close(sig_a, 0.5) == 1.0

        #Property 3 :sigmoid(-x) = 1 - sigmoid(x)
        sig_neg_a = sigmoid(-a)
        expected = 1.0 - sig_a
        assert is_close(sig_neg_a, expected) == 1.0


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



#Higher order functions testing using property-based testing


@pytest.mark.task0_3
@given(small_float_lists)
def test_map_length_preserved(xs):
    #Property 1 : Length preserved 
    """Test that map preserves the length of the input list."""
    double = lambda x: x * 2
    result = map(double)(xs)
    assert len(result) == len(xs)

    #property 2 : Identity map returns the same list
    identity_result = map(id)(xs)
    assert identity_result == xs

    #Property 3 : Composite property
    increment = lambda x: x + 1
    composite_result = map(increment)(map(double)(xs))
    expected = [increment(double(x)) for x in xs]
    assert composite_result == expected

    






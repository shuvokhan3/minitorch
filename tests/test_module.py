import pytest
from minitorch.module import Module, Parameter
from hypothesis import given
from hypothesis.strategies import integers


#Parameter 


@pytest.mark.task0_4
class TestParameterInit:

    @given(integers())
    def test_stores_scalar(self, x):
        assert Parameter(x).value == x

    def test_stores_list(self):
        assert Parameter([1, 2, 3]).value == [1, 2, 3]
    
    def test_stores_none(self):
        assert Parameter(None).value is None
    
    def test_stores_string(self):
        assert Parameter("hello").value == "hello"

@pytest.mark.task0_4
class TestParameterShape:

    @given(integers())
    def test_scalar_has_empty_shape(self, x):
        assert Parameter(x).shape == ()
    
    def test_delegates_to_value_shape(self):
        class FakeArray:
            shape = (3, 4)
        assert Parameter(FakeArray()).shape == (3, 4)

    def test_none_value_has_empty_shape(self):
        assert Parameter(None).shape == ()
    
    

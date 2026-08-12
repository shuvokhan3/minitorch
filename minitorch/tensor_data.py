from __future__ import annotations
import random
from typing import Iterable, Optional, Sequence, Tuple, Union
import numpy as np
import numpy.typing as npt
from numpy import array, float64
from typing_extensions import TypeAlias
from .operators import prod
import numba


MAX_DIMS = 32

class IndexingError(RuntimeError):
    "Exception raised for indexing errors"
    pass

Storage: TypeAlias = npt.NDArray[np.float64]
OutIndex: TypeAlias = npt.NDArray[np.int32]
Index: TypeAlias = npt.NDArray[np.int32]
Shape: TypeAlias = npt.NDArray[np.int32]
Strides: TypeAlias = npt.NDArray[np.int32]

UserIndex: TypeAlias = Sequence[int]
UserShape: TypeAlias = Sequence[int]
UserStrides: TypeAlias = Sequence[int]

def index_to_position(index: Index, strides: Strides) -> int:
    """
    Convert a multidimensional tensor index into a single-dimensional position in storage based on strides

    Args: 
        index: index tuple of ints (as numpy array)
        strides: tensor strides (as numpy array)
    
    returns:
        postion in storage

    """
    position = 0
    for i, s in zip(index, strides):
        position += i * s
    return int(position)

def to_index(ordinal: int, shape:Shape, out_index: OutIndex) -> None:
    """
    Convert an ordinal (flat postion o..size-1) to a multi-dimensional index in the given shape

    This is the inverse mapping : given postion in enumeration order, produce the corresposding index.

    Args: 
        ordinal : ordinal postion to concert
        shape: tensor shape
        out_index: output array to fill with index values

    returns :
        None (Modifies out_index in place)

    """

    cur_ord = ordinal
    for i in range(len(shape) - 1, -1 , -1):
        out_index[i] = cur_ord % shape[i]
        cur_ord = cur_ord // shape[i]


def shape_broadcast(shape1: UserShape, shape2: UserShape) -> UserShape:
    """
    Broadcast two shapes to crete a new union shape.

    Args : 
    Shape 1 : first shape
    shape2  : second shape

    returns: broadcasted shape

    raises:
    indexingError: if shapes cannot broadcast

    """
    # Hint: Work from the right side of both shapes
    # At each position, take the max of the two dimensions
    # But raise an error if neither is 1 and they differ

    result = []
    len1, len2 = len(shape1), len(shape2)
    max_len = max(len1, len2)

    for i in range(max_len):
        d1 = shape1[len1 - 1 - i] if i < len1 else 1
        d2 = shape2[len2 - 1 - i] if i < len2 else 1

        if d1 == d2:
            result.append(d1)
        elif d1 == 1:
            result.append(d2)
        elif d2 == 1:
            result.append(d1)
        else:
            raise IndexingError(f"Cannot broadcast shapes {shape1} and {shape2}")

    return tuple(reversed(result))






#Add a temporary broadcast_index
def broadcast_index(
    big_index: Index,
    big_shape: Shape,
    shape: Shape,
    out_index: OutIndex,
) -> None:
    """
    Convert a big_index into big_shape to a smaller out_index into shape
    following broadcasting rules.

    If the shape dimension is 1, the index for that dimension should be 0.
    If the shape has fewer dimensions, ignore leading dimensions of big_index.

    Args:
        big_index: multidimensional index of bigger tensor
        big_shape: shape of bigger tensor
        shape: shape of smaller tensor
        out_index: output array to fill

    Returns:
        None (modifies out_index in place)
    """

    offset = len(big_shape) - len(shape)

    for i in range(len(shape)):
        if shape[i] == 1:
            out_index[i] = 0
        else:
            out_index[i] = big_index[i + offset]

def strides_from_shape(shape:UserShape) -> UserStrides:
    layout = [1]
    offset = 1

    for s in reversed(shape):
        layout.append(s * offset)
        offset = s * offset

    return tuple(reversed(layout[:-1]))





class TensorData:
    _storage: Storage
    _strides: Strides
    _shape: Shape
    strides: UserStrides
    shape: UserShape
    dims : int

    def __init__(
            self,
            storage: Union[Sequence[float], Storage],
            shape: UserShape,
            strides: Optional[UserStrides] = None,
    ):
        if isinstance(storage, np.ndarray):
            self._storage = storage
        else:
            self._storage = array(storage, dtype=float64)

        if strides is None:
            strides = strides_from_shape(shape)

        assert isinstance(strides,tuple), "Strides must be tuple"
        assert isinstance(shape, tuple), "Shape must be tuple"
        if len(strides) != len(shape):
            raise IndexingError(f"Len of strides {strides} must match {shape} .")
        self._strides = array(strides)
        self._shape = array(shape)
        self.strides = strides
        self.dims = len(strides)
        self.size = int(prod(shape))
        self.shape = shape
        assert len(self._storage) == self.size

    def to_cuda_(self) -> None:
        if not numba.cuda.is_cuda_array(self._storage):
            self._storage = numba.cuda.to_device(self._storage)

    def is_contiguous(self) -> bool:
        """
        Check that the layout is contiguous, i.e. outer dimensions have bigger strides than inner dimensions.

        Returns:
            bool : True if contiguous
        """
        last = 1e9
        for stride in self._strides:
            if stride > last:
                return False
            last = stride
        return True
    
    @staticmethod
    def shape_broadcast(shape_a: UserShape, shape_b: UserShape) -> UserShape:
        return shape_broadcast(shape_a, shape_b)

    def index(self, index: Union[int, UserIndex]) -> int:
        if isinstance(index, int):
            aindex: Index = array([index])
        if isinstance(index, tuple):
            aindex = array(index)

        #pretend 0-dim shape is 1-dim shape of singleton
        shape = self.shape
        if len(shape) == 0 and len(aindex) != 0:
            shape = (1,)
        #Check for errors
        if aindex.shape[0] != len(self.shape):
            raise IndexingError(f"Index {aindex} must be size of {self.shape}.")
        for i, ind in enumerate(aindex):
            if ind >= self.shape[i]:
                raise IndexingError(f"Index {aindex} out of range {self.shape}.")
            if ind < 0:
                raise IndexingError(f"Negative indexing for {aindex} not supported.")

        # Call fast indexing.
        return index_to_position(aindex, self._strides)

    def indices(self) -> Iterable[UserIndex]:
        lshape: Shape = array(self.shape)
        out_index: Index = np.zeros(len(self.shape), dtype=np.int32)
        for i in range(self.size):
            to_index(i, lshape, out_index)
            yield tuple(out_index)

    def sample(self) -> UserIndex:
        return tuple((random.randint(0, s - 1) for s in self.shape))

    def get(self, key: UserIndex) -> float:
        x: float = self._storage[self.index(key)]
        return x

    def set(self, key: UserIndex, val: float) -> None:
        self._storage[self.index(key)] = val

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        return (self._storage, self._shape, self._strides)


    def permute(self, *order: int) -> TensorData:
        """
        Permute the dimensions of the tensor.

        Args:
            *order: a permutation of the dimensions

        Returns:
            New TensorData with the same storage and a new dimension order
        """
        assert list(sorted(order)) == list(range(len(self.shape))), (
            f"Must give a position to each dimension . shape: {self.shape} order :{order}"
        )

        new_shape = tuple(self.shape[i] for i in order)
        new_strides = tuple(self.strides[i] for i in order)

        return TensorData(self._storage, new_shape, new_strides)
    

    
    def to_string(self) -> str:
        s = ""
        for index in self.indices():
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == 0:
                    l = "\n%s[" % ("\t" * i) + l
                else:
                    break
            s += l
            v = self.get(index)
            s += f"{v:3.2f}"
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == self.shape[i] - 1:
                    l += "]"
                else:
                    break
            if l:
                s += l
            else:
                s += " "
        return s





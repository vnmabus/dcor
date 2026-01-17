"""Utility functions."""

from __future__ import annotations

import enum
import warnings
from typing import TYPE_CHECKING, Any, Iterable, TypeVar, Union

import numpy as np
from array_api_compat import (
    array_namespace as array_namespace_compat,
    numpy as numpy_namespace,
)

# TODO: Change in the future
if TYPE_CHECKING:
    ArrayType = np.typing.NDArray[np.number[Any]]
else:
    ArrayType = np.ndarray

T = TypeVar("T", bound=ArrayType)
RandomLike = Union[
    np.random.RandomState,
    np.random.Generator,
    int,
    None,
]


class CompileMode(enum.Enum):
    """Compilation mode of the algorithm."""

    AUTO = enum.auto()
    """
    Try to use the fastest available method.
    """

    NO_COMPILE = enum.auto()
    """
    Use a pure Python implementation.
    """

    COMPILE_CPU = enum.auto()
    """
    Compile for execution in one CPU.
    """

    COMPILE_PARALLEL = enum.auto()
    """
    Compile for execution in multicore CPUs.
    """


class RowwiseMode(enum.Enum):
    """Rowwise mode of the algorithm."""

    AUTO = enum.auto()
    """
    Try to use the fastest available method.
    """

    NAIVE = enum.auto()
    """
    Use naive (list comprehension/map) computation.
    """

    OPTIMIZED = enum.auto()
    """
    Use optimized version, or fail if there is none.
    """


# TODO: Change the return type in the future
def array_namespace(*xs: Any) -> Any:
    # `xs` contains one or more arrays, or possibly Python scalars (accepting
    # those is a matter of taste, but doesn't seem unreasonable).
    try:
        return array_namespace_compat(*xs)
    except TypeError:
        warnings.warn(
            "Passing non-array objects to functions in the 'dcor' "
            "package is deprecated and will be removed in a future version",
            DeprecationWarning,
            stacklevel=3,  # TODO: Use skip_file_prefixes in Python 3.12?
        )
        return numpy_namespace


def _sqrt(x: T) -> T:
    """
    Return square root of an array.

    This sqrt function for ndarrays tries to use the exponentiation operator
    if the objects stored do not supply a sqrt method.

    Args:
        x: Input array.

    Returns:
        Square root of the input array.

    """
    # Replace negative values with 0
    xp = array_namespace(x)
    x_copy = xp.asarray(x + 0)
    x_copy[x_copy < 0] = 0

    try:
        return xp.sqrt(x_copy)
    except (AttributeError, TypeError):
        return x_copy**0.5


def _transform_to_1d(*args: T) -> Iterable[T]:
    """Convert column matrices to vectors, to always have a 1d shape."""
    xp = array_namespace(*args)

    for array in args:
        array = xp.asarray(array)

        dim = len(array.shape)
        assert dim <= 2

        if dim == 2:
            assert array.shape[1] == 1
            array = xp.reshape(array, -1)

        yield array


def _transform_to_2d(*args: T) -> Iterable[T]:
    """Convert vectors to column matrices, to always have a 2d shape."""
    xp = array_namespace(*args)

    for array in args:
        array = xp.asarray(array)

        dim = len(array.shape)
        assert dim <= 2

        if dim < 2:
            array = xp.expand_dims(array, axis=1)

        yield array


def _can_be_numpy_double(x: ArrayType) -> bool:
    """
    Return if the array can be safely converted to double.

    That happens when the dtype is a float with the same size of
    a double or narrower, or when is an integer that can be safely
    converted to double (if the roundtrip conversion works).

    """
    if array_namespace(x) != np:
        return False

    return (
        (
            np.issubdtype(x.dtype, np.floating)
            and x.dtype.itemsize <= np.dtype(float).itemsize
        ) or (
            np.issubdtype(x.dtype, np.signedinteger)
            and np.can_cast(x, float)
        )
    )


def _random_state_init(
    random_state: RandomLike,
) -> np.random.RandomState | np.random.Generator:
    """
    Initialize a RandomState object.

    If the object is a RandomState, or cannot be used to
    initialize one, it will be assumed that is a similar object
    and returned.

    """
    if isinstance(random_state, (np.random.RandomState, np.random.Generator)):
        return random_state

    return np.random.RandomState(random_state)

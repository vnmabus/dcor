"""Utility functions"""

from __future__ import annotations

import enum
from typing import TypeVar, Union

import numpy as np
from typing_extensions import Protocol

try:
    from numpy.typing import ArrayLike
except ImportError:
    ArrayLike = np.ndarray  # type: ignore

NumberLikeT = TypeVar("NumberLikeT", bound="NumberLike")


class NumberLike(Protocol):

    def __add__(self: NumberLikeT, other: NumberLikeT) -> NumberLikeT:
        pass

    def __sub__(self: NumberLikeT, other: NumberLikeT) -> NumberLikeT:
        pass

    def __mul__(
        self: NumberLikeT,
        other: Union[float, NumberLikeT],
    ) -> NumberLikeT:
        pass


class CompileMode(enum.Enum):
    """
    Compilation mode of the algorithm.
    """

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
    """
    Rowwise mode of the algorithm.
    """

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


def _sqrt(x: np.ndarray) -> np.ndarray:
    """
    Return square root of an ndarray.

    This sqrt function for ndarrays tries to use the exponentiation operator
    if the objects stored do not supply a sqrt method.

    """
    x = np.clip(x, a_min=0, a_max=None)

    try:
        return np.sqrt(x)
    except (AttributeError, TypeError):
        exponent = 0.5

        try:
            exponent = np.take(x, 0).from_float(exponent)
        except AttributeError:
            pass

        return x ** exponent


def _transform_to_2d(t: ArrayLike) -> np.ndarray:
    """Convert vectors to column matrices, to always have a 2d shape."""
    t = np.asarray(t)

    dim = len(t.shape)
    assert dim <= 2

    if dim < 2:
        t = np.atleast_2d(t).T

    return t


def _can_be_double(x: np.ndarray) -> bool:
    """
    Return if the array can be safely converted to double.

    That happens when the dtype is a float with the same size of
    a double or narrower, or when is an integer that can be safely
    converted to double (if the roundtrip conversion works).

    """
    return ((np.issubdtype(x.dtype, np.floating) and
             x.dtype.itemsize <= np.dtype(float).itemsize) or
            (np.issubdtype(x.dtype, np.signedinteger) and
             np.can_cast(x, float)))


def _random_state_init(
    random_state: Union[None, int, ArrayLike, np.random.RandomState],
) -> np.random.RandomState:
    """
    Initialize a RandomState object.

    If the object is a RandomState, or cannot be used to
    initialize one, it will be assumed that is a similar object
    and returned.

    """
    try:
        random_state = np.random.RandomState(random_state)
    except TypeError:
        pass

    return random_state

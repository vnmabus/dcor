"""
Distance functions between sets of points.

This module provide functions that compute the distance between one or two
sets of points. The Scipy implementation is used when the conversion to
a double precision floating point number will not cause loss of precision.
"""

from __future__ import annotations

from typing import TypeVar

import numpy as np
import scipy.spatial as spatial

from dcor._utils import ArrayType, _sqrt, _transform_to_2d, array_namespace

from ._utils import _can_be_numpy_double

Array = TypeVar("Array", bound=ArrayType)


def _cdist_naive(x: Array, y: Array, exponent: float = 1) -> Array:
    """Pairwise distance, custom implementation."""
    xp = array_namespace(x, y)

    x = xp.asarray(x)
    y = xp.asarray(y)

    x = xp.expand_dims(x, axis=0)
    y = xp.expand_dims(y, axis=1)

    squared_norms = xp.sum(((x - y) ** 2), axis=-1)

    try:
        return squared_norms ** (exponent / 2)
    except TypeError:
        return _sqrt(squared_norms ** exponent)


def _pdist_scipy(
    x: np.typing.NDArray[float],
    exponent: float = 1,
) -> np.typing.NDArray[float]:
    """Pairwise distance between points in a set."""
    metric = 'euclidean'

    if exponent != 1:
        metric = 'sqeuclidean'

    # cdist is actually FASTER than pdist + squareform
    distances = spatial.distance.cdist(x, x, metric=metric)

    if exponent != 1:
        distances **= exponent / 2

    return distances


def _cdist_scipy(
    x: np.typing.NDArray[float],
    y: np.typing.NDArray[float],
    exponent: float = 1,
) -> np.typing.NDArray[float]:
    """Pairwise distance between the points in two sets."""
    metric = 'euclidean'

    if exponent != 1:
        metric = 'sqeuclidean'

    distances = spatial.distance.cdist(x, y, metric=metric)

    if exponent != 1:
        distances **= exponent / 2

    return distances


def _pdist(x: Array, exponent: float = 1) -> Array:
    """
    Pairwise distance between points in a set.

    As Scipy converts every value to double, this wrapper uses
    a less efficient implementation if the original dtype
    can not be converted to double.

    """
    if _can_be_numpy_double(x):
        return _pdist_scipy(x, exponent)

    return _cdist_naive(x, x, exponent)


def _cdist(x: Array, y: Array, exponent: float = 1) -> Array:
    """
    Pairwise distance between points in two sets.

    As Scipy converts every value to double, this wrapper uses
    a less efficient implementation if the original dtype
    can not be converted to double.

    """
    if _can_be_numpy_double(x) and _can_be_numpy_double(y):
        return _cdist_scipy(x, y, exponent)

    return _cdist_naive(x, y, exponent)


def pairwise_distances(
    x: Array,
    y: Array | None = None,
    *,
    exponent: float = 1,
) -> Array:
    r"""
    Pairwise distance between points.

    Return the pairwise distance between points in two sets, or
    in the same set if only one set is passed.

    Args:
        x: An :math:`n \times m` array of :math:`n` observations in
            a :math:`m`-dimensional space.
        y: An :math:`l \times m` array of :math:`l` observations in
            a :math:`m`-dimensional space. If None, the distances will
            be computed between the points in :math:`x`.
        exponent: Exponent of the Euclidean distance.

    Returns:
        A :math:`n \times l` matrix where the :math:`(i, j)`-th entry is the
        distance between :math:`x[i]` and :math:`y[j]`.

    Examples:
        >>> import numpy as np
        >>> import dcor
        >>> a = np.array([[1, 2, 3, 4],
        ...               [5, 6, 7, 8],
        ...               [9, 10, 11, 12],
        ...               [13, 14, 15, 16]])
        >>> b = np.array([[16, 15, 14, 13],
        ...               [12, 11, 10, 9],
        ...               [8, 7, 6, 5],
        ...               [4, 3, 2, 1]])
        >>> dcor.distances.pairwise_distances(a)
        array([[ 0.,  8., 16., 24.],
               [ 8.,  0.,  8., 16.],
               [16.,  8.,  0.,  8.],
               [24., 16.,  8.,  0.]])
        >>> dcor.distances.pairwise_distances(a, b)
        array([[24.41311123, 16.61324773,  9.16515139,  4.47213595],
               [16.61324773,  9.16515139,  4.47213595,  9.16515139],
               [ 9.16515139,  4.47213595,  9.16515139, 16.61324773],
               [ 4.47213595,  9.16515139, 16.61324773, 24.41311123]])

    """
    if y is None or y is x:
        x, = _transform_to_2d(x)
        return _pdist(x, exponent=exponent)

    x, y = _transform_to_2d(x, y)
    return _cdist(x, y, exponent=exponent)

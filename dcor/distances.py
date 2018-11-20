"""
Distance functions between sets of points.

This module provide functions that compute the distance between one or two
sets of points. The Scipy implementation is used when the conversion to
a double precision floating point number will not cause loss of precision.
"""

from __future__ import absolute_import, division, print_function

from dcor._utils import _transform_to_2d
import numpy as _np
import scipy.spatial as _spatial

from ._utils import _can_be_double


def _cdist_naive(x, y, exponent=1):
    """Pairwise distance, custom implementation."""
    squared_norms = ((x[_np.newaxis, :, :] - y[:, _np.newaxis, :]) ** 2).sum(2)

    exponent = exponent / 2
    try:
        exponent = squared_norms.take(0).from_float(exponent)
    except AttributeError:
        pass

    return squared_norms ** exponent


def _pdist_scipy(x, exponent=1):
    """Pairwise distance between points in a set."""
    metric = 'euclidean'

    if exponent != 1:
        metric = 'sqeuclidean'

    distances = _spatial.distance.pdist(x, metric=metric)
    distances = _spatial.distance.squareform(distances)

    if exponent != 1:
        distances **= exponent / 2

    return distances


def _cdist_scipy(x, y, exponent=1):
    """Pairwise distance between the points in two sets."""
    metric = 'euclidean'

    if exponent != 1:
        metric = 'sqeuclidean'

    distances = _spatial.distance.cdist(x, y, metric=metric)

    if exponent != 1:
        distances **= exponent / 2

    return distances


def _pdist(x, exponent=1):
    """
    Pairwise distance between points in a set.

    As Scipy converts every value to double, this wrapper uses
    a less efficient implementation if the original dtype
    can not be converted to double.

    """
    if _can_be_double(x):
        return _pdist_scipy(x, exponent)
    else:
        return _cdist_naive(x, x, exponent)


def _cdist(x, y, exponent=1):
    """
    Pairwise distance between points in two sets.

    As Scipy converts every value to double, this wrapper uses
    a less efficient implementation if the original dtype
    can not be converted to double.

    """
    if _can_be_double(x) and _can_be_double(y):
        return _cdist_scipy(x, y, exponent)
    else:
        return _cdist_naive(x, y, exponent)


def pairwise_distances(x, y=None, **kwargs):
    r"""
    pairwise_distances(x, y=None, *, exponent=1)

    Pairwise distance between points.

    Return the pairwise distance between points in two sets, or
    in the same set if only one set is passed.

    Parameters
    ----------
    x: array_like
        An :math:`n \times m` array of :math:`n` observations in
        a :math:`m`-dimensional space.
    y: array_like
        An :math:`l \times m` array of :math:`l` observations in
        a :math:`m`-dimensional space. If None, the distances will
        be computed between the points in :math:`x`.
    exponent: float
        Exponent of the Euclidean distance.

    Returns
    -------
    numpy ndarray
        A :math:`n \times l` matrix where the :math:`(i, j)`-th entry is the
        distance between :math:`x[i]` and :math:`y[j]`.

    Examples
    --------
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
    x = _transform_to_2d(x)

    if y is None or y is x:
        return _pdist(x, **kwargs)
    else:
        y = _transform_to_2d(y)
        return _cdist(x, y, **kwargs)

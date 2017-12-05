"""
Distance functions between sets of points.

This module provide functions that compute the distance between one or two
sets of points. The Scipy implementation is used when the conversion to
a double precision floating point number will not cause loss of precision.
"""

from __future__ import absolute_import, division, print_function

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

from __future__ import absolute_import, division, print_function

import scipy.spatial
import numpy as _np


def _cdist_naive(x, y, exponent=1):
    '''
    Pairwise distance, custom implementation.
    '''

    squared_norms = ((x[_np.newaxis, :, :] - y[:, _np.newaxis, :]) ** 2).sum(2)

    exponent = exponent / 2
    try:
        exponent = squared_norms.take(0).from_float(exponent)
    except AttributeError:
        pass

    return squared_norms ** exponent


def _pdist_scipy(x, exponent=1):
    '''
    Pairwise distance between points in a set.
    '''

    metric = 'euclidean'

    if exponent != 1:
        metric = 'sqeuclidean'

    distances = scipy.spatial.distance.pdist(x, metric=metric)
    distances = scipy.spatial.distance.squareform(distances)

    if exponent != 1:
        distances **= exponent / 2

    return distances


def _cdist_scipy(x, y, exponent=1):
    '''
    Pairwise distance between the points in two sets.
    '''

    metric = 'euclidean'

    if exponent != 1:
        metric = 'sqeuclidean'

    distances = scipy.spatial.distance.cdist(x, y, metric=metric)

    if exponent != 1:
        distances **= exponent / 2

    return distances


def _can_be_double(x):
    '''
    Return if the array can be safely converted to double in
    intermediate steps.

    That happens when the dtype is a float with the same size of
    a double or narrower, or when is an integer that can be safely
    converted to double (if the roundtrip conversion works).
    '''

    return ((_np.issubdtype(x.dtype, float) and
            x.dtype.itemsize <= _np.dtype(float).itemsize) or
            (_np.issubdtype(x.dtype, int) and
            _np.all(_np.asfarray(x).astype(dtype=x.dtype) == x)))


def _pdist(x, exponent=1):
    '''
    As Scipy converts every value to double, this wrapper uses
    a less efficient implementation if the original dtype
    can not be converted to double.
    '''

    if _can_be_double(x):
        return _pdist_scipy(x, exponent)
    else:
        return _cdist_naive(x, x, exponent)


def _cdist(x, y, exponent=1):
    '''
    As Scipy converts every value to double, this wrapper uses
    a less efficient implementation if the original dtype
    can not be converted to double.
    '''

    if _can_be_double(x) and _can_be_double(y):
        return _cdist_scipy(x, y, exponent)
    else:
        return _cdist_naive(x, y, exponent)

from __future__ import absolute_import, division, print_function

import scipy.spatial


def _optimiced_distance(**kwargs):
    '''
    Decorator to provide optimized versions of functions for
    cdist and pdist.
    '''

    def decorator(distance_function):
        distance_function._dcor_optimiced_distance_args = kwargs
        return distance_function

    return decorator


@_optimiced_distance(metric='euclidean')
def euclidean(u, v):
    '''
    Computes the euclidean distance between the vectors :math:`u` and
    :math:`v`.

    The Euclidean distance between vectors :math:`u = (u_1, \ldots, u_n)` and
    :math:`v = (v_1, \ldots, v_n)` is

    .. math::
        d(u, v) = || u - v ||_2 = \\sqrt{(u_1 - v_1)^2 + \ldots +
        (u_n - v_n)^2}

    Parameters
    ----------
    u: array_like of length N
        First vector.
    v: array_like of length N
        Second vector.

    Returns
    -------
    float
        Euclidean distance between the two vectors.

    Examples
    --------
    >>> import numpy as np
    >>> import dcor
    >>> a = np.array([1, 1, 1])
    >>> b = np.array([0, 0, 0])
    >>> dcor.distances.euclidean(a, b) # doctest: +ELLIPSIS
    1.7320508...
    '''
    return scipy.spatial.distance.euclidean(u, v)


def minkowski(p):
    '''
    Returns a function that computes the :math:`p`-norm of the diference of
    two vectors :math:`u` and :math:`v`, also called the Minkowski distance.

    The Minkowski distance between vectors :math:`u = (u_1, \ldots, u_n)` and
    :math:`v = (v_1, \ldots, v_n)` is

    .. math::
        d(u, v) = || u - v ||_p = \\sqrt[p]{(u_1 - v_1)^p + \ldots +
        (u_n - v_n)^p}

    Parameters
    ----------
    p: float
        Exponent of the norm.

    Returns
    -------
    callable
        Function implementing the Minkowski distance between the two vectors.

    Examples
    --------
    >>> import numpy as np
    >>> import dcor
    >>> a = np.array([1, 1, 1])
    >>> b = np.array([0, 0, 0])
    >>> dcor.distances.minkowski(2)(a, b) # doctest: +ELLIPSIS
    1.7320508...
    >>> dcor.distances.minkowski(1)(a, b) # doctest: +ELLIPSIS
    3.0
    '''

    @_optimiced_distance(metric='minkowski', p=p)
    def minkowski(u, v):
        return scipy.spatial.distance.minkowski(u, v, p)

    return minkowski


def _get_args(metric):
    '''
    Return the dictionary of additional arguments passed to pdist or cdist.
    '''

    try:
        args = metric._dcor_optimiced_distance_args
    except AttributeError:
        args = {'metric': metric}
    return args


def _pdist(x, metric=None):
    '''
    Pairwise distance between points in a set.
    '''

    if metric is None:
        metric = euclidean

    args = _get_args(metric)

    distances = scipy.spatial.distance.pdist(x, **args)

    return scipy.spatial.distance.squareform(distances)


def _cdist(x, y, metric=None):
    '''
    Pairwise distance between the points in two sets.
    '''

    if metric is None:
        metric = euclidean

    args = _get_args(metric)

    distances = scipy.spatial.distance.cdist(x, y, **args)

    return distances

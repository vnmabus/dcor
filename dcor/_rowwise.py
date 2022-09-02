"""
Functions to compute a pairwise dependency measure.
"""

import numpy as np

from . import _dcor
from ._fast_dcov_avl import _rowwise_distance_covariance_sqr_avl_generic
from ._utils import RowwiseMode as RowwiseMode, _sqrt


def _generate_rowwise_distance_covariance_sqr(unbiased):
    def rowwise_distance_covariance_sqr(
            x, y, exponent=1, *,
            method=_dcor.DistanceCovarianceMethod.AUTO,
            **kwargs):

        if not _dcor._can_use_fast_algorithm(x[0], y[0],
                                             exponent=exponent):
            return NotImplemented

        if (method in (_dcor.DistanceCovarianceMethod.AUTO,
                       _dcor.DistanceCovarianceMethod.AVL)):
            return _rowwise_distance_covariance_sqr_avl_generic(
                x, y, unbiased=unbiased, **kwargs)
        else:
            return NotImplemented

    return rowwise_distance_covariance_sqr


_dcor.distance_covariance_sqr.rowwise_function = (
    _generate_rowwise_distance_covariance_sqr(unbiased=False))

_dcor.u_distance_covariance_sqr.rowwise_function = (
    _generate_rowwise_distance_covariance_sqr(unbiased=True))


def _rowwise_distance_covariance(*args, **kwargs):

    res_covs = _dcor.distance_covariance_sqr.rowwise_function(*args, **kwargs)
    if res_covs is NotImplemented:
        return NotImplemented

    return _sqrt(res_covs)


_dcor.distance_covariance.rowwise_function = _rowwise_distance_covariance


def _generate_rowwise_distance_correlation_sqr(unbiased):
    def rowwise_distance_correlation_sqr(x, y, **kwargs):

        cov_fun = (_dcor.u_distance_covariance_sqr if unbiased
                   else _dcor.distance_covariance_sqr)

        n_comps = len(x)

        concat_x = np.concatenate((x, x, y))
        concat_y = np.concatenate((y, x, y))

        res_covs = cov_fun.rowwise_function(concat_x, concat_y, **kwargs)
        if res_covs is NotImplemented:
            return NotImplemented

        cov = res_covs[:n_comps]
        x_std = _sqrt(res_covs[n_comps:2 * n_comps])
        y_std = _sqrt(res_covs[2 * n_comps:])

        with np.errstate(divide='ignore', invalid='ignore'):
            corr_sqr = cov / x_std / y_std

        corr_sqr[np.isnan(corr_sqr)] = 0

        return corr_sqr

    return rowwise_distance_correlation_sqr


_dcor.distance_correlation_sqr.rowwise_function = (
    _generate_rowwise_distance_correlation_sqr(unbiased=False))

_dcor.u_distance_correlation_sqr.rowwise_function = (
    _generate_rowwise_distance_correlation_sqr(unbiased=True))


def _rowwise_distance_correlation(*args, **kwargs):

    res_corrs = _dcor.distance_correlation_sqr.rowwise_function(
        *args, **kwargs)
    if res_corrs is NotImplemented:
        return NotImplemented

    return _sqrt(res_corrs)


_dcor.distance_correlation.rowwise_function = _rowwise_distance_correlation


def rowwise(function, x, y, *, rowwise_mode=False,
            **kwargs):
    """
    Computes a dependency measure between pairs of elements.

    It will use an optimized implementation if one is available.

    Parameters
    ----------
    function: callable
        Dependency measure function.
    x: iterable of array_like
        First list of random vectors. The columns of each vector correspond
        with the individual random variables while the rows are individual
        instances of the random vector.
    y: array_like
        Second list of random vectors. The columns of each vector correspond
        with the individual random variables while the rows are individual
        instances of the random vector.
    rowwise_mode: RowwiseMode
        Mode of rowwise computations.
    kwargs: dictionary
        Additional options necessary.

    Returns
    -------
    numpy ndarray
        A length :math:`n` vector where the :math:`i`-th entry is the
        dependency between :math:`x[i]` and :math:`y[i]`.

    Examples
    --------
    >>> import numpy as np
    >>> import dcor

    The following example shows two computations of distance covariance
    between random variables. This has an optimized implementation using
    multiple cores if available.

    >>> a = [np.array([1., 2., 3., 4., 5. ,6.]),
    ...      np.array([7., 8., 9., 10., 11., 12.])
    ...     ]
    >>> b = [np.array([1., 4., 9., 16., 25., 36.]),
    ...      np.array([1., 3., 6., 8., 10., 12.])
    ...     ]
    >>> dcor.rowwise(dcor.distance_covariance, a, b)
    array([3.45652005, 1.95789002])

    The following example shows two computations of distance correlation
    between random vectors of length 2. Currently there is no optimized
    implementation for the random vector case, so it will be equivalent to
    calling map.

    >>> a = [np.array([[1., 1.],
    ...                [2., 4.],
    ...                [3., 8.],
    ...                [4., 16.]]),
    ...      np.array([[9., 10.],
    ...                [11., 12.],
    ...                [13., 14.],
    ...                [15., 16.]])
    ...     ]
    >>> b = [np.array([[0., 1.],
    ...                [3., 1.],
    ...                [6., 2.],
    ...                [9., 3.]]),
    ...      np.array([[5., 1.],
    ...                [8., 1.],
    ...                [13., 1.],
    ...                [21., 1.]])
    ...     ]
    >>> dcor.rowwise(dcor.distance_correlation, a, b)
    array([0.98182263, 0.98320103])

    """

    if rowwise_mode is not RowwiseMode.NAIVE:

        rowwise_function = getattr(function, 'rowwise_function', None)
        if rowwise_function:
            result = rowwise_function(x, y, **kwargs)
            if result is not NotImplemented:
                return result

    if rowwise_mode is RowwiseMode.OPTIMIZED:
        raise NotImplementedError(
            "There is not an optimized rowwise implementation")

    return np.array([function(x_elem, y_elem, **kwargs)
                     for x_elem, y_elem in zip(x, y)])

"""
Functions to compute a pairwise dependency measure.
"""

import numpy as np

from . import _dcor
from ._fast_dcov_avl import _rowwise_distance_covariance_sqr_avl_generic
from ._utils import RowwiseMode


def _generate_rowwise_distance_covariance_sqr(unbiased):
    def rowwise_distance_covariance_sqr(
            x, y, *,
            method=_dcor.DistanceCovarianceMethod.AUTO,
            **kwargs):

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


def rowwise(function, x, y, *, rowwise_mode=False,
            **kwargs):
    """
    Computes a dependency measure between pairs of elements.

    Parameters
    ----------
    function: Dependency measure function.
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
    >>> a = [np.array([[1, 1],
    ...                [2, 4],
    ...                [3, 8],
    ...                [4, 16]]),
    ...      np.array([[9, 10],
    ...                [11, 12],
    ...                [13, 14],
    ...                [15, 16]])
    ...     ]
    >>> b = [np.array([[0, 1],
    ...                [3, 1],
    ...                [6, 2],
    ...                [9, 3]]),
    ...      np.array([[5, 1],
    ...                [8, 1],
    ...                [13, 1],
    ...                [21, 1]])
    ...     ]
    >>> dcor.rowwise(dcor.distance_correlation, a, b)
    array([0.98182263, 0.98320103])

    A pool object can be used to improve performance for a large
    number of computations:

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

    return np.array([function(x_elem, y_elem, *kwargs)
                     for x_elem, y_elem in zip(x, y)])

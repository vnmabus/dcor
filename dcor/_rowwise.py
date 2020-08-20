"""
Functions to compute a pairwise dependency measure.
"""

import functools

import numpy as np


def rowwise(function, x, y, *, force_naive=False,
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
    force_naive: bool
        Force the use of the naive implementation even when the function
        offers an optimized alternative.
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
    rowwise_function = getattr(function, 'rowwise_function', None)
    if rowwise_function and not force_naive:
        result = rowwise_function(x, y, **kwargs)
        if result is not NotImplemented:
            return result

    return np.array([function(x_elem, y_elem, *kwargs)
                     for x_elem, y_elem in zip(x, y)])

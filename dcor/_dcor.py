"""
Distance correlation and covariance.

This module contains functions to compute statistics related to the
distance covariance and distance correlation
:cite:`b-distance_correlation`.

References
----------
.. bibliography:: ../refs.bib
   :labelprefix: B
   :keyprefix: b-

"""

from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import collections
import math

from dcor._dcor_internals import _af_inv_scaled
import numpy as np

from ._dcor_internals import _distance_matrix, _u_distance_matrix
from ._dcor_internals import mean_product, u_product
from ._utils import _sqrt, _jit


Stats = collections.namedtuple('Stats', ['covariance_xy', 'correlation_xy',
                               'variance_x', 'variance_y'])


def _distance_covariance_sqr_naive(x, y, exponent=1):
    """
    Naive biased estimator for distance covariance.

    Computes the unbiased estimator for distance covariance between two
    matrices, using an :math:`O(N^2)` algorithm.
    """
    a = _distance_matrix(x, exponent=exponent)
    b = _distance_matrix(y, exponent=exponent)

    return mean_product(a, b)


def _u_distance_covariance_sqr_naive(x, y, exponent=1):
    """
    Naive unbiased estimator for distance covariance.

    Computes the unbiased estimator for distance covariance between two
    matrices, using an :math:`O(N^2)` algorithm.
    """
    a = _u_distance_matrix(x, exponent=exponent)
    b = _u_distance_matrix(y, exponent=exponent)

    return u_product(a, b)


def _distance_sqr_stats_naive_generic(x, y, matrix_centered, product,
                                      exponent=1):
    """Compute generic squared stats."""
    a = matrix_centered(x, exponent=exponent)
    b = matrix_centered(y, exponent=exponent)

    covariance_xy_sqr = product(a, b)
    variance_x_sqr = product(a, a)
    variance_y_sqr = product(b, b)

    denominator_sqr = np.absolute(variance_x_sqr * variance_y_sqr)
    denominator = _sqrt(denominator_sqr)

    # Comparisons using a tolerance can change results if the
    # covariance has a similar order of magnitude
    if denominator == 0.0:
        correlation_xy_sqr = 0.0
    else:
        correlation_xy_sqr = covariance_xy_sqr / denominator

    return Stats(covariance_xy=covariance_xy_sqr,
                 correlation_xy=correlation_xy_sqr,
                 variance_x=variance_x_sqr,
                 variance_y=variance_y_sqr)


def _distance_correlation_sqr_naive(x, y, exponent=1):
    """Biased distance correlation estimator between two matrices."""
    return _distance_sqr_stats_naive_generic(
        x, y,
        matrix_centered=_distance_matrix,
        product=mean_product,
        exponent=exponent).correlation_xy


def _u_distance_correlation_sqr_naive(x, y, exponent=1):
    """Bias-corrected distance correlation estimator between two matrices."""
    return _distance_sqr_stats_naive_generic(
        x, y,
        matrix_centered=_u_distance_matrix,
        product=u_product,
        exponent=exponent).correlation_xy


def _is_random_variable(x):
    """
    Check if the matrix x correspond to a random variable.

    The matrix is considered a random variable if it is a vector
    or a matrix corresponding to a column vector. Otherwise,
    the matrix correspond to a random vector.
    """
    return len(x.shape) == 1 or x.shape[1] == 1


def _can_use_fast_algorithm(x, y, exponent=1):
    """
    Check if the fast algorithm for distance stats can be used.

    The fast algorithm has complexity :math:`O(NlogN)`, better than the
    complexity of the naive algorithm (:math:`O(N^2)`).

    The algorithm can only be used for random variables (not vectors) where
    the number of instances is greater than 3. Also, the exponent must be 1.

    """
    return (_is_random_variable(x) and _is_random_variable(y) and
            x.shape[0] > 3 and y.shape[0] > 3 and exponent == 1)


@_jit
def _dyad_update(y, c):  # pylint:disable=too-many-locals
    # This function has many locals so it can be compared
    # with the original algorithm.
    """
    Inner function of the fast distance covariance.

    This function is compiled because otherwise it would become
    a bottleneck.

    """
    n = y.shape[0]
    gamma = np.zeros(n, dtype=c.dtype)

    # Step 1: get the smallest l such that n <= 2^l
    l_max = int(math.ceil(np.log2(n)))

    # Step 2: assign s(l, k) = 0
    s_len = 2 ** (l_max + 1)
    s = np.zeros(s_len, dtype=c.dtype)

    pos_sums = np.arange(l_max)
    pos_sums[:] = 2 ** (l_max - pos_sums)
    pos_sums = np.cumsum(pos_sums)

    # Step 3: iteration
    for i in range(1, n):

        # Step 3.a: update s(l, k)
        for l in range(l_max):
            k = int(math.ceil(y[i - 1] / 2 ** l))
            pos = k - 1

            if l > 0:
                pos += pos_sums[l - 1]

            s[pos] += c[i - 1]

        # Steps 3.b and 3.c
        for l in range(l_max):
            k = int(math.floor((y[i] - 1) / 2 ** l))
            if k / 2 > math.floor(k / 2):
                pos = k - 1
                if l > 0:
                    pos += pos_sums[l - 1]

                gamma[i] = gamma[i] + s[pos]

    return gamma


def _partial_sum_2d(x, y, c):  # pylint:disable=too-many-locals
    # This function has many locals so it can be compared
    # with the original algorithm.
    x = np.asarray(x)
    y = np.asarray(y)
    c = np.asarray(c)

    n = x.shape[0]

    # Step 1: rearrange x, y and c so x is in ascending order
    temp = range(n)

    ix0 = np.argsort(x)
    ix = np.zeros(n, dtype=int)
    ix[ix0] = temp

    x = x[ix0]
    y = y[ix0]
    c = c[ix0]

    # Step 2
    iy0 = np.argsort(y)
    iy = np.zeros(n, dtype=int)
    iy[iy0] = temp

    y = iy + 1

    # Step 3
    sy = np.cumsum(c[iy0]) - c[iy0]

    # Step 4
    sx = np.cumsum(c) - c

    # Step 5
    c_dot = np.sum(c)

    # Step 6
    y = np.asarray(y)
    c = np.asarray(c)
    gamma1 = _dyad_update(y, c)

    # Step 7
    gamma = c_dot - c - 2 * sy[iy] - 2 * sx + 4 * gamma1
    gamma = gamma[ix]

    return gamma


def _distance_covariance_sqr_fast_generic(
        x, y, unbiased=False):  # pylint:disable=too-many-locals
    # This function has many locals so it can be compared
    # with the original algorithm.
    """Fast algorithm for the squared distance covariance."""
    x = np.asarray(x)
    y = np.asarray(y)

    x = np.ravel(x)
    y = np.ravel(y)

    n = x.shape[0]
    assert n > 3
    assert n == y.shape[0]
    temp = range(n)

    # Step 1
    ix0 = np.argsort(x)
    vx = x[ix0]

    ix = np.zeros(n, dtype=int)
    ix[ix0] = temp

    iy0 = np.argsort(y)
    vy = y[iy0]

    iy = np.zeros(n, dtype=int)
    iy[iy0] = temp

    # Step 2
    sx = np.cumsum(vx)
    sy = np.cumsum(vy)

    # Step 3
    alpha_x = ix
    alpha_y = iy

    beta_x = sx[ix] - vx[ix]
    beta_y = sy[iy] - vy[iy]

    # Step 4
    x_dot = np.sum(x)
    y_dot = np.sum(y)

    # Step 5
    a_i_dot = x_dot + (2 * alpha_x - n) * x - 2 * beta_x
    b_i_dot = y_dot + (2 * alpha_y - n) * y - 2 * beta_y

    sum_ab = np.sum(a_i_dot * b_i_dot)

    # Step 6
    a_dot_dot = 2 * np.sum(alpha_x * x) - 2 * np.sum(beta_x)
    b_dot_dot = 2 * np.sum(alpha_y * y) - 2 * np.sum(beta_y)

    # Step 7
    gamma_1 = _partial_sum_2d(x, y, np.ones(n, dtype=x.dtype))
    gamma_x = _partial_sum_2d(x, y, x)
    gamma_y = _partial_sum_2d(x, y, y)
    gamma_xy = _partial_sum_2d(x, y, x * y)

    # Step 8
    aijbij = np.sum(x * y * gamma_1 + gamma_xy - x * gamma_y - y * gamma_x)

    if unbiased:
        d3 = (n - 3)
        d2 = (n - 2)
        d1 = (n - 1)
    else:
        d3 = d2 = d1 = n

    # Step 9
    d_cov = (aijbij / n / d3 - 2 * sum_ab / n / d2 / d3 +
             a_dot_dot / n * b_dot_dot / d1 / d2 / d3)

    return d_cov


def _distance_covariance_sqr_fast(x, y):
    """Fast algorithm for the biased squared distance covariance"""
    return _distance_covariance_sqr_fast_generic(x, y, unbiased=False)


def _u_distance_covariance_sqr_fast(x, y):
    """Fast algorithm for the unbiased squared distance covariance"""
    return _distance_covariance_sqr_fast_generic(x, y, unbiased=True)


def _distance_stats_sqr_fast_generic(x, y, dcov_function):
    """Compute the distance stats using the fast algorithm."""
    covariance_xy_sqr = dcov_function(x, y)
    variance_x_sqr = dcov_function(x, x)
    variance_y_sqr = dcov_function(y, y)
    denominator_sqr_signed = variance_x_sqr * variance_y_sqr
    denominator_sqr = np.absolute(denominator_sqr_signed)
    denominator = _sqrt(denominator_sqr)

    # Comparisons using a tolerance can change results if the
    # covariance has a similar order of magnitude
    if denominator == 0.0:
        correlation_xy_sqr = denominator.dtype.type(0)
    else:
        correlation_xy_sqr = covariance_xy_sqr / denominator

    return Stats(covariance_xy=covariance_xy_sqr,
                 correlation_xy=correlation_xy_sqr,
                 variance_x=variance_x_sqr,
                 variance_y=variance_y_sqr)


def _distance_stats_sqr_fast(x, y):
    """Compute the biased distance stats using the fast algorithm."""
    return _distance_stats_sqr_fast_generic(x, y,
                                            _distance_covariance_sqr_fast)


def _u_distance_stats_sqr_fast(x, y):
    """Compute the bias-corrected distance stats using the fast algorithm."""
    return _distance_stats_sqr_fast_generic(x, y,
                                            _u_distance_covariance_sqr_fast)


def _distance_correlation_sqr_fast(x, y):
    """Fast algorithm for bias-corrected squared distance correlation."""
    return _distance_stats_sqr_fast(x, y).correlation_xy


def _u_distance_correlation_sqr_fast(x, y):
    """Fast algorithm for bias-corrected squared distance correlation."""
    return _u_distance_stats_sqr_fast(x, y).correlation_xy


def distance_covariance_sqr(x, y, **kwargs):
    """
    distance_covariance_sqr(x, y, *, exponent=1)

    Computes the usual (biased) estimator for the squared distance covariance
    between two random vectors.

    Parameters
    ----------
    x: array_like
        First random vector. The columns correspond with the individual random
        variables while the rows are individual instances of the random vector.
    y: array_like
        Second random vector. The columns correspond with the individual random
        variables while the rows are individual instances of the random vector.
    exponent: float
        Exponent of the Euclidean distance, in the range :math:`(0, 2)`.
        Equivalently, it is twice the Hurst parameter of fractional Brownian
        motion.

    Returns
    -------
    numpy scalar
        Biased estimator of the squared distance covariance.

    See Also
    --------
    distance_covariance
    u_distance_covariance_sqr

    Notes
    -----
    The algorithm uses the fast distance covariance algorithm proposed in
    :cite:`b-fast_distance_correlation` when possible.

    Examples
    --------
    >>> import numpy as np
    >>> import dcor
    >>> a = np.array([[1, 2, 3, 4],
    ...               [5, 6, 7, 8],
    ...               [9, 10, 11, 12],
    ...               [13, 14, 15, 16]])
    >>> b = np.array([[1], [0], [0], [1]])
    >>> dcor.distance_covariance_sqr(a, a)
    52.0
    >>> dcor.distance_covariance_sqr(a, b)
    1.0
    >>> dcor.distance_covariance_sqr(b, b)
    0.25
    >>> dcor.distance_covariance_sqr(a, b, exponent=0.5) # doctest: +ELLIPSIS
    0.3705904...

    """
    if _can_use_fast_algorithm(x, y, **kwargs):
        return _distance_covariance_sqr_fast(x, y)
    else:
        return _distance_covariance_sqr_naive(x, y, **kwargs)


def u_distance_covariance_sqr(x, y, **kwargs):
    """
    u_distance_covariance_sqr(x, y, *, exponent=1)

    Computes the unbiased estimator for the squared distance covariance
    between two random vectors.

    Parameters
    ----------
    x: array_like
        First random vector. The columns correspond with the individual random
        variables while the rows are individual instances of the random vector.
    y: array_like
        Second random vector. The columns correspond with the individual random
        variables while the rows are individual instances of the random vector.
    exponent: float
        Exponent of the Euclidean distance, in the range :math:`(0, 2)`.
        Equivalently, it is twice the Hurst parameter of fractional Brownian
        motion.

    Returns
    -------
    numpy scalar
        Value of the unbiased estimator of the squared distance covariance.

    See Also
    --------
    distance_covariance
    distance_covariance_sqr

    Notes
    -----
    The algorithm uses the fast distance covariance algorithm proposed in
    :cite:`b-fast_distance_correlation` when possible.

    Examples
    --------
    >>> import numpy as np
    >>> import dcor
    >>> a = np.array([[1, 2, 3, 4],
    ...               [5, 6, 7, 8],
    ...               [9, 10, 11, 12],
    ...               [13, 14, 15, 16]])
    >>> b = np.array([[1], [0], [0], [1]])
    >>> dcor.u_distance_covariance_sqr(a, a) # doctest: +ELLIPSIS
    42.6666666...
    >>> dcor.u_distance_covariance_sqr(a, b) # doctest: +ELLIPSIS
    -2.6666666...
    >>> dcor.u_distance_covariance_sqr(b, b) # doctest: +ELLIPSIS
    0.6666666...
    >>> dcor.u_distance_covariance_sqr(a, b, exponent=0.5) # doctest: +ELLIPSIS
    -0.2996598...

    """
    if _can_use_fast_algorithm(x, y, **kwargs):
        return _u_distance_covariance_sqr_fast(x, y)
    else:
        return _u_distance_covariance_sqr_naive(x, y, **kwargs)


def distance_covariance(x, y, **kwargs):
    """
    distance_covariance(x, y, *, exponent=1)

    Computes the usual (biased) estimator for the distance covariance
    between two random vectors.

    Parameters
    ----------
    x: array_like
        First random vector. The columns correspond with the individual random
        variables while the rows are individual instances of the random vector.
    y: array_like
        Second random vector. The columns correspond with the individual random
        variables while the rows are individual instances of the random vector.
    exponent: float
        Exponent of the Euclidean distance, in the range :math:`(0, 2)`.
        Equivalently, it is twice the Hurst parameter of fractional Brownian
        motion.

    Returns
    -------
    numpy scalar
        Biased estimator of the distance covariance.

    See Also
    --------
    distance_covariance_sqr
    u_distance_covariance_sqr

    Notes
    -----
    The algorithm uses the fast distance covariance algorithm proposed in
    :cite:`b-fast_distance_correlation` when possible.

    Examples
    --------
    >>> import numpy as np
    >>> import dcor
    >>> a = np.array([[1, 2, 3, 4],
    ...               [5, 6, 7, 8],
    ...               [9, 10, 11, 12],
    ...               [13, 14, 15, 16]])
    >>> b = np.array([[1], [0], [0], [1]])
    >>> dcor.distance_covariance(a, a) # doctest: +ELLIPSIS
    7.2111025...
    >>> dcor.distance_covariance(a, b)
    1.0
    >>> dcor.distance_covariance(b, b)
    0.5
    >>> dcor.distance_covariance(a, b, exponent=0.5)
    0.6087614...

    """
    return _sqrt(distance_covariance_sqr(x, y, **kwargs))


def distance_stats_sqr(x, y, **kwargs):
    """
    distance_stats_sqr(x, y, *, exponent=1)

    Computes the usual (biased) estimators for the squared distance covariance
    and squared distance correlation between two random vectors, and the
    individual squared distance variances.

    Parameters
    ----------
    x: array_like
        First random vector. The columns correspond with the individual random
        variables while the rows are individual instances of the random vector.
    y: array_like
        Second random vector. The columns correspond with the individual random
        variables while the rows are individual instances of the random vector.
    exponent: float
        Exponent of the Euclidean distance, in the range :math:`(0, 2)`.
        Equivalently, it is twice the Hurst parameter of fractional Brownian
        motion.

    Returns
    -------
    Stats
        Squared distance covariance, squared distance correlation,
        squared distance variance of the first random vector and
        squared distance variance of the second random vector.

    See Also
    --------
    distance_covariance_sqr
    distance_correlation_sqr

    Notes
    -----
    It is less efficient to compute the statistics separately, rather than
    using this function, because some computations can be shared.

    The algorithm uses the fast distance covariance algorithm proposed in
    :cite:`b-fast_distance_correlation` when possible.

    Examples
    --------
    >>> import numpy as np
    >>> import dcor
    >>> a = np.array([[1, 2, 3, 4],
    ...               [5, 6, 7, 8],
    ...               [9, 10, 11, 12],
    ...               [13, 14, 15, 16]])
    >>> b = np.array([[1], [0], [0], [1]])
    >>> dcor.distance_stats_sqr(a, a) # doctest: +NORMALIZE_WHITESPACE
    Stats(covariance_xy=52.0, correlation_xy=1.0, variance_x=52.0,
    variance_y=52.0)
    >>> dcor.distance_stats_sqr(a, b) # doctest: +NORMALIZE_WHITESPACE
    Stats(covariance_xy=1.0, correlation_xy=0.2773500...,
    variance_x=52.0, variance_y=0.25)
    >>> dcor.distance_stats_sqr(b, b) # doctest: +NORMALIZE_WHITESPACE
    Stats(covariance_xy=0.25, correlation_xy=1.0, variance_x=0.25,
    variance_y=0.25)
    >>> dcor.distance_stats_sqr(a, b, exponent=0.5) # doctest: +ELLIPSIS
    ...                                 # doctest: +NORMALIZE_WHITESPACE
    Stats(covariance_xy=0.3705904..., correlation_xy=0.4493308...,
    variance_x=2.7209220..., variance_y=0.25)

    """
    if _can_use_fast_algorithm(x, y, **kwargs):
        return _distance_stats_sqr_fast(x, y)
    else:
        return _distance_sqr_stats_naive_generic(
            x, y,
            matrix_centered=_distance_matrix,
            product=mean_product,
            **kwargs)


def u_distance_stats_sqr(x, y, **kwargs):
    """
    u_distance_stats_sqr(x, y, *, exponent=1)

    Computes the unbiased estimators for the squared distance covariance
    and squared distance correlation between two random vectors, and the
    individual squared distance variances.

    Parameters
    ----------
    x: array_like
        First random vector. The columns correspond with the individual random
        variables while the rows are individual instances of the random vector.
    y: array_like
        Second random vector. The columns correspond with the individual random
        variables while the rows are individual instances of the random vector.
    exponent: float
        Exponent of the Euclidean distance, in the range :math:`(0, 2)`.
        Equivalently, it is twice the Hurst parameter of fractional Brownian
        motion.

    Returns
    -------
    Stats
        Squared distance covariance, squared distance correlation,
        squared distance variance of the first random vector and
        squared distance variance of the second random vector.

    See Also
    --------
    u_distance_covariance_sqr
    u_distance_correlation_sqr

    Notes
    -----
    It is less efficient to compute the statistics separately, rather than
    using this function, because some computations can be shared.

    The algorithm uses the fast distance covariance algorithm proposed in
    :cite:`b-fast_distance_correlation` when possible.

    Examples
    --------
    >>> import numpy as np
    >>> import dcor
    >>> a = np.array([[1, 2, 3, 4],
    ...               [5, 6, 7, 8],
    ...               [9, 10, 11, 12],
    ...               [13, 14, 15, 16]])
    >>> b = np.array([[1], [0], [0], [1]])
    >>> dcor.u_distance_stats_sqr(a, a) # doctest: +ELLIPSIS
    ...                     # doctest: +NORMALIZE_WHITESPACE
    Stats(covariance_xy=42.6666666..., correlation_xy=1.0,
    variance_x=42.6666666..., variance_y=42.6666666...)
    >>> dcor.u_distance_stats_sqr(a, b) # doctest: +ELLIPSIS
    ...                     # doctest: +NORMALIZE_WHITESPACE
    Stats(covariance_xy=-2.6666666..., correlation_xy=-0.5,
    variance_x=42.6666666..., variance_y=0.6666666...)
    >>> dcor.u_distance_stats_sqr(b, b) # doctest: +ELLIPSIS
    ...                     # doctest: +NORMALIZE_WHITESPACE
    Stats(covariance_xy=0.6666666..., correlation_xy=1.0,
    variance_x=0.6666666..., variance_y=0.6666666...)
    >>> dcor.u_distance_stats_sqr(a, b, exponent=0.5) # doctest: +ELLIPSIS
    ...                                   # doctest: +NORMALIZE_WHITESPACE
    Stats(covariance_xy=-0.2996598..., correlation_xy=-0.4050479...,
    variance_x=0.8209855..., variance_y=0.6666666...)

    """
    if _can_use_fast_algorithm(x, y, **kwargs):
        return _u_distance_stats_sqr_fast(x, y)
    else:
        return _distance_sqr_stats_naive_generic(
            x, y,
            matrix_centered=_u_distance_matrix,
            product=u_product,
            **kwargs)


def distance_stats(x, y, **kwargs):
    """
    distance_stats(x, y, *, exponent=1)

    Computes the usual (biased) estimators for the distance covariance
    and distance correlation between two random vectors, and the
    individual distance variances.

    Parameters
    ----------
    x: array_like
        First random vector. The columns correspond with the individual random
        variables while the rows are individual instances of the random vector.
    y: array_like
        Second random vector. The columns correspond with the individual random
        variables while the rows are individual instances of the random vector.
    exponent: float
        Exponent of the Euclidean distance, in the range :math:`(0, 2)`.
        Equivalently, it is twice the Hurst parameter of fractional Brownian
        motion.

    Returns
    -------
    Stats
        Distance covariance, distance correlation,
        distance variance of the first random vector and
        distance variance of the second random vector.

    See Also
    --------
    distance_covariance
    distance_correlation

    Notes
    -----
    It is less efficient to compute the statistics separately, rather than
    using this function, because some computations can be shared.

    The algorithm uses the fast distance covariance algorithm proposed in
    :cite:`b-fast_distance_correlation` when possible.

    Examples
    --------
    >>> import numpy as np
    >>> import dcor
    >>> a = np.array([[1, 2, 3, 4],
    ...               [5, 6, 7, 8],
    ...               [9, 10, 11, 12],
    ...               [13, 14, 15, 16]])
    >>> b = np.array([[1], [0], [0], [1]])
    >>> dcor.distance_stats(a, a) # doctest: +NORMALIZE_WHITESPACE
    Stats(covariance_xy=7.2111025..., correlation_xy=1.0,
    variance_x=7.2111025..., variance_y=7.2111025...)
    >>> dcor.distance_stats(a, b) # doctest: +NORMALIZE_WHITESPACE
    Stats(covariance_xy=1.0, correlation_xy=0.5266403...,
    variance_x=7.2111025..., variance_y=0.5)
    >>> dcor.distance_stats(b, b) # doctest: +NORMALIZE_WHITESPACE
    Stats(covariance_xy=0.5, correlation_xy=1.0, variance_x=0.5,
    variance_y=0.5)
    >>> dcor.distance_stats(a, b, exponent=0.5) # doctest: +ELLIPSIS
    ...                             # doctest: +NORMALIZE_WHITESPACE
    Stats(covariance_xy=0.6087614..., correlation_xy=0.6703214...,
    variance_x=1.6495217..., variance_y=0.5)

    """
    return Stats(*[_sqrt(s) for s in distance_stats_sqr(x, y, **kwargs)])


def distance_correlation_sqr(x, y, **kwargs):
    """
    distance_correlation_sqr(x, y, *, exponent=1)

    Computes the usual (biased) estimator for the squared distance correlation
    between two random vectors.

    Parameters
    ----------
    x: array_like
        First random vector. The columns correspond with the individual random
        variables while the rows are individual instances of the random vector.
    y: array_like
        Second random vector. The columns correspond with the individual random
        variables while the rows are individual instances of the random vector.
    exponent: float
        Exponent of the Euclidean distance, in the range :math:`(0, 2)`.
        Equivalently, it is twice the Hurst parameter of fractional Brownian
        motion.

    Returns
    -------
    numpy scalar
        Value of the biased estimator of the squared distance correlation.

    See Also
    --------
    distance_correlation
    u_distance_correlation_sqr

    Notes
    -----
    The algorithm uses the fast distance covariance algorithm proposed in
    :cite:`b-fast_distance_correlation` when possible.

    Examples
    --------
    >>> import numpy as np
    >>> import dcor
    >>> a = np.array([[1, 2, 3, 4],
    ...               [5, 6, 7, 8],
    ...               [9, 10, 11, 12],
    ...               [13, 14, 15, 16]])
    >>> b = np.array([[1], [0], [0], [1]])
    >>> dcor.distance_correlation_sqr(a, a)
    1.0
    >>> dcor.distance_correlation_sqr(a, b) # doctest: +ELLIPSIS
    0.2773500...
    >>> dcor.distance_correlation_sqr(b, b)
    1.0
    >>> dcor.distance_correlation_sqr(a, b, exponent=0.5) # doctest: +ELLIPSIS
    0.4493308...

    """
    if _can_use_fast_algorithm(x, y, **kwargs):
        return _distance_correlation_sqr_fast(x, y)
    else:
        return _distance_correlation_sqr_naive(x, y, **kwargs)


def u_distance_correlation_sqr(x, y, **kwargs):
    """
    u_distance_correlation_sqr(x, y, *, exponent=1)

    Computes the bias-corrected estimator for the squared distance correlation
    between two random vectors.

    Parameters
    ----------
    x: array_like
        First random vector. The columns correspond with the individual random
        variables while the rows are individual instances of the random vector.
    y: array_like
        Second random vector. The columns correspond with the individual random
        variables while the rows are individual instances of the random vector.
    exponent: float
        Exponent of the Euclidean distance, in the range :math:`(0, 2)`.
        Equivalently, it is twice the Hurst parameter of fractional Brownian
        motion.

    Returns
    -------
    numpy scalar
        Value of the bias-corrected estimator of the squared distance
        correlation.

    See Also
    --------
    distance_correlation
    distance_correlation_sqr

    Notes
    -----
    The algorithm uses the fast distance covariance algorithm proposed in
    :cite:`b-fast_distance_correlation` when possible.

    Examples
    --------
    >>> import numpy as np
    >>> import dcor
    >>> a = np.array([[1, 2, 3, 4],
    ...               [5, 6, 7, 8],
    ...               [9, 10, 11, 12],
    ...               [13, 14, 15, 16]])
    >>> b = np.array([[1], [0], [0], [1]])
    >>> dcor.u_distance_correlation_sqr(a, a)
    1.0
    >>> dcor.u_distance_correlation_sqr(a, b)
    -0.5
    >>> dcor.u_distance_correlation_sqr(b, b)
    1.0
    >>> dcor.u_distance_correlation_sqr(a, b, exponent=0.5)
    ... # doctest: +ELLIPSIS
    -0.4050479...

    """
    if _can_use_fast_algorithm(x, y, **kwargs):
        return _u_distance_correlation_sqr_fast(x, y)
    else:
        return _u_distance_correlation_sqr_naive(x, y, **kwargs)


def distance_correlation(x, y, **kwargs):
    """
    distance_correlation(x, y, *, exponent=1)

    Computes the usual (biased) estimator for the distance correlation
    between two random vectors.

    Parameters
    ----------
    x: array_like
        First random vector. The columns correspond with the individual random
        variables while the rows are individual instances of the random vector.
    y: array_like
        Second random vector. The columns correspond with the individual random
        variables while the rows are individual instances of the random vector.
    exponent: float
        Exponent of the Euclidean distance, in the range :math:`(0, 2)`.
        Equivalently, it is twice the Hurst parameter of fractional Brownian
        motion.

    Returns
    -------
    numpy scalar
        Value of the biased estimator of the distance correlation.

    See Also
    --------
    distance_correlation_sqr
    u_distance_correlation_sqr

    Notes
    -----
    The algorithm uses the fast distance covariance algorithm proposed in
    :cite:`b-fast_distance_correlation` when possible.

    Examples
    --------
    >>> import numpy as np
    >>> import dcor
    >>> a = np.array([[1, 2, 3, 4],
    ...               [5, 6, 7, 8],
    ...               [9, 10, 11, 12],
    ...               [13, 14, 15, 16]])
    >>> b = np.array([[1], [0], [0], [1]])
    >>> dcor.distance_correlation(a, a)
    1.0
    >>> dcor.distance_correlation(a, b) # doctest: +ELLIPSIS
    0.5266403...
    >>> dcor.distance_correlation(b, b)
    1.0
    >>> dcor.distance_correlation(a, b, exponent=0.5) # doctest: +ELLIPSIS
    0.6703214...

    """
    return distance_stats(x, y, **kwargs).correlation_xy


def distance_correlation_af_inv_sqr(x, y):
    """
    Square of the affinely invariant distance correlation.

    Computes the estimator for the square of the affinely invariant distance
    correlation between two random vectors.

    .. warning:: The return value of this function is undefined when the
                 covariance matrix of :math:`x` or :math:`y` is singular.

    Parameters
    ----------
    x: array_like
        First random vector. The columns correspond with the individual random
        variables while the rows are individual instances of the random vector.
    y: array_like
        Second random vector. The columns correspond with the individual random
        variables while the rows are individual instances of the random vector.

    Returns
    -------
    numpy scalar
        Value of the estimator of the squared affinely invariant
        distance correlation.

    See Also
    --------
    distance_correlation
    u_distance_correlation

    Examples
    --------
    >>> import numpy as np
    >>> import dcor
    >>> a = np.array([[1, 3, 2, 5],
    ...               [5, 7, 6, 8],
    ...               [9, 10, 11, 12],
    ...               [13, 15, 15, 16]])
    >>> b = np.array([[1], [0], [0], [1]])
    >>> dcor.distance_correlation_af_inv_sqr(a, a)
    1.0
    >>> dcor.distance_correlation_af_inv_sqr(a, b) # doctest: +ELLIPSIS
    0.5773502...
    >>> dcor.distance_correlation_af_inv_sqr(b, b)
    1.0

    """
    x = _af_inv_scaled(x)
    y = _af_inv_scaled(y)

    correlation = distance_correlation_sqr(x, y)
    return 0 if np.isnan(correlation) else correlation


def distance_correlation_af_inv(x, y):
    """
    Affinely invariant distance correlation.

    Computes the estimator for the affinely invariant distance
    correlation between two random vectors.

    .. warning:: The return value of this function is undefined when the
                 covariance matrix of :math:`x` or :math:`y` is singular.

    Parameters
    ----------
    x: array_like
        First random vector. The columns correspond with the individual random
        variables while the rows are individual instances of the random vector.
    y: array_like
        Second random vector. The columns correspond with the individual random
        variables while the rows are individual instances of the random vector.

    Returns
    -------
    numpy scalar
        Value of the estimator of the squared affinely invariant
        distance correlation.

    See Also
    --------
    distance_correlation
    u_distance_correlation

    Examples
    --------
    >>> import numpy as np
    >>> import dcor
    >>> a = np.array([[1, 3, 2, 5],
    ...               [5, 7, 6, 8],
    ...               [9, 10, 11, 12],
    ...               [13, 15, 15, 16]])
    >>> b = np.array([[1], [0], [0], [1]])
    >>> dcor.distance_correlation_af_inv(a, a)
    1.0
    >>> dcor.distance_correlation_af_inv(a, b) # doctest: +ELLIPSIS
    0.7598356...
    >>> dcor.distance_correlation_af_inv(b, b)
    1.0

    """
    return _sqrt(distance_correlation_af_inv_sqr(x, y))

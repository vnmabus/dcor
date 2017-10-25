'''
    This module contains functions to compute statistics related to the
    distance covariance and distance correlation
    :cite:`b-distance_correlation`.

    References
    ----------
    .. bibliography:: refs.bib
       :labelprefix: B
       :keyprefix: b-
'''

from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import collections
import math
import warnings

import numba

import numpy as np

from . import distances


Stats = collections.namedtuple('Stats', ['covariance_xy', 'correlation_xy',
                               'variance_x', 'variance_y'])


HypothesisTest = collections.namedtuple('HypothesisTest', ['p_value',
                                        'statistic'])


def _check_kwargs_empty(kwargs):
    '''
    Raise an apropiate exception if the kwargs dictionary is not empty.
    '''

    if kwargs:
        raise TypeError("Unexpected keyword argument '{arg}'".format(
            arg=list(kwargs.keys())[0]))


def _check_valid_energy_exponent(exponent):
    if not 0 < exponent < 2:
        warning_msg = ('The energy distance is not guaranteed to be '
                       'a valid metric if the exponent value is '
                       'not in the range (0, 2). The exponent passed '
                       'is {exponent}.'.format(exponent=exponent))

        warnings.warn(warning_msg)


def _check_valid_dcov_exponent(exponent):
    if not 0 < exponent < 2:
        warning_msg = ('Distance covariance is not guaranteed to '
                       'characterize independence if the exponent value is '
                       'not in the range (0, 2). The exponent passed '
                       'is {exponent}.'.format(exponent=exponent))

        warnings.warn(warning_msg)


def double_centered(a):
    '''
    Returns a copy of the matrix :math:`a` in which both the sum of its
    columns and the sum of its rows are 0.

    In order to do that, for every element its row and column averages are
    subtracted, and the total average is added.

    Thus, if the element in the i-th row and j-th column of the original
    matrix :math:`a` is :math:`a_{i,j}`, then the new element will be

    .. math::

        \\tilde{a}_{i, j} = a_{i,j} - \\frac{1}{N} \\sum_{l=1}^N a_{il} -
        \\frac{1}{N}\\sum_{k=1}^N a_{kj} + \\frac{1}{N^2}\\sum_{k=1}^N a_{kj}.

    Parameters
    ----------
    a : (N, N) array_like
        Original matrix.

    Returns
    -------
    (N, N) ndarray
        Double centered matrix.

    See Also
    --------
    u_centered

    Examples
    --------
    >>> import numpy as np
    >>> import dcor
    >>> a = np.array([[1, 2], [3, 4]])
    >>> dcor.double_centered(a)
    array([[ 0.,  0.],
           [ 0.,  0.]])
    >>> b = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
    >>> dcor.double_centered(b)
    array([[ 0.44444444, -0.22222222, -0.22222222],
           [-0.22222222,  0.11111111,  0.11111111],
           [-0.22222222,  0.11111111,  0.11111111]])

    '''

    dim = np.size(a, 0)

    mu = np.sum(a) / (dim * dim)
    sum_cols = np.sum(a, 0, keepdims=True)
    sum_rows = np.sum(a, 1, keepdims=True)
    mu_cols = np.ones((dim, 1)).dot(sum_cols / dim)
    mu_rows = (sum_rows / dim).dot(np.ones((1, dim)))

    return a - mu_rows - mu_cols + mu


def u_centered(a):
    '''
    Returns a copy of the matrix :math:`a` which is :math:`U`-centered.

    If the element of the i-th row and j-th column of the original
    matrix :math:`a` is :math:`a_{i,j}`, then the new element will be

    .. math::

        \\tilde{a}_{i, j} =
        \\begin{cases}
        a_{i,j} - \\frac{1}{n-2} \\sum_{l=1}^n a_{il} -
        \\frac{1}{n-2} \\sum_{k=1}^n a_{kj} +
        \\frac{1}{(n-1)(n-2)}\\sum_{k=1}^n a_{kj},
        &\\text{if } i \\neq j, \\\\
        0,
        &\\text{if } i = j.
        \\end{cases}

    Parameters
    ----------
    a : (N, N) array_like
        Original matrix.

    Returns
    -------
    (N, N) ndarray
        :math:`U`-centered matrix.

    See Also
    --------
    double_centered

    Examples
    --------
    >>> import numpy as np
    >>> import dcor
    >>> a = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
    >>> dcor.u_centered(a)
    array([[ 0. ,  0.5, -1.5],
           [ 0.5,  0. , -4.5],
           [-1.5, -4.5,  0. ]])

    Note that when the matrix is 1x1 or 2x2, the formula performs
    a division by 0

    >>> import warnings
    >>> b = np.array([[1, 2], [3, 4]])
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     dcor.u_centered(b)
    array([[  0.,  nan],
           [ nan,   0.]])

    '''

    dim = np.size(a, 0)

    u_mu = np.sum(a) / ((dim - 1) * (dim - 2))
    sum_cols = np.sum(a, 0, keepdims=True)
    sum_rows = np.sum(a, 1, keepdims=True)
    u_mu_cols = np.ones((dim, 1)).dot(sum_cols / (dim - 2))
    u_mu_rows = (sum_rows / (dim - 2)).dot(np.ones((1, dim)))

    centered_matrix = a - u_mu_rows - u_mu_cols + u_mu

    # The diagonal is zero
    centered_matrix[np.eye(dim, dtype=bool)] = 0

    return centered_matrix


def average_product(a, b):
    '''
    Computes the average of the elements for an element-wise product of two
    matrices. If the matrices are square it is

    .. math::
        \\frac{1}{n^2} \\sum_{i,j=1}^n a_{i, j} b_{i, j}.

    Parameters
    ----------
    a: array_like
        First input array to be multiplied.
    b: array_like
        Second input array to be multiplied.

    Returns
    -------
    numpy scalar
        Average of the product.

    See Also
    --------
    u_product

    Examples
    --------
    >>> import numpy as np
    >>> import dcor
    >>> a = np.array([[1, 2, 4], [1, 2, 4], [1, 2, 4]])
    >>> b = np.array([[1, .5, .25], [1, .5, .25], [1, .5, .25]])
    >>> dcor.average_product(a, b)
    1.0
    >>> dcor.average_product(a, a)
    7.0

    If the matrices involved are not square, but have the same dimensions,
    the average of the product is still well defined

    >>> c = np.array([[1, 2], [1, 2], [1, 2]])
    >>> dcor.average_product(c, c)
    2.5
    '''

    return np.mean(a * b)


def u_product(a, b):
    '''
    Computes the inner product in the Hilbert space of :math:`U`-centered
    distance matrices

    .. math::
        \\frac{1}{n(n-3)} \\sum_{i,j=1}^n a_{i, j} b_{i, j}

    Parameters
    ----------
    a: array_like
        First input array to be multiplied.
    b: array_like
        Second input array to be multiplied.

    Returns
    -------
    numpy scalar
        Inner product.

    See Also
    --------
    average_product

    Examples
    --------
    >>> import numpy as np
    >>> import dcor
    >>> a = np.array([[  0.,   3.,  11.,   6.],
    ...               [  3.,   0.,   8.,   3.],
    ...               [ 11.,   8.,   0.,   5.],
    ...               [  6.,   3.,   5.,   0.]])
    >>> b = np.array([[  0.,  13.,  11.,   3.],
    ...               [ 13.,   0.,   2.,  10.],
    ...               [ 11.,   2.,   0.,   8.],
    ...               [  3.,  10.,   8.,   0.]])
    >>> u_a = dcor.u_centered(a)
    >>> u_a
    array([[ 0., -2.,  1.,  1.],
           [-2.,  0.,  1.,  1.],
           [ 1.,  1.,  0., -2.],
           [ 1.,  1., -2.,  0.]])
    >>> u_b = dcor.u_centered(b)
    >>> u_b
    array([[ 0.        ,  2.66666667,  2.66666667, -5.33333333],
           [ 2.66666667,  0.        , -5.33333333,  2.66666667],
           [ 2.66666667, -5.33333333,  0.        ,  2.66666667],
           [-5.33333333,  2.66666667,  2.66666667,  0.        ]])
    >>> dcor.u_product(u_a, u_a)
    6.0
    >>> dcor.u_product(u_a, u_b)
    -8.0

    Note that the formula is well defined as long as the matrices involved
    are square and have the same dimensions, even if they are not in the
    Hilbert space of :math:`U`-centered distance matrices

    >>> dcor.u_product(a, a)
    132.0

    Also the formula produces a division by 0 for 3x3 matrices

    >>> import warnings
    >>> b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     dcor.u_product(b, b)
    inf

    '''

    n = np.size(a, 0)

    return np.sum(a * b) / (n * (n - 3))


def u_projection(a):
    '''
    Returns a function computing the orthogonal projection over
    :math:`a` in the Hilbert space of :math:`U`-centered distance
    matrices.

    The projection of a matrix :math:`B` over a matrix :math:`A`
    is defined as

    .. math::
        \\text{proj}_A(B) = \\begin{cases}
        \\frac{\\langle A, B \\rangle}{\\langle A, A \\rangle} A,
        & \\text{if} \\langle A, A \\rangle \\neq 0, \\\\
        0, & \\text{if} \\langle A, A \\rangle = 0.
        \\end{cases}

    where :math:`\\langle {}\cdot{}, {}\cdot{} \\rangle` is the scalar
    product in the Hilbert space of :math:`U`-centered distance
    matrices, given by the function :py:func:`u_product`.

    Parameters
    ----------
    a: array_like
        :math:`U`-centered distance matrix.

    Returns
    -------
    callable
        Function that receives a :math:`U`-centered distance matrix and
        computes its orthogonal projection over :math:`a`.

    See Also
    --------
    u_complementary_projection
    u_centered

    Examples
    --------
    >>> import numpy as np
    >>> import dcor
    >>> a = np.array([[  0.,   3.,  11.,   6.],
    ...               [  3.,   0.,   8.,   3.],
    ...               [ 11.,   8.,   0.,   5.],
    ...               [  6.,   3.,   5.,   0.]])
    >>> b = np.array([[  0.,  13.,  11.,   3.],
    ...               [ 13.,   0.,   2.,  10.],
    ...               [ 11.,   2.,   0.,   8.],
    ...               [  3.,  10.,   8.,   0.]])
    >>> u_a = dcor.u_centered(a)
    >>> u_a
    array([[ 0., -2.,  1.,  1.],
           [-2.,  0.,  1.,  1.],
           [ 1.,  1.,  0., -2.],
           [ 1.,  1., -2.,  0.]])
    >>> u_b = dcor.u_centered(b)
    >>> u_b
    array([[ 0.        ,  2.66666667,  2.66666667, -5.33333333],
           [ 2.66666667,  0.        , -5.33333333,  2.66666667],
           [ 2.66666667, -5.33333333,  0.        ,  2.66666667],
           [-5.33333333,  2.66666667,  2.66666667,  0.        ]])
    >>> proj_a = dcor.u_projection(u_a)
    >>> proj_a(u_a)
    array([[ 0., -2.,  1.,  1.],
           [-2.,  0.,  1.,  1.],
           [ 1.,  1.,  0., -2.],
           [ 1.,  1., -2.,  0.]])
    >>> proj_a(u_b)
    array([[-0.        ,  2.66666667, -1.33333333, -1.33333333],
           [ 2.66666667, -0.        , -1.33333333, -1.33333333],
           [-1.33333333, -1.33333333, -0.        ,  2.66666667],
           [-1.33333333, -1.33333333,  2.66666667, -0.        ]])

    The function gives the correct result if
    :math:`\\langle A, A \\rangle = 0`.

    >>> proj_null = dcor.u_projection(np.zeros((4, 4)))
    >>> proj_null(u_a)
    array([[ 0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.]])
    '''

    c = a
    denominator = u_product(c, c)

    if denominator == 0:
        def projection(a):
            return np.zeros_like(c)
    else:
        def projection(a):
            return u_product(a, c) / denominator * c

    return projection


def u_complementary_projection(a):
    '''
    Returns a function computing the orthogonal projection over
    :math:`a^{\perp}` (the complementary projection over a)
    in the Hilbert space of :math:`U`-centered distance matrices.

    The projection of a matrix :math:`B` over a matrix :math:`A^{\perp}`
    is defined as

    .. math::
        \\text{proj}_{A^{\perp}}(B) = B - \\text{proj}_A(B)

    Parameters
    ----------
    a: array_like
        :math:`U`-centered distance matrix.

    Returns
    -------
    callable
        Function that receives a :math:`U`-centered distance matrices
        and computes its orthogonal projection over :math:`a^{\perp}`.

    See Also
    --------
    u_projection
    u_centered

    Examples
    --------
    >>> import numpy as np
    >>> import dcor
    >>> a = np.array([[  0.,   3.,  11.,   6.],
    ...               [  3.,   0.,   8.,   3.],
    ...               [ 11.,   8.,   0.,   5.],
    ...               [  6.,   3.,   5.,   0.]])
    >>> b = np.array([[  0.,  13.,  11.,   3.],
    ...               [ 13.,   0.,   2.,  10.],
    ...               [ 11.,   2.,   0.,   8.],
    ...               [  3.,  10.,   8.,   0.]])
    >>> u_a = dcor.u_centered(a)
    >>> u_a
    array([[ 0., -2.,  1.,  1.],
           [-2.,  0.,  1.,  1.],
           [ 1.,  1.,  0., -2.],
           [ 1.,  1., -2.,  0.]])
    >>> u_b = dcor.u_centered(b)
    >>> u_b
    array([[ 0.        ,  2.66666667,  2.66666667, -5.33333333],
           [ 2.66666667,  0.        , -5.33333333,  2.66666667],
           [ 2.66666667, -5.33333333,  0.        ,  2.66666667],
           [-5.33333333,  2.66666667,  2.66666667,  0.        ]])
    >>> proj_a = dcor.u_complementary_projection(u_a)
    >>> proj_a(u_a)
    array([[ 0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.]])
    >>> proj_a(u_b)
    array([[  0.00000000e+00,  -4.44089210e-16,   4.00000000e+00,
             -4.00000000e+00],
           [ -4.44089210e-16,   0.00000000e+00,  -4.00000000e+00,
              4.00000000e+00],
           [  4.00000000e+00,  -4.00000000e+00,   0.00000000e+00,
             -4.44089210e-16],
           [ -4.00000000e+00,   4.00000000e+00,  -4.44089210e-16,
              0.00000000e+00]])
    >>> proj_null = dcor.u_complementary_projection(np.zeros((4, 4)))
    >>> proj_null(u_a)
    array([[ 0., -2.,  1.,  1.],
           [-2.,  0.,  1.,  1.],
           [ 1.,  1.,  0., -2.],
           [ 1.,  1., -2.,  0.]])
    '''

    proj = u_projection(a)

    def projection(a):
        return a - proj(a)

    return projection


def _transform_to_2d(t):
    '''
    Convert vectors to column matrices, so that every ndarray has a 2d shape.
    '''
    t = np.asarray(t)

    dim = len(t.shape)
    assert dim <= 2

    if dim < 2:
        t = np.atleast_2d(t).T

    return t


def _distance_matrix_generic(x, centering, exponent=1):
    '''
    Computes a centered distance matrix given a matrix.
    '''

    _check_valid_dcov_exponent(exponent)

    x = _transform_to_2d(np.asfarray(x))

    # Calculate distance matrices
    a = distances._pdist(x, exponent=exponent)

    # Double centering
    a = centering(a)

    return a


def _distance_matrix(x, exponent=1):
    '''
    Computes the double centered distance matrix given a matrix.
    '''

    return _distance_matrix_generic(x, centering=double_centered,
                                    exponent=exponent)


def _u_distance_matrix(x, exponent=1):
    '''
    Computes the :math:`U`-centered distance matrices given two matrices.
    '''

    return _distance_matrix_generic(x, centering=u_centered,
                                    exponent=exponent)


def _u_distance_covariance_sqr_naive(x, y, exponent=1):
    '''
    Computes the unbiased estimator for distance covariance between two
    matrices, using an :math:`O(N^2)` algorithm.
    '''

    a = _u_distance_matrix(x, exponent=exponent)
    b = _u_distance_matrix(y, exponent=exponent)

    return u_product(a, b)


def distance_covariance_sqr(x, y, **kwargs):
    '''
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

    '''

    a = _distance_matrix(x, **kwargs)
    b = _distance_matrix(y, **kwargs)

    return average_product(a, b)


def distance_covariance(x, y, **kwargs):
    '''
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

    Examples
    --------
    >>> import numpy as np
    >>> import dcor
    >>> a = np.array([[1, 2, 3, 4],
    ...               [5, 6, 7, 8],
    ...               [9, 10, 11, 12],
    ...               [13, 14, 15, 16]])
    >>> b = np.array([[1], [0], [0], [1]])
    >>> dcor.distance_covariance(a, a)
    7.2111025509279782
    >>> dcor.distance_covariance(a, b)
    1.0
    >>> dcor.distance_covariance(b, b)
    0.5
    >>> dcor.distance_covariance(a, b, exponent=0.5) # doctest: +ELLIPSIS
    0.6087614...
    '''
    return np.sqrt(distance_covariance_sqr(x, y, **kwargs))


def _distance_sqr_stats_naive_generic(x, y, matrix_centered, product,
                                      exponent=1):
    '''
    Compute generic squared stats.
    '''
    a = matrix_centered(x, exponent=exponent)
    b = matrix_centered(y, exponent=exponent)

    covariance_xy_sqr = product(a, b)
    variance_x_sqr = product(a, a)
    variance_y_sqr = product(b, b)

    denominator_sqr = variance_x_sqr * variance_y_sqr
    denominator = math.sqrt(denominator_sqr)

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


def distance_stats_sqr(x, y, **kwargs):
    '''
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

    Examples
    --------
    >>> import numpy as np
    >>> import dcor
    >>> a = np.array([[1, 2, 3, 4],
    ...               [5, 6, 7, 8],
    ...               [9, 10, 11, 12],
    ...               [13, 14, 15, 16]])
    >>> b = np.array([[1], [0], [0], [1]])
    >>> dcor.distance_stats_sqr(a, a)
    Stats(covariance_xy=52.0, correlation_xy=1.0, variance_x=52.0, \
variance_y=52.0)
    >>> dcor.distance_stats_sqr(a, b)
    Stats(covariance_xy=1.0, correlation_xy=0.27735009811261457, \
variance_x=52.0, variance_y=0.25)
    >>> dcor.distance_stats_sqr(b, b)
    Stats(covariance_xy=0.25, correlation_xy=1.0, variance_x=0.25, \
variance_y=0.25)
    >>> dcor.distance_stats_sqr(a, b, exponent=0.5) # doctest: +ELLIPSIS
    Stats(covariance_xy=0.3705904..., correlation_xy=0.4493308..., \
variance_x=2.7209220..., variance_y=0.25)

    '''

    return _distance_sqr_stats_naive_generic(
        x, y,
        matrix_centered=_distance_matrix,
        product=average_product, **kwargs)


def distance_stats(x, y, **kwargs):
    '''
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

    Examples
    --------
    >>> import numpy as np
    >>> import dcor
    >>> a = np.array([[1, 2, 3, 4],
    ...               [5, 6, 7, 8],
    ...               [9, 10, 11, 12],
    ...               [13, 14, 15, 16]])
    >>> b = np.array([[1], [0], [0], [1]])
    >>> dcor.distance_stats(a, a)
    Stats(covariance_xy=7.2111025509279782, correlation_xy=1.0, \
variance_x=7.2111025509279782, variance_y=7.2111025509279782)
    >>> dcor.distance_stats(a, b)
    Stats(covariance_xy=1.0, correlation_xy=0.52664038784792666, \
variance_x=7.2111025509279782, variance_y=0.5)
    >>> dcor.distance_stats(b, b)
    Stats(covariance_xy=0.5, correlation_xy=1.0, variance_x=0.5, \
variance_y=0.5)
    >>> dcor.distance_stats(a, b, exponent=0.5) # doctest: +ELLIPSIS
    Stats(covariance_xy=0.6087614..., correlation_xy=0.6703214..., \
variance_x=1.6495217..., variance_y=0.5)
    '''

    return Stats(*[np.sqrt(s) for s in distance_stats_sqr(x, y, **kwargs)])


def distance_correlation_sqr(x, y, **kwargs):
    '''
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
    '''

    return distance_stats_sqr(x, y, **kwargs).correlation_xy


def distance_correlation(x, y, **kwargs):
    '''
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
    '''

    return distance_stats(x, y, **kwargs).correlation_xy


def _u_distance_correlation_sqr_naive(x, y, exponent=1):
    '''
    Computes distance correlation estimator between two matrices
    using the U-statistic.
    '''

    return _distance_sqr_stats_naive_generic(
        x, y,
        matrix_centered=_u_distance_matrix,
        product=u_product,
        exponent=exponent).correlation_xy


def _is_random_variable(x):
    '''
    Check if the matrix x correspond to a random variable (it is a vector,
    or a matrix corresponding to a column vector).
    '''
    return len(x.shape) == 1 or x.shape[1] == 1


def _can_use_u_fast_algorithm(x, y, exponent=1):
    '''
    Returns a boolean indicating if the fast :math:`O(NlogN)` algorithm for
    the unbiased distance stats can be used.

    The algorithm can only be used for random variables (not vectors) where
    the number of instances is greater than 3. Also, the exponent must be 1.
    '''

    return (_is_random_variable(x) and _is_random_variable(y) and
            x.shape[0] > 3 and y.shape[0] > 3 and exponent == 1)


@numba.jit
def _dyad_update(y, c):

    n = y.shape[0]
    gamma = np.zeros(n)

    # Step 1: get the smallest L such that n <= 2^L
    L = int(math.ceil(np.log2(n)))

    # Step 2: assign s(l, k) = 0
    S_len = 2 ** (L + 1)
    S = np.zeros(S_len)

    pos_sums = np.arange(L)
    pos_sums[:] = 2 ** (L - pos_sums)
    pos_sums = np.cumsum(pos_sums)

    # Step 3: iteration
    for i in range(1, n):

        # Step 3.a: update s(l, k)
        for l in range(L):
            k = int(math.ceil(y[i - 1] / 2 ** l))
            pos = k - 1

            if l > 0:
                pos += pos_sums[l - 1]

            S[pos] += c[i - 1]

        # Steps 3.b and 3.c
        for l in range(L):
            k = int(math.floor((y[i] - 1) / 2 ** l))
            if k / 2 > math.floor(k / 2):
                pos = k - 1
                if l > 0:
                    pos += pos_sums[l - 1]

                gamma[i] = gamma[i] + S[pos]

    return gamma


def _partial_sum_2d(x, y, c):
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


def _u_distance_covariance_sqr_fast(x, y):
    '''
    Fast algorithm for the distance covariance.
    '''
    x = np.asfarray(x)
    y = np.asfarray(y)

    x = np.ravel(x)
    y = np.ravel(y)

    n = x.shape[0]
    if n <= 3:
        raise ValueError(
            "Expected dimension of the matrix > 3 and found {dim}".format(
                dim=n))
    assert(n == y.shape[0])
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
    gamma_1 = _partial_sum_2d(x, y, np.ones(n))
    gamma_x = _partial_sum_2d(x, y, x)
    gamma_y = _partial_sum_2d(x, y, y)
    gamma_xy = _partial_sum_2d(x, y, x * y)

    # Step 8
    aijbij = np.sum(x * y * gamma_1 + gamma_xy - x * gamma_y - y * gamma_x)

    # Step 9
    d_cov = (aijbij / n / (n - 3) - 2 * sum_ab / n / (n - 2) / (n - 3) +
             a_dot_dot * b_dot_dot / n / (n - 1) / (n - 2) / (n - 3))

    return d_cov


def _u_distance_stats_sqr_fast(x, y):
    '''
    Compute the bias-corrected distance stats using the fast
    :math:`O(NlogN)` algorithm.
    '''
    covariance_xy_sqr = _u_distance_covariance_sqr_fast(x, y)
    variance_x_sqr = _u_distance_covariance_sqr_fast(x, x)
    variance_y_sqr = _u_distance_covariance_sqr_fast(y, y)
    denominator_sqr_signed = variance_x_sqr * variance_y_sqr
    denominator_sqr = np.fabs(denominator_sqr_signed)
    denominator = math.sqrt(denominator_sqr)

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


def _u_distance_correlation_sqr_fast(x, y):
    '''
    Fast algorithm for distance correlation.
    '''
    return _u_distance_stats_sqr_fast(x, y).correlation_xy


def u_distance_stats_sqr(x, y, **kwargs):
    '''
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
    Stats(covariance_xy=42.6666666..., correlation_xy=1.0, \
variance_x=42.6666666..., variance_y=42.6666666...)
    >>> dcor.u_distance_stats_sqr(a, b) # doctest: +ELLIPSIS
    Stats(covariance_xy=-2.6666666..., correlation_xy=-0.5, \
variance_x=42.6666666..., variance_y=0.6666666...)
    >>> dcor.u_distance_stats_sqr(b, b) # doctest: +ELLIPSIS
    Stats(covariance_xy=0.6666666..., correlation_xy=1.0, \
variance_x=0.6666666..., variance_y=0.6666666...)
    >>> dcor.u_distance_stats_sqr(a, b, exponent=0.5) # doctest: +ELLIPSIS
    Stats(covariance_xy=-0.2996598..., correlation_xy=-0.4050479..., \
variance_x=0.8209855..., variance_y=0.6666666...)

    '''
    if _can_use_u_fast_algorithm(x, y, **kwargs):
        return _u_distance_stats_sqr_fast(x, y)
    else:
        return _distance_sqr_stats_naive_generic(
            x, y,
            matrix_centered=_u_distance_matrix,
            product=u_product,
            **kwargs)


def u_distance_covariance_sqr(x, y, **kwargs):
    '''
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
    '''

    if _can_use_u_fast_algorithm(x, y, **kwargs):
        return _u_distance_covariance_sqr_fast(x, y)
    else:
        return _u_distance_covariance_sqr_naive(x, y, **kwargs)


def u_distance_correlation_sqr(x, y, **kwargs):
    '''
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
    '''

    if _can_use_u_fast_algorithm(x, y, **kwargs):
        return _u_distance_correlation_sqr_fast(x, y)
    else:
        return _u_distance_correlation_sqr_naive(x, y, **kwargs)


def partial_distance_covariance(x, y, z):
    '''
    Computes the estimator for the partial distance covariance of the
    random vectors corresponding to :math:`x` and :math:`y` with respect
    to the random variable corresponding to :math:`z`.

    Parameters
    ----------
    x: array_like
        First random vector. The columns correspond with the individual random
        variables while the rows are individual instances of the random vector.
    y: array_like
        Second random vector. The columns correspond with the individual random
        variables while the rows are individual instances of the random vector.
    z: array_like
        Random vector with respect to which the partial distance covariance
        is computed. The columns correspond with the individual random
        variables while the rows are individual instances of the random vector.

    Returns
    -------
    numpy scalar
        Value of the estimator of the partial distance covariance.

    See Also
    --------
    partial_distance_correlation

    Examples
    --------
    >>> import numpy as np
    >>> import dcor
    >>> a = np.array([[1, 2, 3, 4],
    ...               [5, 6, 7, 8],
    ...               [9, 10, 11, 12],
    ...               [13, 14, 15, 16]])
    >>> b = np.array([[1], [0], [0], [1]])
    >>> c = np.array([[1, 3, 4],
    ...               [5, 7, 8],
    ...               [9, 11, 15],
    ...               [13, 15, 16]])
    >>> dcor.partial_distance_covariance(a, a, c) # doctest: +ELLIPSIS
    0.0024298...
    >>> dcor.partial_distance_covariance(a, b, c)
    0.0347030...
    >>> dcor.partial_distance_covariance(b, b, c)
    0.4956241...
    '''

    a = _u_distance_matrix(x)
    b = _u_distance_matrix(y)
    c = _u_distance_matrix(z)

    proj = u_complementary_projection(c)

    return u_product(proj(a), proj(b))


def partial_distance_correlation(x, y, z):
    '''
    Computes the estimator for the partial distance correlation of the
    random vectors corresponding to :math:`x` and :math:`y` with respect
    to the random variable corresponding to :math:`z`.

    Parameters
    ----------
    x: array_like
        First random vector. The columns correspond with the individual random
        variables while the rows are individual instances of the random vector.
    y: array_like
        Second random vector. The columns correspond with the individual random
        variables while the rows are individual instances of the random vector.
    z: array_like
        Random vector with respect to which the partial distance correlation
        is computed. The columns correspond with the individual random
        variables while the rows are individual instances of the random vector.

    Returns
    -------
    numpy scalar
        Value of the estimator of the partial distance correlation.

    See Also
    --------
    partial_distance_covariance

    Examples
    --------
    >>> import numpy as np
    >>> import dcor
    >>> a = np.array([[1], [1], [2], [2], [3]])
    >>> b = np.array([[1], [2], [1], [2], [1]])
    >>> c = np.array([[1], [2], [2], [1], [2]])
    >>> dcor.partial_distance_correlation(a, a, c)
    1.0
    >>> dcor.partial_distance_correlation(a, b, c)
    -0.5
    >>> dcor.partial_distance_correlation(b, b, c)
    1.0
    >>> dcor.partial_distance_correlation(a, c, c)
    0.0
    '''

    a = _u_distance_matrix(x)
    b = _u_distance_matrix(y)
    c = _u_distance_matrix(z)

    proj = u_complementary_projection(c)

    a_proj = proj(a)
    b_proj = proj(b)

    denom_sqr = u_product(a_proj, a_proj) * u_product(b_proj, b_proj)

    if denom_sqr == 0:
        correlation = denom_sqr.dtype.type(0)
    else:
        correlation = u_product(a_proj, b_proj) / np.sqrt(denom_sqr)

    return correlation


def _energy_distance_from_distance_matrices(
        distance_xx, distance_yy, distance_xy):
    '''
    Computed energy distance with precalculated distance matrices.
    '''

    return (2 * np.mean(distance_xy) - np.mean(distance_xx) -
            np.mean(distance_yy))


def energy_distance(x, y, **kwargs):
    '''
    energy_distance(x, y, *, exponent=1)

    Computes the estimator for the energy distance of the
    random vectors corresponding to :math:`x` and :math:`y`.
    Both random vectors must have the same number of components.

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

    Returns
    -------
    numpy scalar
        Value of the estimator of the energy distance.

    Examples
    --------
    >>> import numpy as np
    >>> import dcor
    >>> a = np.array([[1, 2, 3, 4],
    ...               [5, 6, 7, 8],
    ...               [9, 10, 11, 12],
    ...               [13, 14, 15, 16]])
    >>> b = np.array([[1, 0, 0, 1],
    ...               [0, 1, 1, 1],
    ...               [1, 1, 1, 1]])
    >>> dcor.energy_distance(a, a)
    0.0
    >>> dcor.energy_distance(a, b) # doctest: +ELLIPSIS
    20.5780594...
    >>> dcor.energy_distance(b, b)
    0.0

    A different exponent for the Euclidean distance in the range
    :math:`(0, 2)` can be used:

    >>> dcor.energy_distance(a, a, exponent=1.5)
    0.0
    >>> dcor.energy_distance(a, b, exponent=1.5)
    ... # doctest: +ELLIPSIS
    99.7863955...
    >>> dcor.energy_distance(b, b, exponent=1.5)
    0.0

    '''

    x = _transform_to_2d(x)
    y = _transform_to_2d(y)

    exponent = kwargs.pop('exponent', 1)
    _check_kwargs_empty(kwargs)

    _check_valid_energy_exponent(exponent)

    distance_xx = distances._pdist(x, exponent=exponent)
    distance_yy = distances._pdist(y, exponent=exponent)
    distance_xy = distances._cdist(x, y, exponent=exponent)

    return _energy_distance_from_distance_matrices(distance_xx=distance_xx,
                                                   distance_yy=distance_yy,
                                                   distance_xy=distance_xy)

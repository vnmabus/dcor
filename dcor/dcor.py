from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import math

# import numba
import scipy.spatial

import numpy as np


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

    '''

    dim = np.size(a, 0)

    if dim <= 2:
        raise ValueError(
            "Expected dimension of the matrix > 2 and found {dim}".format(
                dim=dim))

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
    matrices

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
    (1, 1) ndarray
        Average of the product.

    See Also
    --------
    u_product
    '''

    return np.mean(a * b)


def u_product(a, b):
    '''
    Computes the inner product in the Hilbert space of U-centered distance
    matrices

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
    (1, 1) ndarray
        Inner product.

    See Also
    --------
    average_product
    '''

    n = np.size(a, 0)
    if n <= 3:
        raise ValueError(
            "Expected dimension of the matrix > 3 and found {dim}".format(
                dim=n))

    return np.sum(a * b) / (n * (n - 3))


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


def _distance_matrices_generic(X, Y, centering):
    '''
    Computes the double centered distance matrices given two matrices.
    '''

    X = _transform_to_2d(np.asfarray(X))
    Y = _transform_to_2d(np.asfarray(Y))

    n = X.shape[0]
    assert n == Y.shape[0]

    # Calculate distance matrices
    A = scipy.spatial.distance.cdist(X, X)
    B = scipy.spatial.distance.cdist(Y, Y)

    # Double centering
    A = centering(A)
    B = centering(B)

    return A, B


def _distance_matrices(X, Y):
    '''
    Computes the double centered distance matrices given two matrices.
    '''

    return _distance_matrices_generic(X, Y, centering=double_centered)


def _u_distance_matrices(X, Y):
    '''
    Computes the u-centered distance matrices given two matrices.
    '''

    return _distance_matrices_generic(X, Y, centering=u_centered)


def u_distance_covariance_sqr_naive(X, Y):
    '''
    Computes distance covariance between two matrices.
    '''

    A, B = _u_distance_matrices(X, Y)

    return u_product(A, B)


def distance_covariance_sqr(X, Y):
    '''
    Computes distance covariance between two matrices.
    '''

    A, B = _distance_matrices(X, Y)

    return average_product(A, B)


def _distance_correlation_naive_generic(X, Y, matrices, covariance):
    A, B = matrices(X, Y)

    prod_avg = covariance(A, B)
    if prod_avg == 0:
        return prod_avg
    else:
        return prod_avg / math.sqrt(covariance(A, A) *
                                    covariance(B, B))


def distance_correlation_sqr_naive(X, Y):
    '''
    Computes distance correlation between two matrices.
    '''

    return _distance_correlation_naive_generic(
        X, Y,
        matrices=_distance_matrices,
        covariance=average_product)


def u_distance_correlation_sqr_naive(X, Y):
    '''
    Computes distance correlation between two matrices using the U-statistic.
    '''

    return _distance_correlation_naive_generic(
        X, Y,
        matrices=_u_distance_matrices,
        covariance=u_product)


# @numba.jit
def dyad_update(Y, C):

    Y = np.asarray(Y)
    C = np.asarray(C)

    n = Y.shape[0]
    gamma = np.zeros(n)

    # Step 1: get the smallest L such that n <= 2^L
    L = int(math.ceil(math.log(n, 2)))

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
            k = int(math.ceil(Y[i - 1] / 2 ** l))
            pos = k - 1

            if l > 0:
                pos += pos_sums[l - 1]

            S[pos] += C[i - 1]

        # Steps 3.b and 3.c
        for l in range(L):
            k = int(math.floor((Y[i] - 1) / 2 ** l))
            if k / 2 > math.floor(k / 2):
                pos = k - 1
                if l > 0:
                    pos += pos_sums[l - 1]

                gamma[i] = gamma[i] + S[pos]

    return gamma


def partial_sum_2d(X, Y, C):
    X = np.asarray(X)
    Y = np.asarray(Y)
    C = np.asarray(C)

    n = X.shape[0]

    # Step 1: rearrange X, Y and C so X is in ascending order
    temp = range(n)

    ix0 = np.argsort(X)
    ix = np.zeros(n, dtype=int)
    ix[ix0] = temp

    X = X[ix0]
    Y = Y[ix0]
    C = C[ix0]

    # Step 2
    iy0 = np.argsort(Y)
    iy = np.zeros(n, dtype=int)
    iy[iy0] = temp

    Y = iy + 1

    # Step 3
    sy = np.cumsum(C[iy0]) - C[iy0]

    # Step 4
    sx = np.cumsum(C) - C

    # Step 5
    c_dot = np.sum(C)

    # Step 6
    gamma1 = dyad_update(Y, C)

    # Step 7
    gamma = c_dot - C - 2 * sy[iy] - 2 * sx + 4 * gamma1
    gamma = gamma[ix]

    return gamma


def u_distance_covariance_sqr_fast(X, Y):
    X = np.asfarray(X)
    Y = np.asfarray(Y)

    X = np.ravel(X)
    Y = np.ravel(Y)

    n = X.shape[0]
    if n <= 3:
        raise ValueError(
            "Expected dimension of the matrix > 3 and found {dim}".format(
                dim=n))
    assert(n == Y.shape[0])
    temp = range(n)

    # Step 1
    ix0 = np.argsort(X)
    vx = X[ix0]

    ix = np.zeros(n, dtype=int)
    ix[ix0] = temp

    iy0 = np.argsort(Y)
    vy = Y[iy0]

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
    x_dot = np.sum(X)
    y_dot = np.sum(Y)

    # Step 5
    a_i_dot = x_dot + (2 * alpha_x - n) * X - 2 * beta_x
    b_i_dot = y_dot + (2 * alpha_y - n) * Y - 2 * beta_y

    sum_ab = np.sum(a_i_dot * b_i_dot)

    # Step 6
    a_dot_dot = 2 * np.sum(alpha_x * X) - 2 * np.sum(beta_x)
    b_dot_dot = 2 * np.sum(alpha_y * Y) - 2 * np.sum(beta_y)

    # Step 7
    gamma_1 = partial_sum_2d(X, Y, np.ones(n))
    gamma_x = partial_sum_2d(X, Y, X)
    gamma_y = partial_sum_2d(X, Y, Y)
    gamma_xy = partial_sum_2d(X, Y, X * Y)

    # Step 8
    aijbij = np.sum(X * Y * gamma_1 + gamma_xy - X * gamma_y - Y * gamma_x)

    # Step 9
    d_cov = (aijbij / n / (n - 3) - 2 * sum_ab / n / (n - 2) / (n - 3) +
             a_dot_dot * b_dot_dot / n / (n - 1) / (n - 2) / (n - 3))

    return d_cov


def u_distance_correlation_sqr_fast(X, Y):
    dcov_XY = u_distance_covariance_sqr_fast(X, Y)
    dcov_X = u_distance_covariance_sqr_fast(X, X)
    dcov_Y = u_distance_covariance_sqr_fast(Y, Y)

    if math.fabs(dcov_X * dcov_Y) < 1e-10:
        return 0
    else:
        return dcov_XY / math.sqrt(math.fabs(dcov_X * dcov_Y))


distance_correlation_sqr = distance_correlation_sqr_naive
distance_correlation_sqr_multivariate = distance_correlation_sqr_naive

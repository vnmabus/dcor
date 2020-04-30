'''
Functions to compute fast distance covariance using AVL.
'''
import math

import numpy as np

from ._utils import _jit


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


def _distance_covariance_sqr_avl_generic(
        x, y, *, exponent=1, unbiased=False):  # pylint:disable=too-many-locals
    # This function has many locals so it can be compared
    # with the original algorithm.
    """Fast algorithm for the squared distance covariance."""

    if exponent != 1:
        raise ValueError(f"Exponent should be 1 but is {exponent} instead.")

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

    sum_ab = a_i_dot @ b_i_dot

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

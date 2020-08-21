'''
Functions to compute fast distance covariance using AVL.
'''
from dcor._utils import CompileMode
import math
import warnings

from numba import float64, int64, boolean
import numba
import numba.cuda as cuda

import numpy as np

from ._utils import CompileMode


NO_COMPILE = 0
COMPILE = 1
GPU = 2


def _dyad_update(y, c, gamma, l_max, s, pos_sums):  # pylint:disable=too-many-locals
    # This function has many locals so it can be compared
    # with the original algorithm.
    """
    Inner function of the fast distance covariance.

    This function is compiled because otherwise it would become
    a bottleneck.

    """
    n = y.shape[0]

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


_dyad_update_compiled = numba.njit(
    float64[:](float64[:], float64[:], float64[:],
               int64, float64[:], int64[:]),
    cache=True)(
        _dyad_update)

if cuda.is_available():
    _dyad_update_compiled_gpu = cuda.jit(
        device=True)(
            _dyad_update)


def _generate_partial_sum_2d(compiled):

    def _partial_sum_2d(x, y, c):  # pylint:disable=too-many-locals

        if compiled == NO_COMPILE:
            dyad_update = _dyad_update
        elif compiled == COMPILE:
            dyad_update = _dyad_update_compiled
        else:
            dyad_update = _dyad_update_compiled_gpu

        # This function has many locals so it can be compared
        # with the original algorithm.
        x = np.asarray(x)
        y = np.asarray(y)
        c = np.asarray(c)

        n = x.shape[0]

        # Step 1: rearrange x, y and c so x is in ascending order
        temp = np.arange(n)

        ix0 = np.argsort(x)
        ix = np.zeros(n, dtype=np.int_)
        ix[ix0] = temp

        x = x[ix0]
        y = y[ix0]
        c = c[ix0]

        # Step 2
        iy0 = np.argsort(y)
        iy = np.zeros(n, dtype=np.int_)
        iy[iy0] = temp

        y = iy + 1.

        # Step 3
        sy = np.cumsum(c[iy0]) - c[iy0]

        # Step 4
        sx = np.cumsum(c) - c

        # Step 5
        c_dot = np.sum(c)

        # Step 6
        y = np.asarray(y)
        c = np.asarray(c)
        gamma1 = np.zeros(n, dtype=c.dtype)

        # Step 1: get the smallest l such that n <= 2^l
        l_max = int(math.ceil(np.log2(n)))

        # Step 2: assign s(l, k) = 0
        s_len = 2 ** (l_max + 1)
        s = np.zeros(s_len, dtype=c.dtype)

        pos_sums = np.arange(l_max)
        pos_sums[:] = 2 ** (l_max - pos_sums)
        pos_sums = np.cumsum(pos_sums)

        gamma1 = dyad_update(y, c, gamma1, l_max, s, pos_sums)

        # Step 7
        gamma = c_dot - c - 2 * sy[iy] - 2 * sx + 4 * gamma1
        gamma = gamma[ix]

        return gamma

    return _partial_sum_2d


_partial_sum_2d = _generate_partial_sum_2d(compiled=NO_COMPILE)
_partial_sum_2d_compiled = numba.njit(
    float64[:](float64[:], float64[:], float64[:]),
    cache=True)(
    _generate_partial_sum_2d(compiled=COMPILE))

if cuda.is_available():
    _partial_sum_2d_compiled_gpu = cuda.jit(
        device=True)(
        _generate_partial_sum_2d(compiled=GPU))


def _generate_distance_covariance_sqr_avl_impl(compiled):

    def _distance_covariance_sqr_avl_impl(
            x, y, ix, iy, vx, vy, unbiased):  # pylint:disable=too-many-locals
        # This function has many locals so it can be compared
        # with the original algorithm.
        """Fast algorithm for the squared distance covariance."""

        if compiled == NO_COMPILE:
            partial_sum_2d = _partial_sum_2d
        elif compiled == COMPILE:
            partial_sum_2d = _partial_sum_2d_compiled
        else:
            partial_sum_2d = _partial_sum_2d_compiled_gpu

        n = x.shape[0]

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
        gamma_1 = partial_sum_2d(x, y, np.ones(n, dtype=x.dtype))
        gamma_x = partial_sum_2d(x, y, x)
        gamma_y = partial_sum_2d(x, y, y)
        gamma_xy = partial_sum_2d(x, y, x * y)

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

    return _distance_covariance_sqr_avl_impl


_distance_covariance_sqr_avl_impl = _generate_distance_covariance_sqr_avl_impl(
    compiled=NO_COMPILE)
_distance_covariance_sqr_avl_impl_compiled = numba.njit(
    float64(float64[:], float64[:],
            int64[:], int64[:],
            float64[:], float64[:], boolean),
    cache=True)(
    _generate_distance_covariance_sqr_avl_impl(compiled=COMPILE))

if cuda.is_available():
    _distance_covariance_sqr_avl_impl_compiled_gpu = cuda.jit(
        device=True)(
        _generate_distance_covariance_sqr_avl_impl(compiled=GPU))


impls_dict = {
    CompileMode.AUTO: (_distance_covariance_sqr_avl_impl_compiled,
                       _distance_covariance_sqr_avl_impl),
    CompileMode.NO_COMPILE: (_distance_covariance_sqr_avl_impl,),
    CompileMode.COMPILE_CPU: (_distance_covariance_sqr_avl_impl_compiled,)
}


def _distance_covariance_sqr_avl_generic(
        x, y, *, exponent=1, unbiased=False, compile_mode=CompileMode.AUTO):
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

    ix = np.zeros(n, dtype=np.int64)
    ix[ix0] = temp

    iy0 = np.argsort(y)
    vy = y[iy0]

    iy = np.zeros(n, dtype=np.int64)
    iy[iy0] = temp

    if compile_mode not in impls_dict:
        raise NotImplementedError(
            f"Compile mode {compile_mode} not implemented.")

    for impl in impls_dict[compile_mode]:

        try:

            return impl(x, y,
                        ix, iy,
                        vx, vy,
                        unbiased)

        except TypeError as e:

            if compile_mode is not CompileMode.AUTO:
                raise e

            warnings.warn(f"Falling back to uncompiled AVL fast distance "
                          f"covariance because of TypeError exception "
                          f"raised: {e}. Rembember: only floating point "
                          f"values can be used in the compiled "
                          f"implementations.")


def _rowwise_distance_covariance_sqr_avl_generic_internal(
        x, y, ix, iy, vx, vy, unbiased, res):

    res[0] = _distance_covariance_sqr_avl_impl_compiled(
        x, y, unbiased=unbiased,
        ix=ix, iy=iy,
        vx=vx, vy=vy)


if cuda.is_available():
    def _rowwise_distance_covariance_sqr_avl_generic_internal_gpu(
            x, y, ix, iy, vx, vy, unbiased, res):

        res[0] = 0  # _partial_sum_2d_compiled_gpu(x, y, vx)

#         res[0] = _distance_covariance_sqr_avl_impl_compiled_gpu(
#             x, y, unbiased=unbiased,
#             ix=ix, iy=iy,
#             vx=vx, vy=vy)


def _generate_rowwise_internal(target):
    cache = target != 'cuda'
    fun = (_rowwise_distance_covariance_sqr_avl_generic_internal_gpu
           if cuda.is_available() and target == 'cuda'
           else _rowwise_distance_covariance_sqr_avl_generic_internal)

    return numba.guvectorize(
        [(float64[:], float64[:],
          int64[:], int64[:],
          float64[:], float64[:],
          boolean, float64[:])],
        '(n),(n),(n),(n),(n),(n),()->()', nopython=True,
        cache=cache,
        target=target)(fun)


_rowwise_distance_covariance_sqr_avl_generic_internal_cpu = (
    _generate_rowwise_internal(target='cpu')
)
_rowwise_distance_covariance_sqr_avl_generic_internal_parallel = (
    _generate_rowwise_internal(target='parallel')
)

if cuda.is_available():
    _rowwise_distance_covariance_sqr_avl_generic_internal_gpu = (
        _generate_rowwise_internal(target='cuda')
    )

rowwise_impls_dict = {
    CompileMode.AUTO:
    _rowwise_distance_covariance_sqr_avl_generic_internal_parallel,
    CompileMode.COMPILE_CPU:
    _rowwise_distance_covariance_sqr_avl_generic_internal_cpu,
    CompileMode.COMPILE_PARALLEL:
    _rowwise_distance_covariance_sqr_avl_generic_internal_parallel,
}

if cuda.is_available():
    rowwise_impls_dict[CompileMode.COMPILE_GPU] = (
        _rowwise_distance_covariance_sqr_avl_generic_internal_gpu)
    rowwise_impls_dict[CompileMode.AUTO] = (
        _rowwise_distance_covariance_sqr_avl_generic_internal_gpu)


def _rowwise_distance_covariance_sqr_avl_generic(
        x, y, exponent=1, unbiased=False,
        compile_mode=CompileMode.AUTO):

    if exponent != 1:
        raise ValueError(f"Exponent should be 1 but is {exponent} instead.")

    x = np.asarray(x)
    y = np.asarray(y)

    res = np.zeros(x.shape[0], dtype=x.dtype)

    n = x.shape[1]
    assert n > 3
    assert n == y.shape[1]
    temp = range(n)

    # Step 1
    ix0 = np.argsort(x, axis=1)
    vx = np.take_along_axis(x, ix0, axis=1)

    ix = np.zeros_like(x, dtype=np.int64)
    np.put_along_axis(ix, ix0, temp, axis=1)

    iy0 = np.argsort(y, axis=1)
    vy = np.take_along_axis(y, iy0, axis=1)

    iy = np.zeros_like(y, dtype=np.int64)
    np.put_along_axis(iy, ix0, temp, axis=1)

    if compile_mode not in rowwise_impls_dict:
        return NotImplemented

    impl = rowwise_impls_dict[compile_mode]

    impl(
        x, y,
        ix, iy,
        vx, vy,
        unbiased, res)

    return res

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

    def _partial_sum_2d(x, y, c, ix, iy, sx_c, sy_c, c_sum, l_max,
                        s, gamma):  # pylint:disable=too-many-locals

        if compiled == NO_COMPILE:
            dyad_update = _dyad_update
        elif compiled == COMPILE:
            dyad_update = _dyad_update_compiled
        else:
            dyad_update = _dyad_update_compiled_gpu

        pos_sums = np.arange(l_max)
        pos_sums[:] = 2 ** (l_max - pos_sums)
        pos_sums = np.cumsum(pos_sums)

        gamma = dyad_update(y, c, gamma, l_max, s, pos_sums)

        # Step 7
        gamma = c_sum - c - 2 * sy_c[iy] - 2 * sx_c + 4 * gamma
        gamma = gamma[ix]

        return gamma

    return _partial_sum_2d


_partial_sum_2d = _generate_partial_sum_2d(compiled=NO_COMPILE)
_partial_sum_2d_compiled = numba.njit(
    float64[:](float64[:], float64[:], float64[:],
               int64[:], int64[:], float64[:], float64[:], float64,
               int64, float64[:], float64[:]),
    cache=True)(
    _generate_partial_sum_2d(compiled=COMPILE))

if cuda.is_available():
    _partial_sum_2d_compiled_gpu = cuda.jit(
        device=True)(
        _generate_partial_sum_2d(compiled=GPU))


def _generate_distance_covariance_sqr_avl_impl(compiled):

    def _distance_covariance_sqr_avl_impl(
            x, y, ix, iy, vx, vy, unbiased,
            iy_reord,
            c, sx_c, sy_c, c_sum, l_max, s, gamma):  # pylint:disable=too-many-locals
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

        x_reord = vx

        # Step 2
        new_y = iy_reord + 1.

        # Step 7
        gamma_1 = partial_sum_2d(
            x_reord, new_y, c[0], ix, iy_reord, sx_c[0], sy_c[0], c_sum[0],
            l_max, s[0], gamma[0])
        gamma_x = partial_sum_2d(
            x_reord, new_y, c[1], ix, iy_reord, sx_c[1], sy_c[1], c_sum[1],
            l_max, s[1], gamma[1])
        gamma_y = partial_sum_2d(
            x_reord, new_y, c[2], ix, iy_reord, sx_c[2], sy_c[2], c_sum[2],
            l_max, s[2], gamma[2])
        gamma_xy = partial_sum_2d(
            x_reord, new_y, c[3], ix, iy_reord, sx_c[3], sy_c[3], c_sum[3],
            l_max, s[3], gamma[3])

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
            float64[:], float64[:],
            boolean, int64[:],
            float64[:, :], float64[:, :], float64[:, :], float64[:], int64,
            float64[:, :], float64[:, :]),
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


def _get_impl_args(x, y, *, exponent=1, unbiased=False):
    """
    Get the parameters that cannot be computed in the GPU version.
    """

    if exponent != 1:
        raise ValueError(f"Exponent should be 1 but is {exponent} instead.")

    n = x.shape[-1]
    assert n > 3
    assert n == y.shape[-1]
    temp = range(n)

    argsort_x = np.argsort(x, axis=-1)
    vx = np.take_along_axis(x, argsort_x, axis=-1)

    ix = np.zeros_like(x, dtype=np.int64)
    np.put_along_axis(ix, argsort_x, temp, axis=-1)

    argsort_y = np.argsort(y, axis=-1)
    vy = np.take_along_axis(y, argsort_y, axis=-1)

    iy = np.zeros_like(y, dtype=np.int64)
    np.put_along_axis(iy, argsort_y, temp, axis=-1)

    y_reord = np.take_along_axis(y, argsort_x, axis=-1)
    x_times_y_reord = np.take_along_axis(x * y, argsort_x, axis=-1)

    c = np.stack((
        np.ones_like(x),
        vx,
        y_reord,
        x_times_y_reord
    ), axis=-2)

    argsort_y_reord = np.argsort(y_reord, axis=-1)
    c_reord = np.take_along_axis(
        c, argsort_y_reord[..., np.newaxis, :], axis=-1)

    iy_reord = np.zeros_like(y, dtype=np.int64)
    np.put_along_axis(iy_reord, argsort_y_reord, temp, axis=-1)

    sx_c = np.cumsum(c, axis=-1) - c
    sy_c = np.cumsum(c_reord, axis=-1) - c_reord

    c_sum = np.sum(c, axis=-1)

    # Get the smallest l such that n <= 2^l
    l_max = int(math.ceil(np.log2(n)))

    s_len = 2 ** (l_max + 1)
    s = np.zeros(c.shape[:-1] + (s_len,), dtype=c.dtype)

    gamma = np.zeros_like(c)

    return (x, y,
            ix, iy,
            vx, vy,
            unbiased,
            iy_reord,
            c, sx_c, sy_c,
            c_sum,
            l_max,
            s,
            gamma)


def _distance_covariance_sqr_avl_generic(
        x, y, *, exponent=1, unbiased=False, compile_mode=CompileMode.AUTO):
    """Fast algorithm for the squared distance covariance."""

    if exponent != 1:
        raise ValueError(f"Exponent should be 1 but is {exponent} instead.")

    x = np.asarray(x)
    y = np.asarray(y)

    assert 1 <= x.ndim <= 2
    if x.ndim == 2:
        assert x.shape[1] == 1
        x = x[:, 0]

    assert 1 <= y.ndim <= 2
    if y.ndim == 2:
        assert y.shape[1] == 1
        y = y[:, 0]

    if compile_mode not in impls_dict:
        raise NotImplementedError(
            f"Compile mode {compile_mode} not implemented.")

    args = _get_impl_args(x, y, exponent=exponent,
                          unbiased=unbiased)

    for impl in impls_dict[compile_mode]:

        try:

            return impl(*args)

        except TypeError as e:

            if compile_mode is not CompileMode.AUTO:
                raise e

            warnings.warn(f"Falling back to uncompiled AVL fast distance "
                          f"covariance because of TypeError exception "
                          f"raised: {e}. Rembember: only floating point "
                          f"values can be used in the compiled "
                          f"implementations.")


def _rowwise_distance_covariance_sqr_avl_generic_internal(
        x, y, ix, iy, vx, vy, unbiased, iy_reord,
        c, sx_c, sy_c, c_sum, l_max, s, gamma, res):

    res[0] = _distance_covariance_sqr_avl_impl_compiled(
        x, y, unbiased=unbiased,
        ix=ix, iy=iy,
        vx=vx, vy=vy,
        iy_reord=iy_reord,
        c=c,
        sx_c=sx_c,
        sy_c=sy_c,
        c_sum=c_sum,
        l_max=l_max,
        s=s,
        gamma=gamma)


if cuda.is_available():
    def _rowwise_distance_covariance_sqr_avl_generic_internal_gpu(
            x, y, ix, iy, vx, vy, unbiased, iy_reord,
            c, sx_c, sy_c, c_sum, l_max, s, gamma, res):

        res[0] = 0  # _partial_sum_2d_compiled_gpu(
        #x, y, c[0], ix, iy, sx_c[0], sy_c[0], c_sum[0], l_max, s[0],
        # gamma[0])

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
          boolean, int64[:],
          float64[:, :], float64[:, :], float64[:, :], float64[:],
          int64, float64[:, :], float64[:, :], float64[:])],
        '(n),(n),(n),(n),(n),(n),(),(n),(m,n),(m,n),(m,n),(m),(),(m,l),(m,n)->()',
        nopython=True,
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

    x = np.asarray(x)
    y = np.asarray(y)

    assert 2 <= x.ndim <= 3
    if x.ndim == 3:
        assert x.shape[2] == 1
        x = x[..., 0]

    assert 2 <= y.ndim <= 3
    if y.ndim == 3:
        assert y.shape[2] == 1
        y = y[..., 0]

    res = np.zeros(x.shape[0], dtype=x.dtype)

    if compile_mode not in rowwise_impls_dict:
        return NotImplemented

    impl = rowwise_impls_dict[compile_mode]

    impl(
        *_get_impl_args(x, y, exponent=exponent,
                        unbiased=unbiased),
        res)

    return res

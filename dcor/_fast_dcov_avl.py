'''
Functions to compute fast distance covariance using AVL.
'''
from __future__ import annotations

import math
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Tuple,
    TypeVar,
    overload,
)

import numba
import numpy as np
from numba import boolean, float64, int64
from numba.types import Tuple as NumbaTuple

from ._dcor_internals_numba import (
    NumbaIntVectorReadOnly,
    NumbaMatrix,
    NumbaMatrixReadOnly,
    NumbaVector,
    NumbaVectorReadOnly,
    NumbaVectorReadOnlyNonContiguous,
    _generate_distance_covariance_sqr_from_terms_impl,
)
from ._utils import CompileMode, _transform_to_1d

if TYPE_CHECKING:
    NumpyArrayType = np.typing.NDArray[np.number[Any]]
else:
    NumpyArrayType = np.ndarray


Array = TypeVar("Array", bound=NumpyArrayType)


def _dyad_update(
    y: np.typing.NDArray[np.float64],
    c: np.typing.NDArray[np.float64],
    gamma: np.typing.NDArray[np.float64],
    l_max: int,
    s: np.typing.NDArray[np.float64],
    pos_sums: np.typing.NDArray[np.int64],
) -> np.typing.NDArray[np.float64]:  # pylint:disable=too-many-locals
    # This function has many locals so it can be compared
    # with the original algorithm.
    """
    Inner function of the fast distance covariance.

    This function is compiled because otherwise it would become
    a bottleneck.

    """
    s[...] = 0
    exps2 = 2 ** np.arange(l_max)

    y_col = y[:, np.newaxis]

    # Step 3.a: update s(l, k)
    positions_3a = np.ceil(y_col / exps2).astype(np.int64) - 1
    positions_3a[:, 1:] += pos_sums[:-1]

    # Steps 3.b and 3.c
    positions_3b = np.floor((y_col - 1) / exps2).astype(np.int64) - 1
    valid_positions = positions_3b % 2 == 0
    positions_3b[:, 1:] += pos_sums[:-1]

    # Caution: vectorizing this loop naively can cause the algorithm
    # to use N^2 memory!!
    np_sum = np.sum

    for i, (pos_a, pos_b, valid, c_i) in enumerate(
        zip(positions_3a, positions_3b, valid_positions, c),
    ):
        # Steps 3.b and 3.c
        gamma[i] = np_sum(s[pos_b[valid]])

        # Step 3.a: update s(l, k)
        s[pos_a] += c_i

    return gamma


def _dyad_update_compiled_version(
    y: np.typing.NDArray[np.float64],
    c: np.typing.NDArray[np.float64],
    gamma: np.typing.NDArray[np.float64],
    l_max: int,
    s: np.typing.NDArray[np.float64],
    pos_sums: np.typing.NDArray[np.int64],
) -> np.typing.NDArray[np.float64]:  # pylint:disable=too-many-locals
    # This function has many locals so it can be compared
    # with the original algorithm.
    """
    Inner function of the fast distance covariance.

    This function is compiled because otherwise it would become
    a bottleneck.

    """
    n = y.shape[0]
    s[...] = 0

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
            pos = int(math.floor((y[i] - 1) / 2 ** l)) - 1
            if pos % 2 == 0:
                if l > 0:
                    pos += pos_sums[l - 1]

                gamma[i] += s[pos]

    return gamma


_dyad_update_compiled = numba.njit(
    NumbaVectorReadOnly(
        NumbaVectorReadOnly,
        NumbaVectorReadOnly,
        NumbaVector,
        int64,
        NumbaVector,
        NumbaIntVectorReadOnly,
    ),
    cache=True,
)(
    _dyad_update_compiled_version,
)


def _generate_partial_sum_2d(
    compiled: bool,
) -> Callable[..., np.typing.NDArray[np.float64]]:

    def _partial_sum_2d(
        x: np.typing.NDArray[np.float64],
        y: np.typing.NDArray[np.float64],
        c: np.typing.NDArray[np.float64],
        ix: np.typing.NDArray[np.int64],
        iy: np.typing.NDArray[np.int64],
        sx_c: np.typing.NDArray[np.float64],
        sy_c: np.typing.NDArray[np.float64],
        c_sum: float,
        l_max: int,
        s: np.typing.NDArray[np.float64],
        pos_sums: np.typing.NDArray[np.int64],
        gamma: np.typing.NDArray[np.float64],
    ) -> np.typing.NDArray[np.float64]:  # pylint:disable=too-many-locals

        dyad_update = _dyad_update_compiled if compiled else _dyad_update

        gamma = dyad_update(y, c, gamma, l_max, s, pos_sums)

        # Step 7
        gamma = c_sum - c - 2 * sy_c[iy] - 2 * sx_c + 4 * gamma
        return gamma[ix]

    return _partial_sum_2d


_partial_sum_2d = _generate_partial_sum_2d(compiled=False)
_partial_sum_2d_compiled = numba.njit(
    NumbaVectorReadOnly(
        NumbaVectorReadOnly,
        NumbaVectorReadOnly,
        NumbaVectorReadOnly,
        NumbaIntVectorReadOnly,
        NumbaIntVectorReadOnly,
        NumbaVectorReadOnly,
        NumbaVectorReadOnly,
        float64,
        int64,
        NumbaVector,
        NumbaIntVectorReadOnly,
        NumbaVector,
    ),
    cache=True,
)(
    _generate_partial_sum_2d(compiled=True),
)


def _get_impl_args(
    x: np.typing.NDArray[np.float64],
    y: np.typing.NDArray[np.float64],
):
    """Get the parameters used in the algorithm."""
    x = np.ascontiguousarray(x)
    y = np.ascontiguousarray(y)

    n = x.shape[0]
    assert n > 3
    assert n == y.shape[0]
    temp = np.arange(n)

    argsort_x = np.argsort(x)
    vx = x[argsort_x]

    ix = np.zeros_like(x, dtype=np.int64)
    ix[argsort_x] = temp

    argsort_y = np.argsort(y)
    vy = y[argsort_y]

    iy = np.zeros_like(y, dtype=np.int64)
    iy[argsort_y] = temp

    y_reord = y[argsort_x]
    x_times_y_reord = (x * y)[argsort_x]

    argsort_y_reord = np.argsort(y_reord)

    iy_reord = np.zeros_like(y, dtype=np.int64)
    iy_reord[argsort_y_reord] = temp

    c = np.stack(
        (
            np.ones_like(x),
            vx,
            y_reord,
            x_times_y_reord,
        ),
        axis=-2,
    )

    c_reord = np.zeros_like(c)
    sx_c = np.zeros_like(c)
    sy_c = np.zeros_like(c)

    for i, (c_elem, c_reord_elem) in enumerate(zip(c, c_reord)):
        c_reord[i] = c_elem[argsort_y_reord]

        sx_c[i] = np.cumsum(c_elem) - c_elem
        sy_c[i] = np.cumsum(c_reord_elem) - c_reord_elem

    c_sum = np.sum(c, axis=-1)

    # Get the smallest l such that n <= 2^l
    l_max = int(math.ceil(np.log2(n)))

    s_len = 2 ** (l_max + 1)
    s = np.empty(s_len, dtype=c.dtype)

    pos_sums = np.arange(l_max, dtype=np.int64)
    pos_sums[:] = 2 ** (l_max - pos_sums)
    pos_sums = np.cumsum(pos_sums)

    gamma = np.zeros_like(c)

    return (
        x,
        y,
        ix,
        iy,
        vx,
        vy,
        iy_reord,
        c,
        sx_c,
        sy_c,
        c_sum,
        l_max,
        s,
        pos_sums,
        gamma,
    )


_get_impl_args_compiled = numba.njit(
    NumbaTuple((
        NumbaVectorReadOnly,
        NumbaVectorReadOnly,
        NumbaIntVectorReadOnly,
        NumbaIntVectorReadOnly,
        NumbaVectorReadOnly,
        NumbaVectorReadOnly,
        NumbaIntVectorReadOnly,
        NumbaMatrixReadOnly,
        NumbaMatrixReadOnly,
        NumbaMatrixReadOnly,
        NumbaVectorReadOnly,
        int64,
        NumbaVector,
        NumbaIntVectorReadOnly,
        NumbaMatrix,
    ))(NumbaVectorReadOnlyNonContiguous, NumbaVectorReadOnlyNonContiguous),
    cache=True,
)(_get_impl_args)


def _generate_distance_covariance_sqr_terms_avl_impl(
    compiled: bool,
) -> Callable[..., Any]:

    def _distance_covariance_sqr_terms_avl_impl(
        x: np.typing.NDArray[np.float64],
        y: np.typing.NDArray[np.float64],
        return_var_terms: bool = False,
    ) -> Any:  # pylint:disable=too-many-locals
        # This function has many locals so it can be compared
        # with the original algorithm.
        """Fast algorithm for the squared distance covariance."""
        partial_sum_2d = (
            _partial_sum_2d_compiled
            if compiled
            else _partial_sum_2d
        )
        get_impl_args = (
            _get_impl_args_compiled
            if compiled
            else _get_impl_args
        )

        (
            x,
            y,
            ix,
            iy,
            vx,
            vy,
            iy_reord,
            c,
            sx_c,
            sy_c,
            c_sum,
            l_max,
            s,
            pos_sums,
            gamma,
        ) = get_impl_args(x, y)

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

        # Step 6
        a_dot_dot = 2 * np.sum(alpha_x * x) - 2 * np.sum(beta_x)
        b_dot_dot = 2 * np.sum(alpha_y * y) - 2 * np.sum(beta_y)

        x_reord = vx

        # Step 2
        new_y = iy_reord + 1.0

        # Step 7
        gamma_1, gamma_x, gamma_y, gamma_xy = [
            partial_sum_2d(
                x_reord,
                new_y,
                c[i],
                ix,
                iy_reord,
                sx_c[i],
                sy_c[i],
                c_sum[i],
                l_max,
                s,
                pos_sums,
                gamma[i],
            ) for i in range(4)
        ]

        # Step 8
        aijbij = np.sum(x * y * gamma_1 + gamma_xy - x * gamma_y - y * gamma_x)

        # Step 9
        if return_var_terms:
            return (
                aijbij,
                a_i_dot,
                a_dot_dot,
                b_i_dot,
                b_dot_dot,
                2 * n**2 * np.var(x),
                2 * n**2 * np.var(y),
            )

        return aijbij, a_i_dot, a_dot_dot, b_i_dot, b_dot_dot, None, None

    return _distance_covariance_sqr_terms_avl_impl


_distance_covariance_sqr_terms_avl_impl = _generate_distance_covariance_sqr_terms_avl_impl(
    compiled=False,
)
_distance_covariance_sqr_terms_avl_impl_compiled = numba.njit(
    NumbaTuple((
        float64,
        NumbaVector,
        float64,
        NumbaVector,
        float64,
        numba.optional(float64),
        numba.optional(float64),
    ))(NumbaVectorReadOnlyNonContiguous, NumbaVectorReadOnlyNonContiguous, boolean),
    cache=True,
)(
    _generate_distance_covariance_sqr_terms_avl_impl(compiled=True),
)


_distance_covariance_sqr_avl_impl = _generate_distance_covariance_sqr_from_terms_impl(
    compiled=False,
    terms_compiled=_distance_covariance_sqr_terms_avl_impl_compiled,
    terms_uncompiled=_distance_covariance_sqr_terms_avl_impl,
)
_distance_covariance_sqr_avl_impl_compiled = numba.njit(
    float64(
        NumbaVectorReadOnlyNonContiguous,
        NumbaVectorReadOnlyNonContiguous,
        boolean,
    ),
    cache=True,
)(
    _generate_distance_covariance_sqr_from_terms_impl(
        compiled=True,
        terms_compiled=_distance_covariance_sqr_terms_avl_impl_compiled,
        terms_uncompiled=_distance_covariance_sqr_terms_avl_impl,
    ),
)


impls_dict = {
    CompileMode.AUTO: (
        _distance_covariance_sqr_terms_avl_impl_compiled,
        _distance_covariance_sqr_terms_avl_impl,
    ),
    CompileMode.NO_COMPILE: (
        _distance_covariance_sqr_terms_avl_impl,
    ),
    CompileMode.COMPILE_CPU: (
        _distance_covariance_sqr_terms_avl_impl_compiled,
    ),
}


@overload
def _distance_covariance_sqr_terms_avl(
    __x: Array,
    __y: Array,
    *,
    exponent: float,
    compile_mode: CompileMode = CompileMode.AUTO,
    return_var_terms: Literal[False] = False,
) -> Tuple[
    Array,
    Array,
    Array,
    Array,
    Array,
    None,
    None,
]:
    ...


@overload
def _distance_covariance_sqr_terms_avl(
    __x: Array,
    __y: Array,
    *,
    exponent: float,
    compile_mode: CompileMode = CompileMode.AUTO,
    return_var_terms: Literal[True],
) -> Tuple[
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
]:
    ...


def _distance_covariance_sqr_terms_avl(
    x: Array,
    y: Array,
    *,
    exponent: float = 1,
    compile_mode: CompileMode = CompileMode.AUTO,
    return_var_terms: bool = False,
) -> Tuple[
    Array,
    Array,
    Array,
    Array,
    Array,
    None,
    None,
]:
    """Fast algorithm for the squared distance covariance terms."""
    if exponent != 1:
        raise ValueError(f"Exponent should be 1 but is {exponent} instead.")

    x, y = _transform_to_1d(x, y)

    if not isinstance(x, np.ndarray):
        raise ValueError("AVL method is only implemented for NumPy arrays.")

    if compile_mode not in impls_dict:
        raise NotImplementedError(
            f"Compile mode {compile_mode} not implemented.",
        )

    for impl in impls_dict[compile_mode]:

        try:

            return impl(x, y, return_var_terms)

        except TypeError as e:

            if compile_mode is not CompileMode.AUTO:
                raise e

            warnings.warn(
                f"Falling back to uncompiled AVL fast distance "
                f"covariance terms because of TypeError exception "
                f"raised: {e}. Rembember: only floating point "
                f"values can be used in the compiled "
                f"implementations.",
            )


def _generate_rowwise_internal(
    target: Literal["cpu", "parallel"],
) -> Callable[..., NumpyArrayType]:

    def _rowwise_distance_covariance_sqr_avl_generic_internal(
        x: Array,
        y: Array,
        unbiased: bool,
        res: Array,
    ) -> Array:

        res[0] = _distance_covariance_sqr_avl_impl_compiled(x, y, unbiased)

    return numba.guvectorize(
        [(
            NumbaVectorReadOnlyNonContiguous,
            NumbaVectorReadOnlyNonContiguous,
            boolean,
            float64[:],
        )],
        '(n),(n),()->()',
        nopython=True,
        cache=True,
        target=target,
    )(_rowwise_distance_covariance_sqr_avl_generic_internal)


_rowwise_distance_covariance_sqr_avl_generic_internal_cpu = (
    _generate_rowwise_internal(target='cpu')
)
_rowwise_distance_covariance_sqr_avl_generic_internal_parallel = (
    _generate_rowwise_internal(target='parallel')
)

rowwise_impls_dict = {
    CompileMode.AUTO:
    _rowwise_distance_covariance_sqr_avl_generic_internal_parallel,
    CompileMode.COMPILE_CPU:
    _rowwise_distance_covariance_sqr_avl_generic_internal_cpu,
    CompileMode.COMPILE_PARALLEL:
    _rowwise_distance_covariance_sqr_avl_generic_internal_parallel,
}


def _rowwise_distance_covariance_sqr_avl_generic(
    x: Array,
    y: Array,
    exponent: float = 1,
    unbiased: bool = False,
    compile_mode: CompileMode = CompileMode.AUTO,
) -> Array:

    if exponent != 1:
        raise ValueError(f"Exponent should be 1 but is {exponent} instead.")

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

    impl(x, y, unbiased, res)

    return res

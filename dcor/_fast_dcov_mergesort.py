'''
Functions to compute fast distance covariance using mergesort.
'''
from __future__ import annotations

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
from numba import boolean, float64
from numba.types import Tuple as NumbaTuple

from ._dcor_internals_numba import (
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


def _compute_weight_sums(
    y: np.typing.NDArray[np.float64],
    weights: np.typing.NDArray[np.float64],
) -> np.typing.NDArray[np.float64]:

    n_samples = y.shape[0]

    weight_sums = np.zeros(weights.shape[1:] + (n_samples,), dtype=y.dtype)

    # Buffer that contains the indexes of the current and
    # last iterations
    indexes = np.arange(2 * n_samples).reshape((2, n_samples))
    indexes[1] = 0  # Remove this

    previous_indexes = indexes[0]
    current_indexes = indexes[1]

    weights_cumsum = np.zeros(
        (n_samples + 1,) + weights.shape[1:],
        dtype=weights.dtype,
    )

    merged_subarray_len = 1

    # For all lengths that are a power of two
    while merged_subarray_len < n_samples:
        gap = 2 * merged_subarray_len
        indexes_idx = 0

        # Numba does not support axis, nor out parameter.
        for var in range(weights.shape[1]):
            weights_cumsum[1:, var] = np.cumsum(
                weights[previous_indexes, var],
            )

        # Select the subarrays in pairs
        for subarray_pair_idx in range(0, n_samples, gap):
            subarray_1_idx = subarray_pair_idx
            subarray_2_idx = subarray_pair_idx + merged_subarray_len
            subarray_1_idx_last = min(
                subarray_1_idx + merged_subarray_len - 1,
                n_samples - 1,
            )
            subarray_2_idx_last = min(
                subarray_2_idx + merged_subarray_len - 1,
                n_samples - 1,
            )

            # Merge the subarrays
            while (
                subarray_1_idx <= subarray_1_idx_last
                and subarray_2_idx <= subarray_2_idx_last
            ):
                previous_index_1 = previous_indexes[subarray_1_idx]
                previous_index_2 = previous_indexes[subarray_2_idx]

                if y[previous_index_1].item() >= y[previous_index_2].item():
                    current_indexes[indexes_idx] = previous_index_1
                    subarray_1_idx += 1
                else:
                    current_indexes[indexes_idx] = previous_index_2
                    subarray_2_idx += 1

                    weight_sums[:, previous_index_2] += (
                        weights_cumsum[subarray_1_idx_last + 1]
                        - weights_cumsum[subarray_1_idx]
                    )
                indexes_idx += 1

            # Join the remaining elements of one of the arrays (already sorted)
            if subarray_1_idx <= subarray_1_idx_last:
                n_remaining = subarray_1_idx_last - subarray_1_idx + 1
                indexes_idx_next = indexes_idx + n_remaining
                current_indexes[indexes_idx:indexes_idx_next] = (
                    previous_indexes[subarray_1_idx:subarray_1_idx_last + 1]
                )
                indexes_idx = indexes_idx_next
            elif subarray_2_idx <= subarray_2_idx_last:
                n_remaining = subarray_2_idx_last - subarray_2_idx + 1
                indexes_idx_next = indexes_idx + n_remaining
                current_indexes[indexes_idx:indexes_idx_next] = (
                    previous_indexes[subarray_2_idx:subarray_2_idx_last + 1]
                )
                indexes_idx = indexes_idx_next

        merged_subarray_len = gap

        # Swap buffer
        previous_indexes, current_indexes = (current_indexes, previous_indexes)

    return weight_sums


_compute_weight_sums_compiled = numba.njit(
    NumbaMatrix(NumbaVectorReadOnly, NumbaMatrixReadOnly),
    cache=True,
)(_compute_weight_sums)


def _generate_compute_aijbij_term(
    compiled: bool,
) -> Callable[..., np.typing.NDArray[np.float64]]:
    def _compute_aijbij_term(
        x: np.typing.NDArray[np.float64],
        y: np.typing.NDArray[np.float64],
    ) -> np.typing.NDArray[np.float64]:

        compute_weight_sums = (
            _compute_weight_sums_compiled
            if compiled
            else _compute_weight_sums
        )

        # x must be sorted
        n = x.shape[0]

        weights = np.column_stack((np.ones_like(y), y, x, x * y))
        weight_sums = compute_weight_sums(y, weights)

        term_1 = (x * y) @ weight_sums[0]
        term_2 = x @ weight_sums[1]
        term_3 = y @ weight_sums[2]
        term_4 = np.sum(weight_sums[3])

        # First term in the equation
        sums_term = term_1 - term_2 - term_3 + term_4

        # Second term in the equation
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        cov_term = n * x @ y - np.sum(sum_x * y + sum_y * x) + sum_x * sum_y

        d = 4 * sums_term - 2 * cov_term

        return d.item()

    return _compute_aijbij_term


_compute_aijbij_term = _generate_compute_aijbij_term(compiled=False)
_compute_aijbij_term_compiled = numba.njit(
    float64(NumbaVectorReadOnly, NumbaVectorReadOnly),
    cache=True,
)(
    _generate_compute_aijbij_term(
        compiled=True,
    ),
)


def _compute_row_sums(
    x: np.typing.NDArray[np.float64],
) -> np.typing.NDArray[np.float64]:
    # x must be sorted

    n_samples = x.shape[0]

    term_1 = (2 * np.arange(1, n_samples + 1) - n_samples) * x

    sums = np.cumsum(x)

    term_2 = sums[-1] - 2 * sums

    return term_1 + term_2


_compute_row_sums_compiled = numba.njit(
    NumbaVector(NumbaVectorReadOnly),
    cache=True)(_compute_row_sums)


def _generate_distance_covariance_sqr_terms_mergesort_impl(
    compiled: bool,
) -> Callable[..., Any]:

    def _distance_covariance_sqr_terms_mergesort_impl(
        x: np.typing.NDArray[np.float64],
        y: np.typing.NDArray[np.float64],
        return_var_terms: bool = False,
    ) -> Any:

        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)

        compute_aijbij_term = (
            _compute_aijbij_term_compiled
            if compiled
            else _compute_aijbij_term
        )
        compute_row_sums = (
            _compute_row_sums_compiled
            if compiled
            else _compute_row_sums
        )

        n = x.shape[0]

        # Sort x in ascending order
        ordered_indexes = np.argsort(x)
        x = x[ordered_indexes]
        y = y[ordered_indexes]

        aijbij = compute_aijbij_term(x, y)
        a_i = compute_row_sums(x)

        ordered_indexes_y = np.argsort(y)
        b_i_perm = compute_row_sums(y[ordered_indexes_y])
        b_i = np.empty_like(b_i_perm)
        b_i[ordered_indexes_y] = b_i_perm

        a_dot_dot = np.sum(a_i)
        b_dot_dot = np.sum(b_i)

        if return_var_terms:
            return (
                aijbij,
                a_i,
                a_dot_dot,
                b_i,
                b_dot_dot,
                2 * n**2 * np.var(x),
                2 * n**2 * np.var(y),
            )

        return aijbij, a_i, a_dot_dot, b_i, b_dot_dot, None, None

    return _distance_covariance_sqr_terms_mergesort_impl


_distance_covariance_sqr_terms_mergesort_impl = (
    _generate_distance_covariance_sqr_terms_mergesort_impl(
        compiled=False,
    )
)
_distance_covariance_sqr_terms_mergesort_impl_compiled = numba.njit(
    NumbaTuple((
        float64,
        NumbaVector,
        float64,
        NumbaVector,
        float64,
        numba.optional(float64),
        numba.optional(float64),
    ))(
        NumbaVectorReadOnlyNonContiguous,
        NumbaVectorReadOnlyNonContiguous,
        boolean,
    ),
    cache=True,
)(
    _generate_distance_covariance_sqr_terms_mergesort_impl(
        compiled=True,
    ),
)


_distance_covariance_sqr_mergesort_generic_impl = (
    _generate_distance_covariance_sqr_from_terms_impl(
        compiled=False,
        terms_compiled=_distance_covariance_sqr_terms_mergesort_impl_compiled,
        terms_uncompiled=_distance_covariance_sqr_terms_mergesort_impl,
    )
)
_distance_covariance_sqr_mergesort_generic_impl_compiled = numba.njit(
    float64(
        NumbaVectorReadOnlyNonContiguous,
        NumbaVectorReadOnlyNonContiguous,
        boolean,
    ),
    cache=True,
)(
    _generate_distance_covariance_sqr_from_terms_impl(
        compiled=True,
        terms_compiled=_distance_covariance_sqr_terms_mergesort_impl_compiled,
        terms_uncompiled=_distance_covariance_sqr_terms_mergesort_impl,
    ),
)

impls_dict = {
    CompileMode.AUTO: (
        _distance_covariance_sqr_terms_mergesort_impl_compiled,
        _distance_covariance_sqr_terms_mergesort_impl,
    ),
    CompileMode.NO_COMPILE: (
        _distance_covariance_sqr_terms_mergesort_impl,
    ),
    CompileMode.COMPILE_CPU: (
        _distance_covariance_sqr_terms_mergesort_impl_compiled,
    ),
}


@overload
def _distance_covariance_sqr_terms_mergesort(
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
def _distance_covariance_sqr_terms_mergesort(
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


def _distance_covariance_sqr_terms_mergesort(
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
    Array | None,
    Array | None,
]:

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
                f"Falling back to uncompiled MERGESORT fast "
                f"distance covariance because of TypeError "
                f"exception raised: {e}. Rembember: only floating "
                f"point values can be used in the compiled "
                f"implementations.",
            )

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import numba
import numpy as np
from numba import boolean, float64, int64
from numba.types import Array, Tuple

from ._dcor_internals import _dcov_from_terms

if TYPE_CHECKING:
    import numpy.typing


NumbaVector = Array(dtype=float64, ndim=1, layout="C")
NumbaVectorReadOnly = Array(dtype=float64, ndim=1, layout="C", readonly=True)
NumbaVectorReadOnlyNonContiguous = Array(
    dtype=float64,
    ndim=1,
    layout="A",
    readonly=True,
)
NumbaIntVector = Array(dtype=int64, ndim=1, layout="C")
NumbaIntVectorReadOnly = Array(dtype=int64, ndim=1, layout="C", readonly=True)
NumbaMatrix = Array(dtype=float64, ndim=2, layout="C")
NumbaMatrixReadOnly = Array(dtype=float64, ndim=2, layout="C", readonly=True)

_dcov_from_terms_compiled = numba.njit(
    float64(
        float64,
        NumbaVectorReadOnly,
        float64,
        NumbaVectorReadOnly,
        float64,
        int64,
        boolean,
    ),
    cache=True,
)(_dcov_from_terms)


def _generate_distance_covariance_sqr_from_terms_impl(
    compiled: bool,
    terms_compiled: Callable[..., Any],
    terms_uncompiled: Callable[..., Any],
) -> Callable[..., np.typing.NDArray[np.float64]]:

    def _distance_covariance_sqr_from_terms_impl(
        x: np.typing.NDArray[np.float64],
        y: np.typing.NDArray[np.float64],
        unbiased: bool,
    ) -> np.typing.NDArray[np.float64]:  # pylint:disable=too-many-locals
        # This function has many locals so it can be compared
        # with the original algorithm.
        """Fast algorithm for the squared distance covariance."""
        distance_covariance_sqr_terms = (
            terms_compiled
            if compiled
            else terms_uncompiled
        )
        dcov_from_terms = (
            _dcov_from_terms_compiled
            if compiled
            else _dcov_from_terms
        )

        n = x.shape[0]

        (
            aijbij,
            a_i_dot,
            a_dot_dot,
            b_i_dot,
            b_dot_dot,
            _,
            _,
        ) = distance_covariance_sqr_terms(x, y, return_var_terms=False)

        # Step 9
        return dcov_from_terms(
            aijbij,
            a_i_dot,
            a_dot_dot,
            b_i_dot,
            b_dot_dot,
            n,
            unbiased,
        )

    return _distance_covariance_sqr_from_terms_impl

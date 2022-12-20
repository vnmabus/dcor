import numba
from numba import boolean, float64, int64
from numba.types import Array, Tuple

from ._dcor_internals import _dcov_from_terms

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

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
from __future__ import annotations

from dataclasses import astuple, dataclass
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Iterator,
    TypeVar,
    Union,
)

import numpy as np

from dcor._dcor_internals import _af_inv_scaled

from ._dcor_internals import (
    MatrixCentered,
    _distance_matrix,
    _u_distance_matrix,
    mean_product,
    u_product,
)
from ._fast_dcov_avl import _distance_covariance_sqr_avl_generic
from ._fast_dcov_mergesort import _distance_covariance_sqr_mergesort_generic
from ._utils import ArrayType, CompileMode, _sqrt, get_namespace

T = TypeVar("T", bound=ArrayType)

if TYPE_CHECKING:
    try:
        from typing import Protocol
    except ImportError:
        from typing_extensions import Protocol
else:
    Protocol = object


class DCovFunction(Protocol):
    """Callback protocol for centering method."""

    def __call__(self, __x: T, __y: T, *, compile_mode: CompileMode) -> T:
        ...


class _DcovAlgorithmInternals():

    def __init__(
        self,
        *,
        dcov_sqr=None,
        u_dcov_sqr=None,
        dcor_sqr=None,
        u_dcor_sqr=None,
        stats_sqr=None,
        u_stats_sqr=None,
        dcov_generic=None,
        stats_generic=None,
    ):

        # Dcov and U-Dcov
        if dcov_generic is not None:
            self.dcov_sqr = (
                lambda *args, **kwargs: dcov_generic(
                    *args,
                    **kwargs,
                    unbiased=False,
                )
            )
            self.u_dcov_sqr = (
                lambda *args, **kwargs: dcov_generic(
                    *args,
                    **kwargs,
                    unbiased=True,
                )
            )
        else:
            self.dcov_sqr = dcov_sqr
            self.u_dcov_sqr = u_dcov_sqr

        # Stats
        if stats_sqr is not None:
            self.stats_sqr = stats_sqr
        else:
            if stats_generic is None:
                self.stats_sqr = (
                    lambda *args, **kwargs: _distance_stats_sqr_generic(
                        *args,
                        **kwargs,
                        dcov_function=self.dcov_sqr,
                    )
                )
            else:
                self.stats_sqr = (
                    lambda *args, **kwargs: stats_generic(
                        *args,
                        **kwargs,
                        matrix_centered=_distance_matrix,
                        product=mean_product,
                    )
                )

        # U-Stats
        if u_stats_sqr is not None:
            self.u_stats_sqr = u_stats_sqr
        else:
            if stats_generic is None:
                self.u_stats_sqr = (
                    lambda *args, **kwargs: _distance_stats_sqr_generic(
                        *args,
                        **kwargs,
                        dcov_function=self.u_dcov_sqr,
                    )
                )
            else:
                self.u_stats_sqr = (
                    lambda *args, **kwargs: stats_generic(
                        *args,
                        **kwargs,
                        matrix_centered=_u_distance_matrix,
                        product=u_product,
                    )
                )

        # Dcor
        if dcor_sqr is not None:
            self.dcor_sqr = dcor_sqr
        else:
            self.dcor_sqr = lambda *args, **kwargs: self.stats_sqr(
                *args,
                **kwargs,
            ).correlation_xy

        # U-Dcor
        if u_dcor_sqr is not None:
            self.u_dcor_sqr = u_dcor_sqr
        else:
            self.u_dcor_sqr = lambda *args, **kwargs: self.u_stats_sqr(
                *args,
                **kwargs,
            ).correlation_xy


class _DcovAlgorithmInternalsAuto():
    def _dispatch(
        self,
        x: T,
        y: T,
        *,
        method: str,
        exponent: float,
        **kwargs: Any,
    ) -> Any:
        xp = get_namespace(x, y)

        if xp == np and _can_use_fast_algorithm(x, y, exponent):
            return getattr(DistanceCovarianceMethod.AVL.value, method)(
                x,
                y,
                exponent=exponent,
                **kwargs,
            )
        else:
            return getattr(
                DistanceCovarianceMethod.NAIVE.value, method)(
                    x,
                    y,
                    exponent=exponent,
                    **kwargs,
            )

    def __getattr__(self, method: str) -> Any:
        if method[0] != '_':
            return lambda *args, **kwargs: self._dispatch(
                *args,
                **kwargs,
                method=method,
            )
        else:
            raise AttributeError(
                f"{self.__class__.__name__!r} object has no "
                f"attribute {method!r}",
            )


@dataclass(frozen=True)
class Stats(Generic[T]):
    """Distance covariance related stats."""
    covariance_xy: T
    correlation_xy: T
    variance_x: T
    variance_y: T

    def __iter__(self) -> Iterator[T]:
        return iter(astuple(self))


def _naive_check_compile_mode(compile_mode: CompileMode) -> None:
    """Check that compile mode is AUTO or NO_COMPILE and raises otherwise."""

    if compile_mode not in (CompileMode.AUTO, CompileMode.NO_COMPILE):
        raise NotImplementedError(
            f"Compile mode {compile_mode} not implemented.",
        )


def _distance_covariance_sqr_naive(
    x: T,
    y: T,
    exponent: float = 1,
    compile_mode: CompileMode = CompileMode.AUTO,
) -> T:
    """
    Naive biased estimator for distance covariance.

    Computes the unbiased estimator for distance covariance between two
    matrices, using an :math:`O(N^2)` algorithm.

    """
    _naive_check_compile_mode(compile_mode)

    a = _distance_matrix(x, exponent=exponent)
    b = _distance_matrix(y, exponent=exponent)

    return mean_product(a, b)


def _u_distance_covariance_sqr_naive(
    x: T,
    y: T,
    exponent: float = 1,
    compile_mode: CompileMode = CompileMode.AUTO,
) -> T:
    """
    Naive unbiased estimator for distance covariance.

    Computes the unbiased estimator for distance covariance between two
    matrices, using an :math:`O(N^2)` algorithm.

    """
    _naive_check_compile_mode(compile_mode)

    a = _u_distance_matrix(x, exponent=exponent)
    b = _u_distance_matrix(y, exponent=exponent)

    return u_product(a, b)


def _distance_sqr_stats_naive_generic(
    x: T,
    y: T,
    matrix_centered: MatrixCentered,
    product: Callable[[T, T], T],
    exponent: float = 1,
    compile_mode: CompileMode = CompileMode.AUTO,
) -> Stats[T]:
    """Compute generic squared stats."""
    _naive_check_compile_mode(compile_mode)

    a = matrix_centered(x, exponent=exponent)
    b = matrix_centered(y, exponent=exponent)

    covariance_xy_sqr = product(a, b)
    variance_x_sqr = product(a, a)
    variance_y_sqr = product(b, b)

    xp = get_namespace(x, y)

    denominator_sqr = xp.abs(variance_x_sqr * variance_y_sqr)
    denominator = _sqrt(denominator_sqr)

    # Comparisons using a tolerance can change results if the
    # covariance has a similar order of magnitude
    if denominator == 0.0:
        correlation_xy_sqr = xp.zeros_like(covariance_xy_sqr)
    else:
        correlation_xy_sqr = covariance_xy_sqr / denominator

    return Stats(
        covariance_xy=covariance_xy_sqr,
        correlation_xy=correlation_xy_sqr,
        variance_x=variance_x_sqr,
        variance_y=variance_y_sqr,
    )


def _distance_correlation_sqr_naive(
    x: T,
    y: T,
    exponent: float = 1,
    compile_mode: CompileMode = CompileMode.AUTO,
) -> T:
    """Biased distance correlation estimator between two matrices."""
    _naive_check_compile_mode(compile_mode)

    return _distance_sqr_stats_naive_generic(
        x,
        y,
        matrix_centered=_distance_matrix,
        product=mean_product,
        exponent=exponent,
    ).correlation_xy


def _u_distance_correlation_sqr_naive(
    x: T,
    y: T,
    exponent: float = 1,
    compile_mode: CompileMode = CompileMode.AUTO,
) -> T:
    """Bias-corrected distance correlation estimator between two matrices."""
    _naive_check_compile_mode(compile_mode)

    return _distance_sqr_stats_naive_generic(
        x,
        y,
        matrix_centered=_u_distance_matrix,
        product=u_product,
        exponent=exponent,
    ).correlation_xy


def _is_random_variable(x: T) -> bool:
    """
    Check if the matrix x correspond to a random variable.

    The matrix is considered a random variable if it is a vector
    or a matrix corresponding to a column vector. Otherwise,
    the matrix correspond to a random vector.
    """
    return len(x.shape) == 1 or x.shape[1] == 1


def _can_use_fast_algorithm(x: T, y: T, exponent: float = 1) -> bool:
    """
    Check if the fast algorithm for distance stats can be used.

    The fast algorithm has complexity :math:`O(NlogN)`, better than the
    complexity of the naive algorithm (:math:`O(N^2)`).

    The algorithm can only be used for random variables (not vectors) where
    the number of instances is greater than 3. Also, the exponent must be 1.

    """
    return (
        _is_random_variable(x) and _is_random_variable(y)
        and x.shape[0] > 3 and y.shape[0] > 3 and exponent == 1
    )


def _distance_stats_sqr_generic(
    x: T,
    y: T,
    *,
    exponent: float = 1,
    dcov_function: DCovFunction,
    compile_mode: CompileMode = CompileMode.AUTO,
) -> Stats[T]:
    """Compute the distance stats using a dcov algorithm."""
    if exponent != 1:
        raise ValueError(f"Exponent should be 1 but is {exponent} instead.")

    xp = get_namespace(x, y)

    covariance_xy_sqr = dcov_function(x, y, compile_mode=compile_mode)
    variance_x_sqr = dcov_function(x, x, compile_mode=compile_mode)
    variance_y_sqr = dcov_function(y, y, compile_mode=compile_mode)
    denominator_sqr_signed = variance_x_sqr * variance_y_sqr
    denominator_sqr = xp.abs(denominator_sqr_signed)
    denominator = _sqrt(denominator_sqr)

    # Comparisons using a tolerance can change results if the
    # covariance has a similar order of magnitude
    if denominator == 0.0:
        correlation_xy_sqr = xp.zeros_like(covariance_xy_sqr)
    else:
        correlation_xy_sqr = covariance_xy_sqr / denominator

    return Stats(
        covariance_xy=covariance_xy_sqr,
        correlation_xy=correlation_xy_sqr,
        variance_x=variance_x_sqr,
        variance_y=variance_y_sqr,
    )


class DistanceCovarianceMethod(Enum):
    """
    Method used for computing the distance covariance.

    """
    AUTO = _DcovAlgorithmInternalsAuto()
    """
    Try to select the best algorithm. It will try to use a fast
    algorithm if possible. Otherwise it will use the naive
    implementation.
    """
    NAIVE = _DcovAlgorithmInternals(
        dcov_sqr=_distance_covariance_sqr_naive,
        u_dcov_sqr=_u_distance_covariance_sqr_naive,
        dcor_sqr=_distance_correlation_sqr_naive,
        u_dcor_sqr=_u_distance_correlation_sqr_naive,
        stats_generic=_distance_sqr_stats_naive_generic,
    )
    r"""
    Use the usual estimator of the distance covariance, which is
    :math:`O(n^2)`
    """
    AVL = _DcovAlgorithmInternals(
        dcov_generic=_distance_covariance_sqr_avl_generic,
    )
    r"""
    Use the fast implementation from
    :cite:`b-fast_distance_correlation_avl` which is
    :math:`O(n\log n)`
    """
    MERGESORT = _DcovAlgorithmInternals(
        dcov_generic=_distance_covariance_sqr_mergesort_generic,
    )
    r"""
    Use the fast implementation from
    :cite:`b-fast_distance_correlation_mergesort` which is
    :math:`O(n\log n)`
    """

    def __repr__(self) -> str:
        return '%s.%s' % (self.__class__.__name__, self.name)


DistanceCovarianceMethodLike = Union[DistanceCovarianceMethod, str]


def _to_algorithm(
    algorithm: DistanceCovarianceMethodLike,
) -> DistanceCovarianceMethod:
    """Convert to algorithm if string."""
    if isinstance(algorithm, DistanceCovarianceMethod):
        return algorithm

    return DistanceCovarianceMethod[algorithm.upper()]


def distance_covariance_sqr(
    x: T,
    y: T,
    *,
    exponent: float = 1,
    method: DistanceCovarianceMethodLike = DistanceCovarianceMethod.AUTO,
    compile_mode: CompileMode = CompileMode.AUTO,
) -> T:
    """
    Usual (biased) estimator for the squared distance covariance.

    Args:
        x: First random vector. The columns correspond with the individual
            random variables while the rows are individual instances of the
            random vector.
        y: Second random vector. The columns correspond with the individual
            random variables while the rows are individual instances of the
            random vector.
        exponent: Exponent of the Euclidean distance, in the range
            :math:`(0, 2)`. Equivalently, it is twice the Hurst parameter of
            fractional Brownian motion.
        method: Method to use internally to compute the distance covariance.
        compile_mode: Compilation mode used. By default it tries to use the
            fastest available type of compilation.

    Returns:
        Biased estimator of the squared distance covariance.

    See Also:
        distance_covariance
        u_distance_covariance_sqr

    Examples:
        >>> import numpy as np
        >>> import dcor
        >>> a = np.array([[1., 2., 3., 4.],
        ...               [5., 6., 7., 8.],
        ...               [9., 10., 11., 12.],
        ...               [13., 14., 15., 16.]])
        >>> b = np.array([[1.], [0.], [0.], [1.]])
        >>> dcor.distance_covariance_sqr(a, a)
        52.0
        >>> dcor.distance_covariance_sqr(a, b)
        1.0
        >>> dcor.distance_covariance_sqr(b, b)
        0.25
        >>> dcor.distance_covariance_sqr(a, b, exponent=0.5) # doctest: +ELLIPSIS
        0.3705904...

    """
    method = _to_algorithm(method)

    return method.value.dcov_sqr(
        x,
        y,
        exponent=exponent,
        compile_mode=compile_mode,
    )


def u_distance_covariance_sqr(
    x: T,
    y: T,
    *,
    exponent: float = 1,
    method: DistanceCovarianceMethodLike = DistanceCovarianceMethod.AUTO,
    compile_mode: CompileMode = CompileMode.AUTO,
) -> T:
    """
    Unbiased estimator for the squared distance covariance.

    Args:
        x: First random vector. The columns correspond with the individual
            random variables while the rows are individual instances of the
            random vector.
        y: Second random vector. The columns correspond with the individual
            random variables while the rows are individual instances of the
            random vector.
        exponent: Exponent of the Euclidean distance, in the range
            :math:`(0, 2)`. Equivalently, it is twice the Hurst parameter of
            fractional Brownian motion.
        method: Method to use internally to compute the distance covariance.
        compile_mode: Compilation mode used. By default it tries to use the
            fastest available type of compilation.

    Returns:
        Value of the unbiased estimator of the squared distance covariance.

    See Also:
        distance_covariance
        distance_covariance_sqr

    Examples:
        >>> import numpy as np
        >>> import dcor
        >>> a = np.array([[1., 2., 3., 4.],
        ...               [5., 6., 7., 8.],
        ...               [9., 10., 11., 12.],
        ...               [13., 14., 15., 16.]])
        >>> b = np.array([[1.], [0.], [0.], [1.]])
        >>> dcor.u_distance_covariance_sqr(a, a) # doctest: +ELLIPSIS
        42.6666666...
        >>> dcor.u_distance_covariance_sqr(a, b) # doctest: +ELLIPSIS
        -2.6666666...
        >>> dcor.u_distance_covariance_sqr(b, b) # doctest: +ELLIPSIS
        0.6666666...
        >>> dcor.u_distance_covariance_sqr(a, b, exponent=0.5) # doctest: +ELLIPSIS
        -0.2996598...

    """
    method = _to_algorithm(method)

    return method.value.u_dcov_sqr(
        x,
        y,
        exponent=exponent,
        compile_mode=compile_mode,
    )


def distance_covariance(
    x: T,
    y: T,
    *,
    exponent: float = 1,
    method: DistanceCovarianceMethodLike = DistanceCovarianceMethod.AUTO,
    compile_mode: CompileMode = CompileMode.AUTO,
) -> T:
    """
    Usual (biased) estimator for the distance covariance.

    Args:
        x: First random vector. The columns correspond with the individual
            random variables while the rows are individual instances of the
            random vector.
        y: Second random vector. The columns correspond with the individual
            random variables while the rows are individual instances of the
            random vector.
        exponent: Exponent of the Euclidean distance, in the range
            :math:`(0, 2)`. Equivalently, it is twice the Hurst parameter of
            fractional Brownian motion.
        method: Method to use internally to compute the distance covariance.
        compile_mode: Compilation mode used. By default it tries to use the
            fastest available type of compilation.

    Returns:
        Biased estimator of the distance covariance.

    See Also:
        distance_covariance_sqr
        u_distance_covariance_sqr

    Examples:
        >>> import numpy as np
        >>> import dcor
        >>> a = np.array([[1., 2., 3., 4.],
        ...               [5., 6., 7., 8.],
        ...               [9., 10., 11., 12.],
        ...               [13., 14., 15., 16.]])
        >>> b = np.array([[1.], [0.], [0.], [1.]])
        >>> dcor.distance_covariance(a, a) # doctest: +ELLIPSIS
        7.2111025...
        >>> dcor.distance_covariance(a, b)
        1.0
        >>> dcor.distance_covariance(b, b)
        0.5
        >>> dcor.distance_covariance(a, b, exponent=0.5)
        0.6087614...

    """
    return _sqrt(
        distance_covariance_sqr(
            x,
            y,
            exponent=exponent,
            method=method,
            compile_mode=compile_mode,
        ),
    )


def distance_stats_sqr(
    x: T,
    y: T,
    *,
    exponent: float = 1,
    method: DistanceCovarianceMethodLike = DistanceCovarianceMethod.AUTO,
    compile_mode: CompileMode = CompileMode.AUTO,
) -> Stats[T]:
    """
    Usual (biased) statistics related with the squared distance covariance.

    Computes the usual (biased) estimators for the squared distance covariance
    and squared distance correlation between two random vectors, and the
    individual squared distance variances.

    Args:
        x: First random vector. The columns correspond with the individual
            random variables while the rows are individual instances of the
            random vector.
        y: Second random vector. The columns correspond with the individual
            random variables while the rows are individual instances of the
            random vector.
        exponent: Exponent of the Euclidean distance, in the range
            :math:`(0, 2)`. Equivalently, it is twice the Hurst parameter of
            fractional Brownian motion.
        method: Method to use internally to compute the distance covariance.
        compile_mode: Compilation mode used. By default it tries to use the
            fastest available type of compilation.

    Returns:
        Stats object containing squared distance covariance,
        squared distance correlation,
        squared distance variance of the first random vector and
        squared distance variance of the second random vector.

    See Also:
        distance_covariance_sqr
        distance_correlation_sqr

    Notes:
        It is less efficient to compute the statistics separately, rather than
        using this function, because some computations can be shared.

    Examples:
        >>> import numpy as np
        >>> import dcor
        >>> a = np.array([[1., 2., 3., 4.],
        ...               [5., 6., 7., 8.],
        ...               [9., 10., 11., 12.],
        ...               [13., 14., 15., 16.]])
        >>> b = np.array([[1.], [0.], [0.], [1.]])
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
    method = _to_algorithm(method)

    return method.value.stats_sqr(
        x,
        y,
        exponent=exponent,
        compile_mode=compile_mode,
    )


def u_distance_stats_sqr(
    x: T,
    y: T,
    *,
    exponent: float = 1,
    method: DistanceCovarianceMethodLike = DistanceCovarianceMethod.AUTO,
    compile_mode: CompileMode = CompileMode.AUTO,
) -> T:
    """
    Unbiased statistics related with the squared distance covariance.

    Computes the unbiased estimators for the squared distance covariance
    and squared distance correlation between two random vectors, and the
    individual squared distance variances.

    Args:
        x: First random vector. The columns correspond with the individual
            random variables while the rows are individual instances of the
            random vector.
        y: Second random vector. The columns correspond with the individual
            random variables while the rows are individual instances of the
            random vector.
        exponent: Exponent of the Euclidean distance, in the range
            :math:`(0, 2)`. Equivalently, it is twice the Hurst parameter of
            fractional Brownian motion.
        method: Method to use internally to compute the distance covariance.
        compile_mode: Compilation mode used. By default it tries to use the
            fastest available type of compilation.

    Returns:
        Stats object containing squared distance covariance,
        squared distance correlation,
        squared distance variance of the first random vector and
        squared distance variance of the second random vector.

    See Also:
        u_distance_covariance_sqr
        u_distance_correlation_sqr

    Notes:
        It is less efficient to compute the statistics separately, rather than
        using this function, because some computations can be shared.

    Examples:
        >>> import numpy as np
        >>> import dcor
        >>> a = np.array([[1., 2., 3., 4.],
        ...               [5., 6., 7., 8.],
        ...               [9., 10., 11., 12.],
        ...               [13., 14., 15., 16.]])
        >>> b = np.array([[1.], [0.], [0.], [1.]])
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
    method = _to_algorithm(method)

    return method.value.u_stats_sqr(
        x,
        y,
        exponent=exponent,
        compile_mode=compile_mode,
    )


def distance_stats(
    x: T,
    y: T,
    *,
    exponent: float = 1,
    method: DistanceCovarianceMethodLike = DistanceCovarianceMethod.AUTO,
    compile_mode: CompileMode = CompileMode.AUTO,
) -> Stats[T]:
    """
    Usual (biased) statistics related with the distance covariance.

    Computes the usual (biased) estimators for the distance covariance
    and distance correlation between two random vectors, and the
    individual distance variances.

    Args:
        x: First random vector. The columns correspond with the individual
            random variables while the rows are individual instances of the
            random vector.
        y: Second random vector. The columns correspond with the individual
            random variables while the rows are individual instances of the
            random vector.
        exponent: Exponent of the Euclidean distance, in the range
            :math:`(0, 2)`. Equivalently, it is twice the Hurst parameter of
            fractional Brownian motion.
        method: Method to use internally to compute the distance covariance.
        compile_mode: Compilation mode used. By default it tries to use the
            fastest available type of compilation.

    Returns:
        Stats object containing distance covariance,
        distance correlation,
        distance variance of the first random vector and
        distance variance of the second random vector.

    See Also:
        distance_covariance
        distance_correlation

    Notes:
        It is less efficient to compute the statistics separately, rather than
        using this function, because some computations can be shared.

    Examples:
        >>> import numpy as np
        >>> import dcor
        >>> a = np.array([[1., 2., 3., 4.],
        ...               [5., 6., 7., 8.],
        ...               [9., 10., 11., 12.],
        ...               [13., 14., 15., 16.]])
        >>> b = np.array([[1.], [0.], [0.], [1.]])
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
    return Stats(
        *[
            _sqrt(s) for s in astuple(
                distance_stats_sqr(
                    x,
                    y,
                    exponent=exponent,
                    method=method,
                    compile_mode=compile_mode,
                ),
            )
        ],
    )


def distance_correlation_sqr(
    x: T,
    y: T,
    *,
    exponent: float = 1,
    method: DistanceCovarianceMethodLike = DistanceCovarianceMethod.AUTO,
    compile_mode: CompileMode = CompileMode.AUTO,
) -> T:
    """
    Usual (biased) estimator for the squared distance correlation.

    Args:
        x: First random vector. The columns correspond with the individual
            random variables while the rows are individual instances of the
            random vector.
        y: Second random vector. The columns correspond with the individual
            random variables while the rows are individual instances of the
            random vector.
        exponent: Exponent of the Euclidean distance, in the range
            :math:`(0, 2)`. Equivalently, it is twice the Hurst parameter of
            fractional Brownian motion.
        method: Method to use internally to compute the distance covariance.
        compile_mode: Compilation mode used. By default it tries to use the
            fastest available type of compilation.

    Returns:
        Value of the biased estimator of the squared distance correlation.

    See Also:
        distance_correlation
        u_distance_correlation_sqr

    Examples:
        >>> import numpy as np
        >>> import dcor
        >>> a = np.array([[1., 2., 3., 4.],
        ...               [5., 6., 7., 8.],
        ...               [9., 10., 11., 12.],
        ...               [13., 14., 15., 16.]])
        >>> b = np.array([[1.], [0.], [0.], [1.]])
        >>> dcor.distance_correlation_sqr(a, a)
        1.0
        >>> dcor.distance_correlation_sqr(a, b) # doctest: +ELLIPSIS
        0.2773500...
        >>> dcor.distance_correlation_sqr(b, b)
        1.0
        >>> dcor.distance_correlation_sqr(a, b, exponent=0.5) # doctest: +ELLIPSIS
        0.4493308...

    """
    method = _to_algorithm(method)

    return method.value.dcor_sqr(
        x,
        y,
        exponent=exponent,
        compile_mode=compile_mode,
    )


def u_distance_correlation_sqr(
    x: T,
    y: T,
    *,
    exponent: float = 1,
    method: DistanceCovarianceMethodLike = DistanceCovarianceMethod.AUTO,
    compile_mode: CompileMode = CompileMode.AUTO,
) -> T:
    """
    Bias-corrected estimator for the squared distance correlation.

    Args:
        x: First random vector. The columns correspond with the individual
            random variables while the rows are individual instances of the
            random vector.
        y: Second random vector. The columns correspond with the individual
            random variables while the rows are individual instances of the
            random vector.
        exponent: Exponent of the Euclidean distance, in the range
            :math:`(0, 2)`. Equivalently, it is twice the Hurst parameter of
            fractional Brownian motion.
        method: Method to use internally to compute the distance covariance.
        compile_mode: Compilation mode used. By default it tries to use the
            fastest available type of compilation.

    Returns:
        Value of the bias-corrected estimator of the squared distance
        correlation.

    See Also:
        distance_correlation
        distance_correlation_sqr

    Examples:
        >>> import numpy as np
        >>> import dcor
        >>> a = np.array([[1., 2., 3., 4.],
        ...               [5., 6., 7., 8.],
        ...               [9., 10., 11., 12.],
        ...               [13., 14., 15., 16.]])
        >>> b = np.array([[1.], [0.], [0.], [1.]])
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
    method = _to_algorithm(method)

    return method.value.u_dcor_sqr(
        x,
        y,
        exponent=exponent,
        compile_mode=compile_mode,
    )


def distance_correlation(
    x: T,
    y: T,
    *,
    exponent: float = 1,
    method: DistanceCovarianceMethodLike = DistanceCovarianceMethod.AUTO,
    compile_mode: CompileMode = CompileMode.AUTO,
) -> T:
    """
    Usual (biased) estimator for the distance correlation.

    Args:
        x: First random vector. The columns correspond with the individual
            random variables while the rows are individual instances of the
            random vector.
        y: Second random vector. The columns correspond with the individual
            random variables while the rows are individual instances of the
            random vector.
        exponent: Exponent of the Euclidean distance, in the range
            :math:`(0, 2)`. Equivalently, it is twice the Hurst parameter of
            fractional Brownian motion.
        method: Method to use internally to compute the distance covariance.
        compile_mode: Compilation mode used. By default it tries to use the
            fastest available type of compilation.

    Returns:
        Value of the biased estimator of the distance correlation.

    See Also:
        distance_correlation_sqr
        u_distance_correlation_sqr

    Examples:
        >>> import numpy as np
        >>> import dcor
        >>> a = np.array([[1., 2., 3., 4.],
        ...               [5., 6., 7., 8.],
        ...               [9., 10., 11., 12.],
        ...               [13., 14., 15., 16.]])
        >>> b = np.array([[1.], [0.], [0.], [1.]])
        >>> dcor.distance_correlation(a, a)
        1.0
        >>> dcor.distance_correlation(a, b) # doctest: +ELLIPSIS
        0.5266403...
        >>> dcor.distance_correlation(b, b)
        1.0
        >>> dcor.distance_correlation(a, b, exponent=0.5) # doctest: +ELLIPSIS
        0.6703214...

    """
    return _sqrt(
        distance_correlation_sqr(
            x,
            y,
            exponent=exponent,
            method=method,
            compile_mode=compile_mode,
        ),
    )


def distance_correlation_af_inv_sqr(
    x: T,
    y: T,
    method: DistanceCovarianceMethodLike = DistanceCovarianceMethod.AUTO,
    compile_mode: CompileMode = CompileMode.AUTO,
) -> T:
    """
    Square of the affinely invariant distance correlation.

    Computes the estimator for the square of the affinely invariant distance
    correlation between two random vectors.

    Warning:
        The return value of this function is undefined when the
        covariance matrix of :math:`x` or :math:`y` is singular.

    Args:
        x: First random vector. The columns correspond with the individual
            random variables while the rows are individual instances of the
            random vector.
        y: Second random vector. The columns correspond with the individual
            random variables while the rows are individual instances of the
            random vector.
        method: Method to use internally to compute the distance covariance.
        compile_mode: Compilation mode used. By default it tries to use the
            fastest available type of compilation.

    Returns:
        Value of the estimator of the squared affinely invariant
        distance correlation.

    See Also:
        distance_correlation
        u_distance_correlation

    Examples:
        >>> import numpy as np
        >>> import dcor
        >>> a = np.array([[1., 3., 2., 5.],
        ...               [5., 7., 6., 8.],
        ...               [9., 10., 11., 12.],
        ...               [13., 15., 15., 16.]])
        >>> b = np.array([[1.], [0.], [0.], [1.]])
        >>> dcor.distance_correlation_af_inv_sqr(a, a)
        1.0
        >>> dcor.distance_correlation_af_inv_sqr(a, b) # doctest: +ELLIPSIS
        0.5773502...
        >>> dcor.distance_correlation_af_inv_sqr(b, b)
        1.0

    """
    x = _af_inv_scaled(x)
    y = _af_inv_scaled(y)

    correlation = distance_correlation_sqr(
        x,
        y,
        method=method,
        compile_mode=compile_mode,
    )

    xp = get_namespace(x, y)

    return (
        xp.zeros_like(correlation)
        if xp.isnan(correlation)
        else correlation
    )


def distance_correlation_af_inv(
    x: T,
    y: T,
    method: DistanceCovarianceMethodLike = DistanceCovarianceMethod.AUTO,
    compile_mode: CompileMode = CompileMode.AUTO,
) -> T:
    """
    Affinely invariant distance correlation.

    Computes the estimator for the affinely invariant distance
    correlation between two random vectors.

    Warning:
        The return value of this function is undefined when the
        covariance matrix of :math:`x` or :math:`y` is singular.

    Args:
        x: First random vector. The columns correspond with the individual
            random variables while the rows are individual instances of the
            random vector.
        y: Second random vector. The columns correspond with the individual
            random variables while the rows are individual instances of the
            random vector.
        method: Method to use internally to compute the distance covariance.
        compile_mode: Compilation mode used. By default it tries to use the
            fastest available type of compilation.

    Returns:
        Value of the estimator of the squared affinely invariant
        distance correlation.

    See Also:
        distance_correlation
        u_distance_correlation

    Examples:
        >>> import numpy as np
        >>> import dcor
        >>> a = np.array([[1., 3., 2., 5.],
        ...               [5., 7., 6., 8.],
        ...               [9., 10., 11., 12.],
        ...               [13., 15., 15., 16.]])
        >>> b = np.array([[1.], [0.], [0.], [1.]])
        >>> dcor.distance_correlation_af_inv(a, a)
        1.0
        >>> dcor.distance_correlation_af_inv(a, b) # doctest: +ELLIPSIS
        0.7598356...
        >>> dcor.distance_correlation_af_inv(b, b)
        1.0

    """
    return _sqrt(
        distance_correlation_af_inv_sqr(
            x,
            y,
            method=method,
            compile_mode=compile_mode,
        ),
    )

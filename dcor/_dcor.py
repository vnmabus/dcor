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
    Generic,
    Iterator,
    Literal,
    Protocol,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import numpy as np

from dcor._dcor_internals import _af_inv_scaled

from ._dcor_internals import _dcov_from_terms, _dcov_terms_naive
from ._fast_dcov_avl import _distance_covariance_sqr_terms_avl
from ._fast_dcov_mergesort import _distance_covariance_sqr_terms_mergesort
from ._utils import (
    ArrayType,
    CompileMode,
    _sqrt,
    array_namespace,
    numpy_namespace,
)
##Additional module for Multivariate dcov test--------------------------------------------------------------
from scipy.special import gammaln
import math
from distances import dist_sum
from _rowise import rowwise
##-------------------------------------------------------------------------------------

Array = TypeVar("Array", bound=ArrayType)


@dataclass(frozen=True)
class Stats(Generic[Array]):
    """Distance covariance related stats."""
    covariance_xy: Array
    correlation_xy: Array
    variance_x: Array
    variance_y: Array

    def __iter__(self) -> Iterator[Array]:
        return iter(astuple(self))


class DCovFunction(Protocol):
    """Callback protocol for dcov method."""

    def __call__(
        self,
        __x: Array,
        __y: Array,
        *,
        compile_mode: CompileMode,
    ) -> Array:
        ...


class DCovTermsFunction(Protocol):
    """Callback protocol for dcov terms method."""

    @overload
    def __call__(
        self,
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
    def __call__(
        self,
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

    def __call__(
        self,
        __x: Array,
        __y: Array,
        *,
        exponent: float,
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
        ...


@overload
def _dcov_terms_auto(
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
def _dcov_terms_auto(
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


def _dcov_terms_auto(
    x: Array,
    y: Array,
    *,
    exponent: float,
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
    xp = array_namespace(x, y)

    dcov_terms = _dcov_terms_naive

    if xp == numpy_namespace and _can_use_fast_algorithm(x, y, exponent):
        dcov_terms = _distance_covariance_sqr_terms_avl

    return dcov_terms(
        x,
        y,
        exponent=exponent,
        compile_mode=compile_mode,
        return_var_terms=return_var_terms,
    )


class _DcovAlgorithmInternals():

    def __init__(
        self,
        *,
        terms: DCovTermsFunction | None = None,
    ):
        self.terms = terms

    def dcov_sqr(
        self,
        x: Array,
        y: Array,
        *,
        exponent: float = 1,
        compile_mode: CompileMode = CompileMode.AUTO,
        bias_corrected=False,
    ) -> Array:
        """Generic estimator for distance covariance."""
        terms = self.terms(
            x,
            y,
            exponent=exponent,
            compile_mode=compile_mode,
        )

        return _dcov_from_terms(
            *terms[:-2],
            n_samples=x.shape[0],
            bias_corrected=bias_corrected,
        )

    def stats_sqr(
        self,
        x: Array,
        y: Array,
        *,
        bias_corrected: bool = False,
        exponent: float = 1,
        compile_mode: CompileMode = CompileMode.AUTO,
    ) -> Stats[Array]:
        """Compute generic squared stats."""
        n_samples = x.shape[0]

        (
            mean_prod,
            a_axis_sum,
            a_total_sum,
            b_axis_sum,
            b_total_sum,
            a_mean_prod,
            b_mean_prod,
        ) = self.terms(
            x,
            y,
            exponent=exponent,
            compile_mode=compile_mode,
            return_var_terms=True,
        )

        covariance_xy_sqr = _dcov_from_terms(
            mean_prod=mean_prod,
            a_axis_sum=a_axis_sum,
            a_total_sum=a_total_sum,
            b_axis_sum=b_axis_sum,
            b_total_sum=b_total_sum,
            n_samples=n_samples,
            bias_corrected=bias_corrected,
        )
        variance_x_sqr = _dcov_from_terms(
            mean_prod=a_mean_prod,
            a_axis_sum=a_axis_sum,
            a_total_sum=a_total_sum,
            b_axis_sum=a_axis_sum,
            b_total_sum=a_total_sum,
            n_samples=n_samples,
            bias_corrected=bias_corrected,
        )
        variance_y_sqr = _dcov_from_terms(
            mean_prod=b_mean_prod,
            a_axis_sum=b_axis_sum,
            a_total_sum=b_total_sum,
            b_axis_sum=b_axis_sum,
            b_total_sum=b_total_sum,
            n_samples=n_samples,
            bias_corrected=bias_corrected,
        )

        xp = array_namespace(x, y)

        denominator_sqr = xp.abs(variance_x_sqr * variance_y_sqr)
        denominator = _sqrt(xp.asarray(denominator_sqr))

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


def _is_random_variable(x: Array) -> bool:
    """
    Check if the matrix x correspond to a random variable.

    The matrix is considered a random variable if it is a vector
    or a matrix corresponding to a column vector. Otherwise,
    the matrix correspond to a random vector.
    """
    return len(x.shape) == 1 or x.shape[1] == 1


def _can_use_fast_algorithm(x: Array, y: Array, exponent: float = 1) -> bool:
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
    x: Array,
    y: Array,
    *,
    exponent: float = 1,
    dcov_function: DCovFunction,
    compile_mode: CompileMode = CompileMode.AUTO,
) -> Stats[Array]:
    """Compute the distance stats using a dcov algorithm."""
    if exponent != 1:
        raise ValueError(f"Exponent should be 1 but is {exponent} instead.")

    xp = array_namespace(x, y)

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
    """Method used for computing the distance covariance."""

    AUTO = _DcovAlgorithmInternals(
        terms=_dcov_terms_auto,
    )
    """
    Try to select the best algorithm.

    It will try to use a fast algorithm if possible.
    Otherwise it will use the naive implementation.
    """

    NAIVE = _DcovAlgorithmInternals(
        terms=_dcov_terms_naive,
    )
    r"""Usual estimator of the distance covariance, which is :math:`O(n^2)`"""

    AVL = _DcovAlgorithmInternals(
        terms=_distance_covariance_sqr_terms_avl,
    )
    r"""
    Use the AVL fast implementation.

    This is the implementation described in
    :cite:`b-fast_distance_correlation_avl` which is
    :math:`O(n\log n)`
    """
    MERGESORT = _DcovAlgorithmInternals(
        terms=_distance_covariance_sqr_terms_mergesort,
    )
    r"""
    Use the mergesort fast implementation.

    This is the implementation described in
    :cite:`b-fast_distance_correlation_mergesort` which is
    :math:`O(n\log n)`
    """

    def __repr__(self) -> str:
        return '%s.%s' % (self.__class__.__name__, self.name)


_DistanceCovarianceMethodName = Literal["auto", "naive", "avl", "mergesort"]
DistanceCovarianceMethodLike = Union[
    DistanceCovarianceMethod,
    _DistanceCovarianceMethodName,
]


def _to_algorithm(
    algorithm: DistanceCovarianceMethodLike,
) -> DistanceCovarianceMethod:
    """Convert to algorithm if string."""
    if isinstance(algorithm, DistanceCovarianceMethod):
        return algorithm

    return DistanceCovarianceMethod[algorithm.upper()]


def distance_covariance_sqr(
    x: Array,
    y: Array,
    *,
    exponent: float = 1,
    method: DistanceCovarianceMethodLike = DistanceCovarianceMethod.AUTO,
    compile_mode: CompileMode = CompileMode.AUTO,
) -> Array:
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
    x: Array,
    y: Array,
    *,
    exponent: float = 1,
    method: DistanceCovarianceMethodLike = DistanceCovarianceMethod.AUTO,
    compile_mode: CompileMode = CompileMode.AUTO,
) -> Array:
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

    return method.value.dcov_sqr(
        x,
        y,
        exponent=exponent,
        compile_mode=compile_mode,
        bias_corrected=True,
    )


def distance_covariance(
    x: Array,
    y: Array,
    *,
    exponent: float = 1,
    method: DistanceCovarianceMethodLike = DistanceCovarianceMethod.AUTO,
    compile_mode: CompileMode = CompileMode.AUTO,
) -> Array:
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
    x: Array,
    y: Array,
    *,
    exponent: float = 1,
    method: DistanceCovarianceMethodLike = DistanceCovarianceMethod.AUTO,
    compile_mode: CompileMode = CompileMode.AUTO,
) -> Stats[Array]:
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
    x: Array,
    y: Array,
    *,
    exponent: float = 1,
    method: DistanceCovarianceMethodLike = DistanceCovarianceMethod.AUTO,
    compile_mode: CompileMode = CompileMode.AUTO,
) -> Stats[Array]:
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
        Stats(covariance_xy=-2.6666666..., correlation_xy=-0.4999999...,
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

    return method.value.stats_sqr(
        x,
        y,
        exponent=exponent,
        compile_mode=compile_mode,
        bias_corrected=True,
    )


def distance_stats(
    x: Array,
    y: Array,
    *,
    exponent: float = 1,
    method: DistanceCovarianceMethodLike = DistanceCovarianceMethod.AUTO,
    compile_mode: CompileMode = CompileMode.AUTO,
) -> Stats[Array]:
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
    x: Array,
    y: Array,
    *,
    exponent: float = 1,
    method: DistanceCovarianceMethodLike = DistanceCovarianceMethod.AUTO,
    compile_mode: CompileMode = CompileMode.AUTO,
) -> Array:
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

    return method.value.stats_sqr(
        x,
        y,
        exponent=exponent,
        compile_mode=compile_mode,
    ).correlation_xy


def u_distance_correlation_sqr(
    x: Array,
    y: Array,
    *,
    exponent: float = 1,
    method: DistanceCovarianceMethodLike = DistanceCovarianceMethod.AUTO,
    compile_mode: CompileMode = CompileMode.AUTO,
) -> Array:
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
        -0.4999999...
        >>> dcor.u_distance_correlation_sqr(b, b)
        1.0
        >>> dcor.u_distance_correlation_sqr(a, b, exponent=0.5)
        ... # doctest: +ELLIPSIS
        -0.4050479...

    """
    method = _to_algorithm(method)

    return method.value.stats_sqr(
        x,
        y,
        exponent=exponent,
        compile_mode=compile_mode,
        bias_corrected=True,
    ).correlation_xy


def distance_correlation(
    x: Array,
    y: Array,
    *,
    exponent: float = 1,
    method: DistanceCovarianceMethodLike = DistanceCovarianceMethod.AUTO,
    compile_mode: CompileMode = CompileMode.AUTO,
) -> Array:
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
    x: Array,
    y: Array,
    method: DistanceCovarianceMethodLike = DistanceCovarianceMethod.AUTO,
    compile_mode: CompileMode = CompileMode.AUTO,
) -> Array:
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

    xp = array_namespace(x, y)

    return (
        xp.zeros_like(correlation)
        if xp.isnan(correlation)
        else correlation
    )


def distance_correlation_af_inv(
    x: Array,
    y: Array,
    method: DistanceCovarianceMethodLike = DistanceCovarianceMethod.AUTO,
    compile_mode: CompileMode = CompileMode.AUTO,
) -> Array:
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





"""
A Statistically and Numerically Efficient Independence Test Based on Random Projections and Distance Covariance

:cite:`b-dcov_random_projection`.

References
----------
.. bibliography:: ../refs.bib
   :labelprefix: B
   :keyprefix: b-
"""


def gamma_ratio(p): return np.exp(gammaln((p+1)/2) - gammaln(p/2))  # For Calculating C_p and C_q



def rndm_projection(X, p):
    """
    Parameters
    ----------
    X : N x p, array of arrays
    where, p: number of dimensions (p >= 1) and N: number of samples 
    p : number of dimensions (p >= 1)      

    Returns
    -------
    X_new : an array of size N
    DESCRIPTION: Random projection of multivariate array
    """

    # X_std = multivariate_normal.rvs( np.zeros(p), np.identity(p), size = 1)
    X_std = np.random.standard_normal(p)
            
    X_norm = np.linalg.norm(X_std)
    U_sphere = np.array(X_std)/X_norm  # Normalize X_std
    
    if p > 1:
        X_new = U_sphere @ X.T
    else:
        X_new = U_sphere * X
    return X_new


def u_dist_cov_sqr_mv(X, Y, n_projs=800, method='mergesort'):
    """
    Parameters
    ----------
    X : N x p, array of arrays, where p > 1
    Y : N x q, array of arrays, where q >= 1
    where p and q: number of dimensions of variable X and Y, respectively and N: number of samples

    n_projs : Number of projections (integer type), optional
        DESCRIPTION. The default is 500.(paper suggested to consider: n_projs < N/logN, larger n_projs provides better results)
    method : fast computation method either 'mergesort' or 'avl', optional
        DESCRIPTION. The default is 'mergesort'.

    Returns
    -------
    omega_bar : Float type
        DESCRIPTION: Produce fastly computed unbiased distance covariance between X and Y

    
    
    
    Examples:
        >>> import numpy as np
        >>> import dcor
        >>> from scipy.stats import multivariate_normal
        >>> mean_vector = [2, 3, 5, 3, 2, 1]
        >>> matrix_size = 6
        >>> A = 0.5*np.random.rand(matrix_size, matrix_size)
        >>> B = np.dot(A, A.transpose())
        >>> n_samples = 3000
        >>> X = multivariate_normal.rvs(mean_vector, B, size=n_samples)
        >>> X1 = X.T[:4]
        >>> X2 = X.T[4:] 
        >>> print(f"Computing fast distance covariance = {u_dist_cov_sqr_mv(X1.T, X2.T)}")
    """

    n_samples = np.shape(X)[0]
    p = np.shape(X)[1]
    q = np.shape(Y)[1]
    
    sqrt_pi_value = math.sqrt(math.pi)
    C_p = sqrt_pi_value*gamma_ratio(p)
    C_q = sqrt_pi_value*gamma_ratio(q)

    
    X_proj = np.empty(( n_projs, n_samples))    
    Y_proj = np.empty(( n_projs, n_samples))
    
    for i in range(n_projs):
        X_proj[i, :] = rndm_projection(X, p)
        Y_proj[i, :] = rndm_projection(Y, q)    
        pass
          
    omega_ = rowwise(u_distance_covariance_sqr, X_proj, Y_proj, rowwise_mode= method)
    omega_bar = C_p*C_q*np.mean(omega_)
    
    return omega_bar

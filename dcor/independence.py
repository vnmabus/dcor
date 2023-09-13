"""
Functions for testing independence of several distributions.

The functions in this module provide methods for testing if
the samples generated from two random vectors are independent.
"""
from __future__ import annotations

from typing import TypeVar

import numpy as np
import scipy.stats

from ._dcor import u_distance_correlation_sqr
from ._dcor_internals import (
    _check_same_n_elements,
    _distance_matrix_generic,
    _u_distance_matrix,
    double_centered,
    mean_product,
    u_complementary_projection,
    u_product,
)
from ._hypothesis import HypothesisTest, _permutation_test_with_sym_matrix
from ._utils import (
    ArrayType,
    RandomLike,
    _random_state_init,
    _sqrt,
    _transform_to_2d,
)

Array = TypeVar("Array", bound=ArrayType)


def distance_covariance_test(
    x: Array,
    y: Array,
    *,
    num_resamples: int = 0,
    exponent: float = 1,
    random_state: RandomLike = None,
    n_jobs: int = 1,
) -> HypothesisTest[Array]:
    """
    Test of distance covariance independence.

    Compute the test of independence based on the distance
    covariance, for two random vectors.

    The test is a permutation test where the null hypothesis is that the two
    random vectors are independent.

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
        num_resamples: Number of permutations resamples to take in the
            permutation test.
        random_state: Random state to generate the permutations.
        n_jobs: Number of jobs executed in parallel by Joblib.

    Returns:
        Results of the hypothesis test.

    See Also:
        distance_covariance

    Examples:
        >>> import numpy as np
        >>> import dcor
        >>> a = np.array([[1, 2, 3, 4],
        ...               [5, 6, 7, 8],
        ...               [9, 10, 11, 12],
        ...               [13, 14, 15, 16]])
        >>> b = np.array([[1, 0, 0, 1],
        ...               [0, 1, 1, 1],
        ...               [1, 1, 1, 1],
        ...               [1, 1, 0, 1]])
        >>> dcor.independence.distance_covariance_test(a, a)
        HypothesisTest(pvalue=1.0, statistic=208.0)
        >>> dcor.independence.distance_covariance_test(a, b)
        ...                                      # doctest: +ELLIPSIS
        HypothesisTest(pvalue=1.0, statistic=11.75323056...)
        >>> dcor.independence.distance_covariance_test(b, b)
        HypothesisTest(pvalue=1.0, statistic=1.3604610...)
        >>> dcor.independence.distance_covariance_test(a, b,
        ... num_resamples=5, random_state=0)
        HypothesisTest(pvalue=0.8333333333333334, statistic=11.7532305...)
        >>> dcor.independence.distance_covariance_test(a, b,
        ... num_resamples=5, random_state=13)
        HypothesisTest(pvalue=0.5..., statistic=11.7532305...)
        >>> dcor.independence.distance_covariance_test(a, a,
        ... num_resamples=7, random_state=0)
        HypothesisTest(pvalue=0.125, statistic=208.0)

    """
    x, y = _transform_to_2d(x, y)

    _check_same_n_elements(x, y)

    random_state = _random_state_init(random_state)

    # Compute U-centered matrices
    u_x = _distance_matrix_generic(
        x,
        centering=double_centered,
        exponent=exponent,
    )
    u_y = _distance_matrix_generic(
        y,
        centering=double_centered,
        exponent=exponent,
    )

    # Use the dcov statistic
    def statistic_function(distance_matrix: Array) -> Array:
        return u_x.shape[0] * mean_product(
            distance_matrix,
            u_y,
        )

    return _permutation_test_with_sym_matrix(
        u_x,
        statistic_function=statistic_function,
        num_resamples=num_resamples,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def partial_distance_covariance_test(
    x: Array,
    y: Array,
    z: Array,
    *,
    num_resamples: int = 0,
    exponent: float = 1,
    random_state: RandomLike = None,
    n_jobs: int | None = 1,
) -> HypothesisTest[Array]:
    """
    Test of partial distance covariance independence.

    Compute the test of independence based on the partial distance
    covariance, for two random vectors conditioned on a third.

    The test is a permutation test where the null hypothesis is that the first
    two random vectors are independent given the third one.

    Args:
        x: First random vector. The columns correspond with the individual
            random variables while the rows are individual instances of the
            random vector.
        y: Second random vector. The columns correspond with the individual
            random variables while the rows are individual instances of the
            random vector.
        z: Observed random vector. The columns correspond with the individual
            random variables while the rows are individual instances of the
            random vector.
        exponent: Exponent of the Euclidean distance, in the range
            :math:`(0, 2)`. Equivalently, it is twice the Hurst parameter of
            fractional Brownian motion.
        num_resamples: Number of permutations resamples to take in the
            permutation test.
        random_state: Random state to generate the permutations.
        n_jobs: Number of jobs executed in parallel by Joblib.

    Returns:
        Results of the hypothesis test.

    See Also:
        partial_distance_covariance

    Examples:
        >>> import numpy as np
        >>> import dcor
        >>> a = np.array([[1, 2, 3, 4],
        ...               [5, 6, 7, 8],
        ...               [9, 10, 11, 12],
        ...               [13, 14, 15, 16]])
        >>> b = np.array([[1, 0, 0, 1],
        ...               [0, 1, 1, 1],
        ...               [1, 1, 1, 1],
        ...               [1, 1, 0, 1]])
        >>> c = np.array([[1000, 0, 0, 1000],
        ...               [0, 1000, 1000, 1000],
        ...               [1000, 1000, 1000, 1000],
        ...               [1000, 1000, 0, 1000]])
        >>> dcor.independence.partial_distance_covariance_test(a, a, b)
        ...                                       # doctest: +ELLIPSIS
        HypothesisTest(pvalue=1.0, statistic=142.6664416...)
        >>> test = dcor.independence.partial_distance_covariance_test(a, b, c)
        >>> test.pvalue # doctest: +ELLIPSIS
        1.0
        >>> np.allclose(test.statistic, 0, atol=1e-6)
        True
        >>> test = dcor.independence.partial_distance_covariance_test(a, b, c,
        ... num_resamples=5, random_state=0)
        >>> test.pvalue # doctest: +ELLIPSIS
        0.1666666...
        >>> np.allclose(test.statistic, 0, atol=1e-6)
        True

    """
    random_state = _random_state_init(random_state)

    # Compute U-centered matrices
    u_x = _u_distance_matrix(x, exponent=exponent)
    u_y = _u_distance_matrix(y, exponent=exponent)
    u_z = _u_distance_matrix(z, exponent=exponent)

    # Compute projections
    proj = u_complementary_projection(u_z)

    p_xz = proj(u_x)
    p_yz = proj(u_y)

    # Use the pdcor statistic
    def statistic_function(distance_matrix: Array) -> Array:
        return u_x.shape[0] * u_product(
            distance_matrix,
            p_yz,
        )

    return _permutation_test_with_sym_matrix(
        p_xz,
        statistic_function=statistic_function,
        num_resamples=num_resamples,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def distance_correlation_t_statistic(
    x: Array,
    y: Array,
) -> Array:
    """
    Statistic used in :func:`distance_correlation_t_test`.

    Args:
        x: First random vector. The columns correspond with the individual
            random variables while the rows are individual instances of the
            random vector.
        y: Second random vector. The columns correspond with the individual
            random variables while the rows are individual instances of the
            random vector.

    Returns:
        T statistic.

    See Also:
        distance_correlation_t_test

    Examples:
        >>> import numpy as np
        >>> import dcor
        >>> a = np.array([[1, 2, 3, 4],
        ...               [5, 6, 7, 8],
        ...               [9, 10, 11, 12],
        ...               [13, 14, 15, 16]])
        >>> b = np.array([[1, 0, 0, 1],
        ...               [0, 1, 1, 1],
        ...               [1, 1, 1, 1],
        ...               [1, 1, 0, 1]])
        >>> with np.errstate(divide='ignore'):
        ...     dcor.independence.distance_correlation_t_statistic(a, a)
        inf
        >>> dcor.independence.distance_correlation_t_statistic(a, b)
        ...                                      # doctest: +ELLIPSIS
        -0.4430164...
        >>> with np.errstate(divide='ignore'):
        ...     dcor.independence.distance_correlation_t_statistic(b, b)
        inf

    """
    bcdcor = u_distance_correlation_sqr(x, y)

    n = x.shape[0]
    v = n * (n - 3) / 2

    return np.sqrt(v - 1) * bcdcor / _sqrt(1 - bcdcor**2)


def distance_correlation_t_test(
    x: Array,
    y: Array,
) -> HypothesisTest[Array]:
    """
    Test of independence for high dimension.

    It is based on convergence to a Student t distribution.
    The null hypothesis is that the two random vectors are
    independent.

    Args:
        x: First random vector. The columns correspond with the individual
            random variables while the rows are individual instances of the
            random vector.
        y: Second random vector. The columns correspond with the individual
            random variables while the rows are individual instances of the
            random vector.

    Returns:
        Results of the hypothesis test.

    See Also:
        distance_correlation_t_statistic

    Examples:
        >>> import numpy as np
        >>> import dcor
        >>> a = np.array([[1, 2, 3, 4],
        ...               [5, 6, 7, 8],
        ...               [9, 10, 11, 12],
        ...               [13, 14, 15, 16]])
        >>> b = np.array([[1, 0, 0, 1],
        ...               [0, 1, 1, 1],
        ...               [1, 1, 1, 1],
        ...               [1, 1, 0, 1]])
        >>> with np.errstate(divide='ignore'):
        ...     dcor.independence.distance_correlation_t_test(a, a)
        ...                                      # doctest: +ELLIPSIS
        HypothesisTest(pvalue=0.0, statistic=inf)
        >>> dcor.independence.distance_correlation_t_test(a, b)
        ...                                      # doctest: +ELLIPSIS
        HypothesisTest(pvalue=0.6327451..., statistic=-0.4430164...)
        >>> with np.errstate(divide='ignore'):
        ...     dcor.independence.distance_correlation_t_test(b, b)
        ...                                      # doctest: +ELLIPSIS
        HypothesisTest(pvalue=0.0, statistic=inf)

    """
    t_test = distance_correlation_t_statistic(x, y)

    n = x.shape[0]
    v = n * (n - 3) / 2
    df = v - 1

    p_value = 1 - scipy.stats.t.cdf(t_test, df=df)

    return HypothesisTest(pvalue=p_value, statistic=t_test)

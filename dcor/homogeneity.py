"""
Functions for testing homogeneity of several distributions.

The functions in this module provide methods for testing if
the samples generated from two random vectors have the same
distribution.
"""
from __future__ import annotations

from typing import Callable, Sequence, TypeVar

import numpy as np

from . import distances as _distances
from ._energy import (
    EstimationStatistic,
    EstimationStatisticLike,
    _check_valid_energy_exponent,
    _energy_distance_from_distance_matrices,
    energy_distance,
)
from ._hypothesis import HypothesisTest, _permutation_test_with_sym_matrix
from ._utils import ArrayType, RandomLike, _transform_to_2d, array_namespace

Array = TypeVar("Array", bound=ArrayType)


def _energy_test_statistic_coefficient(
    n: int,
    m: int,
) -> float:
    """Coefficient of the test statistic."""
    return n * m / (n + m)


def _energy_test_statistic_from_distance_matrices(
    distance_xx: Array,
    distance_yy: Array,
    distance_xy: Array,
    n: int,
    m: int,
    average: Callable[[Array], Array] | None = None,
    estimation_stat: EstimationStatisticLike = EstimationStatistic.V_STATISTIC,
) -> Array:
    """Test statistic with precomputed distance matrices."""
    energy_distance = _energy_distance_from_distance_matrices(
        distance_xx=distance_xx,
        distance_yy=distance_yy,
        distance_xy=distance_xy,
        average=average,
        estimation_stat=estimation_stat,
    )

    return _energy_test_statistic_coefficient(n, m) * energy_distance


def energy_test_statistic(
    x: Array,
    y: Array,
    *,
    exponent: float = 1,
    average: Callable[[Array], Array] | None = None,
    estimation_stat: EstimationStatisticLike = EstimationStatistic.V_STATISTIC,
) -> Array:
    """
    Homogeneity statistic.

    Computes the statistic for homogeneity based on the energy distance, for
    random vectors corresponding to :math:`x` and :math:`y`.

    Args:
        x: First random vector. The columns correspond with the individual
            random variables while the rows are individual instances of the
            random vector.
        y: Second random vector. The columns correspond with the individual
            random variables while the rows are individual instances of the
            random vector.
        exponent: Exponent of the Euclidean distance, in the range
            :math:`(0, 2)`.
        average: A function that will be used to calculate an average of
            distances. This defaults to np.mean.
        estimation_stat: If EstimationStatistic.U_STATISTIC, calculate energy
            distance using Hoeffding's unbiased U-statistics. Otherwise, use
            von Mises's biased V-statistics.
            If this is provided as a string, it will first be converted to
            an EstimationStatistic enum instance.

    Returns:
        Value of the statistic for homogeneity based on the energy distance.

    See Also:
        energy_distance

    Examples:
        >>> import numpy as np
        >>> import dcor
        >>> a = np.array([[1, 2, 3, 4],
        ...               [5, 6, 7, 8],
        ...               [9, 10, 11, 12],
        ...               [13, 14, 15, 16]])
        >>> b = np.array([[1, 0, 0, 1],
        ...               [0, 1, 1, 1],
        ...               [1, 1, 1, 1]])
        >>> dcor.homogeneity.energy_test_statistic(a, a)
        0.0
        >>> dcor.homogeneity.energy_test_statistic(a, b) # doctest: +ELLIPSIS
        35.2766732...
        >>> dcor.homogeneity.energy_test_statistic(b, b)
        0.0

    """
    x, y = _transform_to_2d(x, y)

    n = x.shape[0]
    m = y.shape[0]

    coefficient = _energy_test_statistic_coefficient(n, m)

    return coefficient * energy_distance(
        x,
        y,
        exponent=exponent,
        average=average,
        estimation_stat=estimation_stat,
    )


def _energy_test_statistic_multivariate_from_distance_matrix(
    distance: Array,
    indexes: Sequence[int],
    sizes: Sequence[int],
    average: Callable[[Array], Array] | None = None,
    estimation_stat: EstimationStatisticLike = EstimationStatistic.V_STATISTIC,
) -> Array:
    """Statistic for several random vectors given the distance matrix."""
    first_iter = True

    for i, _ in enumerate(indexes):
        for j in range(i + 1, len(indexes)):
            slice_i = slice(indexes[i], indexes[i] + sizes[i])
            slice_j = slice(indexes[j], indexes[j] + sizes[j])

            n = sizes[i]
            m = sizes[j]

            distance_xx = distance[slice_i, slice_i]
            distance_yy = distance[slice_j, slice_j]
            distance_xy = distance[slice_i, slice_j]

            pairwise_energy = _energy_test_statistic_from_distance_matrices(
                distance_xx=distance_xx,
                distance_yy=distance_yy,
                distance_xy=distance_xy,
                n=n,
                m=m,
                average=average,
                estimation_stat=estimation_stat,
            )

            if first_iter:
                energy = pairwise_energy
                first_iter = False
            else:
                energy += pairwise_energy

    return energy


def energy_test(
    *args: T,
    num_resamples: int = 0,
    exponent: float = 1,
    random_state: RandomLike = None,
    average: Callable[[Array], Array] | None = None,
    estimation_stat: EstimationStatisticLike = EstimationStatistic.V_STATISTIC,
    n_jobs: int | None = 1,
) -> HypothesisTest[Array]:
    """
    Test of homogeneity based on the energy distance.

    Compute the test of homogeneity based on the energy distance, for
    an arbitrary number of random vectors.

    The test is a permutation test where the null hypothesis is that all
    random vectors have the same distribution.

    Args:
        args: Random vectors. The columns correspond with the individual random
            variables while the rows are individual instances of the random
            vector.
        num_resamples: Number of permutations resamples to take in the
            permutation test.
        exponent: Exponent of the Euclidean distance, in the range
            :math:`(0, 2)`.
        random_state: Random state to generate the permutations.
        average: A function that will be used to calculate an average of
            distances. This defaults to np.mean.
        estimation_stat: If EstimationStatistic.U_STATISTIC, calculate energy
            distance using Hoeffding's unbiased U-statistics. Otherwise, use
            von Mises's biased V-statistics. If this is provided as a string,
            it will first be converted to an EstimationStatistic enum instance.
        n_jobs: Number of jobs executed in parallel by Joblib.

    Returns:
        Results of the hypothesis test.

    See Also:
        energy_distance

    Examples:
        >>> import numpy as np
        >>> import dcor
        >>> a = np.array([[1, 2, 3, 4],
        ...               [5, 6, 7, 8],
        ...               [9, 10, 11, 12],
        ...               [13, 14, 15, 16]])
        >>> b = np.array([[1, 0, 0, 1],
        ...               [0, 1, 1, 1],
        ...               [1, 1, 1, 1]])
        >>> c = np.array([[1000, 0, 0, 1000],
        ...               [0, 1000, 1000, 1000],
        ...               [1000, 1000, 1000, 1000]])
        >>> dcor.homogeneity.energy_test(a, a)
        HypothesisTest(pvalue=1.0, statistic=0.0)
        >>> dcor.homogeneity.energy_test(a, b) # doctest: +ELLIPSIS
        HypothesisTest(pvalue=1.0, statistic=35.2766732...)
        >>> dcor.homogeneity.energy_test(b, b)
        HypothesisTest(pvalue=1.0, statistic=0.0)
        >>> dcor.homogeneity.energy_test(a, b, num_resamples=5, random_state=0)
        HypothesisTest(pvalue=0.1666666..., statistic=35.2766732...)
        >>> dcor.homogeneity.energy_test(a, b, num_resamples=5, random_state=6)
        HypothesisTest(pvalue=0.3333333..., statistic=35.2766732...)
        >>> dcor.homogeneity.energy_test(a, c, num_resamples=7, random_state=0)
        HypothesisTest(pvalue=0.125, statistic=4233.8935035...)

        A different exponent for the Euclidean distance in the range
        :math:`(0, 2)` can be used:

        >>> dcor.homogeneity.energy_test(a, b, exponent=1.5)
        ...                                               # doctest: +ELLIPSIS
        HypothesisTest(pvalue=1.0, statistic=171.0623923...)

    """
    samples = list(_transform_to_2d(*args))

    num_samples = len(samples)

    _check_valid_energy_exponent(exponent)

    sample_sizes = tuple(a.shape[0] for a in samples)

    xp = array_namespace(*samples)

    # NumPy namespace has no concat function yet
    try:
        concat = xp.concat
    except AttributeError:
        concat = np.concatenate
    pooled_samples = concat(samples)

    sample_indexes_array = np.zeros(num_samples, dtype=int)
    sample_indexes_array[1:] = np.cumsum(sample_sizes)[:-1]
    sample_indexes = tuple(sample_indexes_array)

    # Compute the distance matrix once
    sample_distances = _distances.pairwise_distances(
        pooled_samples,
        exponent=exponent,
    )

    # Use the energy statistic with appropiate values
    def statistic_function(distance_matrix: Array) -> Array:
        return _energy_test_statistic_multivariate_from_distance_matrix(
            distance=distance_matrix,
            indexes=sample_indexes,
            sizes=sample_sizes,
            average=average,
            estimation_stat=estimation_stat,
        )

    return _permutation_test_with_sym_matrix(
        sample_distances,
        statistic_function=statistic_function,
        num_resamples=num_resamples,
        random_state=random_state,
        n_jobs=n_jobs,
    )

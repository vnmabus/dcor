"""
Functions for testing homogeneity of several distributions.

The functions in this module provide methods for testing if
the samples generated from two random vectors have the same
distribution.
"""

import numpy as _np

from numba import njit

from . import _energy, _hypothesis
from . import distances as _distances
from ._utils import _transform_to_2d


@njit()
def _energy_test_statistic_coefficient(n, m):
    """Coefficient of the test statistic."""
    return n * m / (n + m)


@njit()
def _energy_test_statistic_from_distance_matrices(
        distance_xx, distance_yy, distance_xy, n, m, average='mean',
        stat_type=_energy.EstimationStatistic.V_STATISTIC):
    """Test statistic with precomputed distance matrices."""
    energy_distance = _energy._energy_distance_from_distance_matrices(
        distance_xx=distance_xx, distance_yy=distance_yy,
        distance_xy=distance_xy, average=average, stat_type=stat_type
    )

    return _energy_test_statistic_coefficient(n, m) * energy_distance


def energy_test_statistic(x, y, *, exponent=1, average='mean', stat_type='v'):
    """
    Homogeneity statistic.

    Computes the statistic for homogeneity based on the energy distance, for
    random vectors corresponding to :math:`x` and :math:`y`.

    Parameters
    ----------
    x: array_like
        First random vector. The columns correspond with the individual random
        variables while the rows are individual instances of the random vector.
    y: array_like
        Second random vector. The columns correspond with the individual random
        variables while the rows are individual instances of the random vector.
    exponent: float
        Exponent of the Euclidean distance, in the range :math:`(0, 2)`.
    average: str
        Specify the type of average used to calculate an average of distances.
        Either "mean" or "median". Defaults to "mean"
    stat_type: Union[str, EstimationStatistic]
        If EstimationStatistic.U_STATISTIC, calculate energy distance using
        Hoeffding's unbiased U-statistics. Otherwise, use von Mises's biased
        V-statistics.
        If this is provided as a string, it will first be converted to
        an EstimationStatistic enum instance.

    Returns
    -------
    numpy scalar
        Value of the statistic for homogeneity based on the energy distance.

    See Also
    --------
    energy_distance

    Examples
    --------
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
    if isinstance(stat_type, str):
        stat_type = _energy.EstimationStatistic.from_string(stat_type)

    x = _transform_to_2d(x)
    y = _transform_to_2d(y)

    n = x.shape[0]
    m = y.shape[0]

    coefficient = _energy_test_statistic_coefficient(n, m)

    return coefficient * _energy.energy_distance(
        x,
        y,
        exponent=exponent,
        average=average,
        stat_type=stat_type
    )


@njit()
def _energy_test_statistic_multivariate_from_distance_matrix(
        distance, indexes, sizes, average='mean', stat_type='v'):
    """Statistic for several random vectors given the distance matrix."""
    energy = 0.0

    for i, _ in enumerate(indexes):
        for j in range(i + 1, len(indexes)):
            n = sizes[i]
            m = sizes[j]

            distance_xx = distance[
                indexes[i]:indexes[i] + n,
                indexes[i]:indexes[i] + n
            ]
            distance_yy = distance[
                indexes[j]:indexes[j] + m,
                indexes[j]:indexes[j] + m
            ]
            distance_xy = distance[
                indexes[i]:indexes[i] + n,
                indexes[j]:indexes[j] + m
            ]

            pairwise_energy = _energy_test_statistic_from_distance_matrices(
                distance_xx=distance_xx, distance_yy=distance_yy,
                distance_xy=distance_xy, n=n, m=m, average=average,
                stat_type=stat_type
                )

            energy += pairwise_energy

    return energy


def energy_test(
    *args,
    num_resamples=0,
    exponent=1,
    average='mean',
    stat_type=_energy.EstimationStatistic.V_STATISTIC
):
    """
    Test of homogeneity based on the energy distance.

    Compute the test of homogeneity based on the energy distance, for
    an arbitrary number of random vectors.

    The test is a permutation test where the null hypothesis is that all
    random vectors have the same distribution.

    Parameters
    ----------
    *args: array_like
        Random vectors. The columns correspond with the individual random
        variables while the rows are individual instances of the random vector.
    num_resamples: int
        Number of permutations resamples to take in the permutation test.
    exponent: float
        Exponent of the Euclidean distance, in the range :math:`(0, 2)`.
    average: str
        Specify the type of average used to calculate an average of distances.
        Either "mean" or "median". Defaults to "mean"
    stat_type: Union[str, EstimationStatistic]
        If EstimationStatistic.U_STATISTIC, calculate energy distance using
        Hoeffding's unbiased U-statistics. Otherwise, use von Mises's biased
        V-statistics.
        If this is provided as a string, it will first be converted to
        an EstimationStatistic enum instance.

    Returns
    -------
    HypothesisTest
        Results of the hypothesis test.

    See Also
    --------
    energy_distance

    Examples
    --------
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
    HypothesisTest(p_value=1.0, statistic=0.0)
    >>> dcor.homogeneity.energy_test(a, b) # doctest: +ELLIPSIS
    HypothesisTest(p_value=1.0, statistic=35.2766732...)
    >>> dcor.homogeneity.energy_test(b, b)
    HypothesisTest(p_value=1.0, statistic=0.0)
    >>> np.random.seed(0)
    >>> dcor.homogeneity.energy_test(a, b, num_resamples=5)
    HypothesisTest(p_value=0.1666666..., statistic=35.2766732...)
    >>> np.random.seed(6)
    >>> dcor.homogeneity.energy_test(a, b, num_resamples=5)
    HypothesisTest(p_value=0.3333333..., statistic=35.2766732...)
    >>> np.random.seed(0)
    >>> dcor.homogeneity.energy_test(a, c, num_resamples=7)
    HypothesisTest(p_value=0.125, statistic=4233.8935035...)

    A different exponent for the Euclidean distance in the range
    :math:`(0, 2)` can be used:

    >>> dcor.homogeneity.energy_test(a, b, exponent=1.5) # doctest: +ELLIPSIS
    HypothesisTest(p_value=1.0, statistic=171.0623923...)
    """
    if isinstance(stat_type, str):
        stat_type = _energy.EstimationStatistic.from_string(stat_type)

    samples = [_transform_to_2d(a) for a in args]

    # k
    num_samples = len(samples)

    _energy._check_valid_energy_exponent(exponent)

    # alpha
    # significance_level = 1.0 / (num_resamples + 1)

    # {n_1, ..., n_k}
    sample_sizes = tuple(a.shape[0] for a in samples)

    # {W_1, ..., W_n}
    pooled_samples = _np.concatenate(samples)

    # {m_0, ..., m_(k-1)}
    sample_indexes = _np.zeros(num_samples, dtype=int)
    sample_indexes[1:] = _np.cumsum(sample_sizes)[:-1]

    # Compute the distance matrix once
    sample_distances = _distances.pairwise_distances(pooled_samples,
                                                     exponent=exponent)

    # Use the energy statistic with appropriate values
    @njit()
    def statistic_function(distance_matrix):
        return _energy_test_statistic_multivariate_from_distance_matrix(
            distance=distance_matrix,
            indexes=sample_indexes,
            sizes=sample_sizes,
            average=average,
            stat_type=stat_type
        )

    return _hypothesis._permutation_test_with_sym_matrix(
        sample_distances,
        statistic_function=statistic_function,
        num_resamples=num_resamples
    )

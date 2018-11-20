"""
Functions for testing homogeneity of several distributions.

The functions in this module provide methods for testing if
the samples generated from two random vectors have the same
distribution.
"""

from __future__ import absolute_import, division, print_function

import numpy as _np

from . import _energy
from . import _hypothesis
from . import distances as _distances
from ._utils import _transform_to_2d


def _energy_test_statistic_coefficient(n, m):
    """Coefficient of the test statistic."""
    return n * m / (n + m)


def _energy_test_statistic_from_distance_matrices(
        distance_xx, distance_yy, distance_xy, n, m):
    """Test statistic with precomputed distance matrices."""
    energy_distance = _energy._energy_distance_from_distance_matrices(
        distance_xx=distance_xx, distance_yy=distance_yy,
        distance_xy=distance_xy
    )

    return _energy_test_statistic_coefficient(n, m) * energy_distance


def energy_test_statistic(x, y, **kwargs):
    """
    energy_test_statistic(x, y, *, exponent=1)

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
    x = _transform_to_2d(x)
    y = _transform_to_2d(y)

    n = x.shape[0]
    m = y.shape[0]

    coefficient = _energy_test_statistic_coefficient(n, m)

    return coefficient * _energy.energy_distance(x, y, **kwargs)


def _energy_test_statistic_multivariate_from_distance_matrix(
        distance, indexes, sizes):
    """Statistic for several random vectors given the distance matrix."""
    energy = 0.0

    for i, _ in enumerate(indexes):
        for j in range(i + 1, len(indexes)):
            range_i = range(indexes[i], indexes[i] + sizes[i])
            range_j = range(indexes[j], indexes[j] + sizes[j])

            n = sizes[i]
            m = sizes[j]

            distance_xx = distance[_np.ix_(range_i, range_i)]
            distance_yy = distance[_np.ix_(range_j, range_j)]
            distance_xy = distance[_np.ix_(range_i, range_j)]

            pairwise_energy = _energy_test_statistic_from_distance_matrices(
                distance_xx=distance_xx, distance_yy=distance_yy,
                distance_xy=distance_xy, n=n, m=m)

            energy += pairwise_energy

    return energy


def _energy_test_imp(samples, num_resamples=0,
                     exponent=1, random_state=None):
    """
    Real implementation of :func:`energy_test`.

    This function is used to make parameters ``num_resamples``, ``exponent``
    and ``random_state`` keyword-only in Python 2.

    """
    # k
    num_samples = len(samples)

    _energy._check_valid_energy_exponent(exponent)

    # alpha
    # significance_level = 1.0 / (num_resamples + 1)

    # {n_1, ..., n_k}
    sample_sizes = [a.shape[0] for a in samples]

    # {W_1, ..., W_n}
    pooled_samples = _np.concatenate(samples)

    # {m_0, ..., m_(k-1)}
    sample_indexes = _np.zeros(num_samples, dtype=int)
    sample_indexes[1:] = _np.cumsum(sample_sizes)[:-1]

    # Compute the distance matrix once
    sample_distances = _distances.pairwise_distances(pooled_samples,
                                                     exponent=exponent)

    # Use the energy statistic with appropiate values
    def statistic_function(distance_matrix):
        return _energy_test_statistic_multivariate_from_distance_matrix(
            distance=distance_matrix,
            indexes=sample_indexes,
            sizes=sample_sizes)

    return _hypothesis._permutation_test_with_sym_matrix(
        sample_distances,
        statistic_function=statistic_function,
        num_resamples=num_resamples,
        random_state=random_state)


def energy_test(*args, **kwargs):
    """
    energy_test(*args, num_resamples=0, exponent=1, random_state=None)

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
    random_state: {None, int, array_like, numpy.random.RandomState}
        Random state to generate the permutations.

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
    >>> dcor.homogeneity.energy_test(a, b, num_resamples=5, random_state=0)
    HypothesisTest(p_value=0.1666666..., statistic=35.2766732...)
    >>> dcor.homogeneity.energy_test(a, b, num_resamples=5, random_state=6)
    HypothesisTest(p_value=0.3333333..., statistic=35.2766732...)
    >>> dcor.homogeneity.energy_test(a, c, num_resamples=7, random_state=0)
    HypothesisTest(p_value=0.125, statistic=4233.8935035...)

    A different exponent for the Euclidean distance in the range
    :math:`(0, 2)` can be used:

    >>> dcor.homogeneity.energy_test(a, b, exponent=1.5) # doctest: +ELLIPSIS
    HypothesisTest(p_value=1.0, statistic=171.0623923...)

    """

    samples = [_transform_to_2d(a) for a in args]

    return _energy_test_imp(samples, **kwargs)

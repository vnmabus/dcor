import collections

import numpy as np
from numba import njit, prange

HypothesisTest = collections.namedtuple('HypothesisTest', ['p_value',
                                        'statistic'])


@njit()
def _numba_permute(matrix):
    """
    Calculates a permutation of a matrix, in a numba-compatible manner
    """
    permuted_matrix = np.zeros_like(matrix)
    perm = np.random.permutation(matrix.shape[0])
    for out_i, in_i in enumerate(perm):
        for out_j, in_j in enumerate(perm):
            permuted_matrix[out_i, out_j] = matrix[in_i, in_j]
    return permuted_matrix


@njit()
def _permutation_test_with_sym_matrix(matrix, statistic_function,
                                      num_resamples, random_state=None):
    """
    Execute a permutation test in a symmetric matrix.

    Parameters
    ----------
    matrix: array_like
        Matrix that will perform the permutation test.
    statistic_function: callable
        Function that computes the desired statistic from the matrix.
    num_resamples: int
        Number of permutations resamples to take in the permutation test.
    random_state: int
        Integer used for seeding the random number generator

    Returns
    -------
    HypothesisTest
        Results of the hypothesis test.
    """
    if random_state:
        np.random.seed(random_state)

    statistic = statistic_function(matrix)

    bootstrap_statistics = np.ones(num_resamples, np.float_)

    for bootstrap in prange(num_resamples):
        permuted_matrix = _numba_permute(matrix)
        bootstrap_statistics[bootstrap] = statistic_function(permuted_matrix)

    extreme_results = bootstrap_statistics > statistic
    p_value = (np.sum(extreme_results) + 1.0) / (num_resamples + 1)

    return HypothesisTest(
        p_value=p_value,
        statistic=statistic
    )

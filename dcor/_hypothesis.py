import collections

import numpy as np
from joblib import Parallel, delayed
from ._utils import _random_state_init

HypothesisTest = collections.namedtuple('HypothesisTest', ['p_value',
                                        'statistic'])


def _permutation_test_with_sym_matrix(matrix, statistic_function,
                                      num_resamples, random_state,n_jobs=1):
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
    random_state: {None, int, array_like, numpy.random.RandomState}
        Random state to generate the permutations.

    Returns
    -------
    HypothesisTest
        Results of the hypothesis test.
    """
    matrix = np.asarray(matrix)
    random_state = _random_state_init(random_state)

    statistic = statistic_function(matrix)

    def bootstrapPerms(mat):
        permuted_index = random_state.permutation(mat.shape[0])

        permuted_matrix = mat[
            np.ix_(permuted_index, permuted_index)]

        return statistic_function(permuted_matrix)

    bootstrap_statistics = Parallel(n_jobs=n_jobs)(delayed(bootstrapPerms)(matrix) for bootstrap in range(num_resamples))
    bootstrap_statistics = np.array(bootstrap_statistics, dtype=statistic.dtype)

    extreme_results = bootstrap_statistics > statistic
    p_value = (np.sum(extreme_results) + 1.0) / (num_resamples + 1)

    return HypothesisTest(
        p_value=p_value,
        statistic=statistic
    )

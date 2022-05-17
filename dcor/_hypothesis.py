from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Iterator

import numpy as np
from dcor._utils import ArrayType
from joblib import Parallel, delayed

from ._utils import _random_state_init


@dataclass
class HypothesisTest():
    pvalue: float
    statistic: ArrayType

    @property
    def p_value(self) -> float:
        """Old name for pvalue."""
        warnings.warn(
            "Attribute \"p_value\" deprecated, use \"pvalue\" instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.pvalue

    def __iter__(self) -> Iterator[Any]:
        warnings.warn(
            "HypothesisTest will cease to be iterable.",
            DeprecationWarning,
        )
        return iter((self.pvalue, self.statistic))

    def __len__(self) -> int:
        warnings.warn(
            "HypothesisTest will cease to be sized.",
            DeprecationWarning,
        )
        return 2


def _permuted_statistic(
    matrix: ArrayType,
    statistic_function: Callable[[ArrayType], ArrayType],
    permutation: np.typing.NDArray[int],
) -> ArrayType:

    permuted_matrix = matrix[np.ix_(permutation, permutation)]

    return statistic_function(permuted_matrix)


def _permutation_test_with_sym_matrix(
    matrix: ArrayType,
    *,
    statistic_function: Callable[[ArrayType], ArrayType],
    num_resamples: int,
    random_state: np.random.RandomState | np.random.Generator | int | None,
    n_jobs: int | None = None,
) -> HypothesisTest:
    """
    Execute a permutation test in a symmetric matrix.

    Parameters:
        matrix: Matrix that will perform the permutation test.
        statistic_function: Function that computes the desired statistic from
            the matrix.
        num_resamples: Number of permutations resamples to take in the
            permutation test.
        random_state: Random state to generate the permutations.
        n_jobs: Number of jobs executed in parallel by Joblib.

    Returns:
        Results of the hypothesis test.

    """
    matrix = np.asarray(matrix)
    random_state = _random_state_init(random_state)

    statistic = statistic_function(matrix)

    permutations = (
        random_state.permutation(matrix.shape[0])
        for _ in range(num_resamples)
    )

    bootstrap_statistics = Parallel(n_jobs=n_jobs)(
        delayed(_permuted_statistic)(
            matrix,
            statistic_function,
            permutation,
        ) for permutation in permutations
    )
    bootstrap_statistics = np.array(
        bootstrap_statistics,
        dtype=statistic.dtype,
    )

    extreme_results = bootstrap_statistics > statistic
    pvalue = (np.sum(extreme_results) + 1.0) / (num_resamples + 1)

    return HypothesisTest(
        pvalue=pvalue,
        statistic=statistic,
    )

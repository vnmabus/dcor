from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Generic, Iterator, TypeVar

import numpy as np
from joblib import Parallel, delayed

from ._utils import ArrayType, RandomLike, _random_state_init, get_namespace

T = TypeVar("T", bound=ArrayType)


@dataclass
class HypothesisTest(Generic[T]):
    """
    Class containing the results of an hypothesis test.
    """
    pvalue: float
    statistic: T

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
    matrix: T,
    statistic_function: Callable[[T], T],
    permutation: np.typing.NDArray[int],
) -> T:

    xp = get_namespace(matrix)

    # We implicitly convert to NumPy for permuting the array if we don't
    # have a take function.
    # take is probably going to be included in the final version of the
    # standard, so not much to worry about.
    take = getattr(xp, "take", np.take)

    permuted_rows = take(matrix, permutation, axis=0)
    permuted_matrix = take(permuted_rows, permutation, axis=1)

    # Transform back to the original type if NumPy conversion was needed.
    permuted_matrix = xp.asarray(permuted_matrix)

    return statistic_function(permuted_matrix)


def _permutation_test_with_sym_matrix(
    matrix: T,
    *,
    statistic_function: Callable[[T], T],
    num_resamples: int,
    random_state: RandomLike,
    n_jobs: int | None = None,
) -> HypothesisTest[T]:
    """
    Execute a permutation test in a symmetric matrix.

    Args:
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
    xp = get_namespace(matrix)
    matrix = xp.asarray(matrix)
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

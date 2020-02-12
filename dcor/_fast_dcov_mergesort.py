'''
Functions to compute fast distance covariance using mergesort.
'''

import numpy as np

from ._utils import _jit


#@_jit
def _compute_weight_sums(y, weights):

    n_samples = len(y)

    weight_sums = np.zeros((n_samples,) + weights.shape[1:], dtype=y.dtype)

    # Buffer that contains the indexes of the current and
    # last iterations
    indexes = np.arange(2 * n_samples).reshape((2, n_samples))
    indexes[1] = 0  # Remove this

    previous_indexes = indexes[0]
    current_indexes = indexes[1]

    weights_cumsum = np.zeros(
        (n_samples + 1,) + weights.shape[1:], dtype=weights.dtype)

    merged_subarray_len = 1

    # For all lengths that are a power of two
    while merged_subarray_len < n_samples:
        gap = 2 * merged_subarray_len
        indexes_idx = 0

        # Numba does not support axis, nor out parameter.
        for var in range(weights.shape[1]):
            weights_cumsum[1:, var] = np.cumsum(
                weights[previous_indexes, var])

        # Select the subarrays in pairs
        for subarray_pair_idx in range(0, n_samples, gap):
            subarray_1_idx = subarray_pair_idx
            subarray_2_idx = subarray_pair_idx + merged_subarray_len
            subarray_1_idx_last = min(
                subarray_1_idx + merged_subarray_len - 1, n_samples - 1)
            subarray_2_idx_last = min(
                subarray_2_idx + merged_subarray_len - 1, n_samples - 1)

            # Merge the subarrays
            while (subarray_1_idx <= subarray_1_idx_last and
                   subarray_2_idx <= subarray_2_idx_last):
                previous_index_1 = previous_indexes[subarray_1_idx]
                previous_index_2 = previous_indexes[subarray_2_idx]

                if y[previous_index_1] >= y[previous_index_2]:
                    current_indexes[indexes_idx] = previous_index_1
                    subarray_1_idx += 1
                else:
                    current_indexes[indexes_idx] = previous_index_2
                    subarray_2_idx += 1

                    weight_sums[previous_index_2] += (
                        weights_cumsum[subarray_1_idx_last + 1] -
                        weights_cumsum[subarray_1_idx])
                indexes_idx += 1

            # Join the remaining elements of one of the arrays (already sorted)
            if subarray_1_idx <= subarray_1_idx_last:
                n_remaining = subarray_1_idx_last - subarray_1_idx + 1
                indexes_idx_next = indexes_idx + n_remaining
                current_indexes[indexes_idx:indexes_idx_next] = (
                    previous_indexes[subarray_1_idx:subarray_1_idx_last + 1])
                indexes_idx = indexes_idx_next
            elif subarray_2_idx <= subarray_2_idx_last:
                n_remaining = subarray_2_idx_last - subarray_2_idx + 1
                indexes_idx_next = indexes_idx + n_remaining
                current_indexes[indexes_idx:indexes_idx_next] = (
                    previous_indexes[subarray_2_idx:subarray_2_idx_last + 1])
                indexes_idx = indexes_idx_next

        merged_subarray_len = gap

        # Swap buffer
        previous_indexes, current_indexes = (current_indexes, previous_indexes)

    return weight_sums


def _distance_covariance_mergesort(x, y):

    # Sort x in ascending order
    ordered_indexes = np.argsort(x, axis=0).ravel()
    x = x[ordered_indexes]
    y = y[ordered_indexes]

    weights = np.hstack((np.ones_like(y), y, x, x * y))
    weight_sums = _compute_weight_sums(y, weights)

    term_1 = (x * y).T @ weight_sums[:, 0]
    term_2 = x.T @ weight_sums[:, 0]
    term_3 = y.T @ weight_sums[:, 0]
    term_4 = np.sum(weight_sums[:, 0])

    # First term in the equation
    sums_term = term_1 - term_2 - term_3 + term_4
    print(sums_term)

    # Second term in the equation
    cov_term = (x - np.mean(x)).T @ (y - np.mean(y))
    print(cov_term)

    d = 4 * sums_term - 2 * cov_term

    return d.item()

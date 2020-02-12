'''
Functions to compute fast distance covariance using mergesort.
'''

import numpy as np


def _compute_d(y, weights):

    n_samples = len(y)

    d = np.empty(n_samples, dtype=y.dtype)

    # Buffer that contains the indexes of the current and
    # last iterations
    indexes = np.arange(2 * n_samples).reshape((2, n_samples))
    indexes[1] = 0  # Remove this

    previous_indexes = indexes[0]
    current_indexes = indexes[1]

    merged_subarray_len = 1

    # For all lengths that are a power of two
    while merged_subarray_len < n_samples:
        gap = 2 * merged_subarray_len
        indexes_idx = 0
        weights_cumsum = np.cumsum(weights[previous_indexes])
        weights_cumsum = np.concatenate(([0], weights_cumsum))

        # Select the subarrays in pairs
        subarray_pair_idx = 0
        while subarray_pair_idx < n_samples:
            subarray_1_idx = subarray_pair_idx
            subarray_2_idx = subarray_pair_idx + merged_subarray_len
            subarray_1_idx_last = min(
                subarray_1_idx + merged_subarray_len - 1, n_samples - 1)
            subarray_2_idx_last = min(
                subarray_2_idx + merged_subarray_len - 1, n_samples - 1)
            e1_plus_1 = subarray_1_idx_last + 1

            # Merge the subarrays
            while (subarray_1_idx <= subarray_1_idx_last and
                   subarray_2_idx <= subarray_2_idx_last):
                previous_index_1 = previous_indexes[subarray_1_idx]
                previous_index_2 = previous_indexes[subarray_2_idx]

                if y[previous_index_1] > y[previous_index_2]:
                    current_indexes[indexes_idx] = previous_index_1
                    subarray_1_idx += 1
                else:
                    current_indexes[indexes_idx] = previous_index_2
                    subarray_2_idx += 1

                    d[previous_index_2] += (weights_cumsum[e1_plus_1] -
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

            subarray_pair_idx += gap

        merged_subarray_len = gap

        # Swap buffer
        previous_indexes, current_indexes = (current_indexes, previous_indexes)

        print(previous_indexes + 1)

    return d

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

    i = 1
    previous_idx = 1
    current_idx = 2

    while i < n_samples:
        gap = 2 * i
        k = 0
        previous_indexes = indexes[previous_idx]
        weights_cumsum = np.cumsum(weights[previous_indexes])
        weights_cumsum = np.concatenate(([0], weights_cumsum))

        j = 1
        while j < n_samples:
            st1 = j
            st2 = j + i
            e1 = min(st1 + i - 1, n_samples)
            e2 = min(st2 + i - 1, n_samples)
            e1_plus_1 = e1 + 1

            while st1 <= e1 and st2 <= e2:
                k += 1
                idx1 = previous_indexes[st1]
                idx2 = previous_indexes[st2]

                if y[idx1] > y[idx2]:
                    indexes[current_idx, k] = idx1
                    st1 += 1
                else:
                    indexes[current_idx, k] = idx2
                    st2 += 1

                    d[idx2] += (weights_cumsum[e1_plus_1] -
                                weights_cumsum[st1])

            if st1 <= e1:
                kf = k + e1 - st1 + 1
                indexes[current_idx, (k + 1):kf] = previous_indexes[:, st1:e1]
                k = kf
            elif st2 <= e2:
                kf = k + e2 - st2 + 1
                indexes[current_idx, (k + 1):kf] = previous_indexes[:, st2:e2]
                k = kf

            j += gap

        i = gap

        # Swap buffer
        previous_idx, current_idx = (current_idx, previous_idx)

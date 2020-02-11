'''
Functions to compute fast distance covariance using mergesort.
'''

import numpy as np


def _compute_d(y, t):

    n_samples = len(y)

    d = np.empty(n_samples, dtype=y.dtype)

    idx = np.arange(2 * n_samples).reshape((2, n_samples))
    idx[1] = 0

    i = 1
    r = 1
    s = 2

    while i < n_samples:
        gap = 2 * i
        k = 0
        idx_r = idx[r]
        csumT = np.cumsum(t[idx_r])
        csumT = np.concatenate(([0], csumT))

        j = 1
        while j < n_samples:
            st1 = j
            st2 = j + i
            e1 = min(st1 + i - 1, n_samples)
            e2 = min(st2 + i - 1, n_samples)
            e1_plus_1 = e1 + 1

            while st1 <= e1 and st2 <= e2:
                k += 1
                idx1 = idx_r[st1]
                idx2 = idx_r[st2]

                if y[idx1] > y[idx2]:
                    idx[s, k] = idx1
                    st1 += 1
                else:
                    idx[s, k] = idx2
                    st2 += 1

                    d[idx2] += (csumT[e1_plus_1] - csumT[st1])

            if st1 <= e1:
                kf = k + e1 - st1 + 1
                idx[s, (k + 1):kf] = idx_r[:, st1:e1]
                k = kf
            elif st2 <= e2:
                kf = k + e2 - st2 + 1
                idx[s, (k + 1):kf] = idx_r[:, st2:e2]
                k = kf

            j += gap

        i = gap
        r = 3 - r
        s = 3 - s

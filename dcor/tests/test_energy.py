import numpy as np
from dcor._energy import _energy_distance_from_distance_matrices
import unittest


class TestEnergyTest(unittest.TestCase):
    """Tests for energy statistics"""

    def test_u_v_statistics(self):
        """
        V-statistics count the 0 terms on the diagonal, so the impact of the
        within-sample distance will be smaller, making the overall distance
        larger
        """
        distance_within = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ])
        distance_between = np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 2],
        ])

        u_dist = _energy_distance_from_distance_matrices(
            distance_xx=distance_within,
            distance_yy=distance_within,
            distance_xy=distance_between,
            stat_type='u'
        )
        v_dist = _energy_distance_from_distance_matrices(
            distance_xx=distance_within,
            distance_yy=distance_within,
            distance_xy=distance_between,
            stat_type='v'
        )

        assert u_dist > v_dist

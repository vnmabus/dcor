import numpy as np
from dcor._energy import _energy_distance_from_distance_matrices,\
    EstimationStatistic
import unittest


class TestEnergyTest(unittest.TestCase):
    """Tests for energy statistics"""

    def test_u_v_statistics(self):
        """
        Test that we are correctly applying the provided type of estimation
        statistics

        Also tests that we can choose a statistic type using either a string
        or an enum instance
        """
        distance_within = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ])
        distance_between = np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ])

        u_dist = _energy_distance_from_distance_matrices(
            distance_xx=distance_within,
            distance_yy=distance_within,
            distance_xy=distance_between,
            stat_type=EstimationStatistic.USTATISTIC
        )
        v_dist = _energy_distance_from_distance_matrices(
            distance_xx=distance_within,
            distance_yy=distance_within,
            distance_xy=distance_between,
            stat_type='v'
        )

        # V-statistics count the 0 terms on the diagonal, so the impact of the
        # within-sample distance will be smaller, making the overall distance
        # larger.
        self.assertGreater(v_dist, u_dist)

        # Also test for exact values
        self.assertEqual(u_dist, 0)
        self.assertAlmostEqual(v_dist, 2/3)

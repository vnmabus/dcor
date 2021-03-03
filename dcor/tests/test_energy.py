import numpy as np
from dcor._energy import _energy_distance_from_distance_matrices
from dcor import EstimationStatistic
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
            estimation_stat=EstimationStatistic.U_STATISTIC
        )
        u_dist_2 = _energy_distance_from_distance_matrices(
            distance_xx=distance_within,
            distance_yy=distance_within,
            distance_xy=distance_between,
            estimation_stat='U_STATISTIC'
        )
        u_dist_3 = _energy_distance_from_distance_matrices(
            distance_xx=distance_within,
            distance_yy=distance_within,
            distance_xy=distance_between,
            estimation_stat='u_statistic'
        )
        v_dist = _energy_distance_from_distance_matrices(
            distance_xx=distance_within,
            distance_yy=distance_within,
            distance_xy=distance_between,
            estimation_stat='v'
        )

        # The first 3 invocations should be identical
        self.assertEqual(u_dist, u_dist_2)
        self.assertEqual(u_dist, u_dist_3)

        # V-statistics count the 0 terms on the diagonal, so the impact of the
        # within-sample distance will be smaller, making the overall distance
        # larger.
        self.assertGreater(v_dist, u_dist)

        # Also test for exact values
        self.assertEqual(u_dist, 0)
        self.assertAlmostEqual(v_dist, 2/3)

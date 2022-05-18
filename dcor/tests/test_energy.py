import unittest

import numpy as np
import numpy.array_api

from dcor import EstimationStatistic, energy_distance
from dcor._energy import _energy_distance_from_distance_matrices


class TestEnergy(unittest.TestCase):
    """Tests for energy statistics."""

    def test_u_v_statistics(self) -> None:
        """
        Check we are correctly applying the type of estimation statistics.

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
            estimation_stat=EstimationStatistic.U_STATISTIC,
        )
        u_dist_2 = _energy_distance_from_distance_matrices(
            distance_xx=distance_within,
            distance_yy=distance_within,
            distance_xy=distance_between,
            estimation_stat='U_STATISTIC',
        )
        u_dist_3 = _energy_distance_from_distance_matrices(
            distance_xx=distance_within,
            distance_yy=distance_within,
            distance_xy=distance_between,
            estimation_stat='u_statistic',
        )
        v_dist = _energy_distance_from_distance_matrices(
            distance_xx=distance_within,
            distance_yy=distance_within,
            distance_xy=distance_between,
            estimation_stat='v',
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
        self.assertAlmostEqual(float(v_dist), 2 / 3)


class TestEnergyArrayAPI(unittest.TestCase):
    """Check that the energy distance works with the Array API standard."""

    def setUp(self) -> None:
        """Initialize Array API arrays."""
        self.a = numpy.array_api.asarray(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16],
            ],
            dtype=numpy.array_api.float64,
        )
        self.b = numpy.array_api.asarray(
            [
                [1, 0, 0, 1],
                [0, 1, 1, 1],
                [1, 1, 1, 1],
            ],
            dtype=numpy.array_api.float64,
        )

    def test_basic(self) -> None:
        """Basic check of energy_distance."""
        edist_aa = energy_distance(self.a, self.a)
        edist_ab = energy_distance(self.a, self.b)
        edist_bb = energy_distance(self.b, self.b)

        self.assertIsInstance(edist_aa, type(self.a))
        self.assertIsInstance(edist_ab, type(self.a))
        self.assertIsInstance(edist_bb, type(self.a))

        self.assertAlmostEqual(float(edist_aa), 0)
        self.assertAlmostEqual(float(edist_ab), 20.5780594)
        self.assertAlmostEqual(float(edist_bb), 0)

    def test_exponent(self) -> None:
        """Check for non default exponent."""
        edist_aa = energy_distance(self.a, self.a, exponent=1.5)
        edist_ab = energy_distance(self.a, self.b, exponent=1.5)
        edist_bb = energy_distance(self.b, self.b, exponent=1.5)

        self.assertIsInstance(edist_aa, type(self.a))
        self.assertIsInstance(edist_ab, type(self.a))
        self.assertIsInstance(edist_bb, type(self.a))

        self.assertAlmostEqual(float(edist_aa), 0)
        self.assertAlmostEqual(float(edist_ab), 99.786395559)
        self.assertAlmostEqual(float(edist_bb), 0)

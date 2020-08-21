"""Tests of the distance covariance and correlation"""

import dcor
import dcor._fast_dcov_avl
from decimal import Decimal
from fractions import Fraction
import math
import unittest

import numpy as np


class TestDistanceCorrelation(unittest.TestCase):
    """Distance correlation tests"""

    def setUp(self):
        """Set the common parameters"""
        self.test_max_size = 10

    def test_double_centered(self):
        """
        Test that the double centering is right.

        Check that the sum of the rows and colums of a double centered
        matrix is 0.

        """
        for i in range(1, self.test_max_size + 1):
            matrix = np.random.RandomState(i).rand(i, i)

            centered_matrix = dcor.double_centered(matrix)

            column_sum = np.sum(centered_matrix, 0)
            row_sum = np.sum(centered_matrix, 1)

            np.testing.assert_allclose(column_sum, np.zeros(i), atol=1e-8)
            np.testing.assert_allclose(row_sum, np.zeros(i), atol=1e-8)

    def test_dyad_update(self):  # pylint:disable=no-self-use
        """Compare dyad update results with the original code in the paper."""
        y = np.array([1, 2, 3])
        c = np.array([4, 5, 6])

        n = len(y)
        gamma1 = np.zeros(n, dtype=c.dtype)

        # Step 1: get the smallest l such that n <= 2^l
        l_max = int(math.ceil(np.log2(n)))

        # Step 2: assign s(l, k) = 0
        s_len = 2 ** (l_max + 1)
        s = np.zeros(s_len, dtype=c.dtype)

        pos_sums = np.arange(l_max)
        pos_sums[:] = 2 ** (l_max - pos_sums)
        pos_sums = np.cumsum(pos_sums)

        gamma = dcor._fast_dcov_avl._dyad_update(
            y, c, gamma1, l_max, s, pos_sums)
        expected_gamma = [0., 4., 9.]

        np.testing.assert_allclose(gamma, expected_gamma)

    def test_distance_correlation_multivariate(self):
        """Compare distance correlation with the energy package."""
        matrix1 = np.array(((1, 2, 3), (4, 5, 6), (7, 8, 9)))
        matrix2 = np.array(((7, 3, 6), (2, 1, 4), (3, 8, 1)))
        matrix3 = np.array(((1, 1, 1), (2, 1, 1), (1, 1, 1)))
        constant_matrix = np.ones((3, 3))

        correlation = dcor.distance_correlation_sqr(
            matrix1, matrix1)
        self.assertAlmostEqual(correlation, 1)

        correlation = dcor.distance_correlation_sqr(
            matrix1, constant_matrix)
        self.assertAlmostEqual(correlation, 0)

        correlation = dcor.distance_correlation_sqr(
            matrix1, matrix2)
        self.assertAlmostEqual(correlation, 0.93387, places=5)

        correlation = dcor.distance_correlation_sqr(
            matrix1, matrix3)
        self.assertAlmostEqual(correlation, 0.31623, places=5)

    def test_distance_correlation_comparison(self):
        """
        Compare all implementations of the distance covariance and correlation.
        """
        arr1 = np.array(((1.,), (2.,), (3.,), (4.,), (5.,), (6.,)))
        arr2 = np.array(((1.,), (7.,), (5.,), (5.,), (6.,), (2.,)))

        for method in dcor.DistanceCovarianceMethod:
            with self.subTest(method=method):

                compile_modes = [dcor.CompileMode.AUTO,
                                 dcor.CompileMode.COMPILE_CPU,
                                 dcor.CompileMode.NO_COMPILE]

                if method is not dcor.DistanceCovarianceMethod.NAIVE:
                    compile_modes += [dcor.CompileMode.COMPILE_CPU]

                for compile_mode in compile_modes:
                    with self.subTest(compile_mode=compile_mode):

                        # Unbiased versions

                        covariance = dcor.u_distance_covariance_sqr(
                            arr1, arr2, method=method,
                            compile_mode=compile_mode)
                        self.assertAlmostEqual(covariance, -0.88889, places=5)

                        correlation = dcor.u_distance_correlation_sqr(
                            arr1, arr2, method=method,
                            compile_mode=compile_mode)
                        self.assertAlmostEqual(correlation, -0.41613, places=5)

                        covariance = dcor.u_distance_covariance_sqr(
                            arr1, arr1,  method=method,
                            compile_mode=compile_mode)
                        self.assertAlmostEqual(covariance, 1.55556, places=5)

                        correlation = dcor.u_distance_correlation_sqr(
                            arr1, arr1,  method=method,
                            compile_mode=compile_mode)
                        self.assertAlmostEqual(correlation, 1, places=5)

                        covariance = dcor.u_distance_covariance_sqr(
                            arr2, arr2,  method=method,
                            compile_mode=compile_mode)
                        self.assertAlmostEqual(covariance, 2.93333, places=5)

                        correlation = dcor.u_distance_correlation_sqr(
                            arr2, arr2,  method=method,
                            compile_mode=compile_mode)
                        self.assertAlmostEqual(correlation, 1, places=5)

                        stats = dcor.u_distance_stats_sqr(
                            arr1, arr2, method=method,
                            compile_mode=compile_mode)
                        np.testing.assert_allclose(
                            stats, (-0.88889, -0.41613, 1.55556, 2.93333),
                            rtol=1e-4)

                        # Biased

                        covariance = dcor.distance_covariance_sqr(
                            arr1, arr2, method=method,
                            compile_mode=compile_mode)
                        self.assertAlmostEqual(covariance, 0.68519, places=5)

                        correlation = dcor.distance_correlation_sqr(
                            arr1, arr2, method=method,
                            compile_mode=compile_mode)
                        self.assertAlmostEqual(correlation, 0.30661, places=5)

                        covariance = dcor.distance_covariance_sqr(
                            arr1, arr1,  method=method,
                            compile_mode=compile_mode)
                        self.assertAlmostEqual(covariance, 1.70679, places=5)

                        correlation = dcor.distance_correlation_sqr(
                            arr1, arr1,  method=method,
                            compile_mode=compile_mode)
                        self.assertAlmostEqual(correlation, 1, places=5)

                        covariance = dcor.distance_covariance_sqr(
                            arr2, arr2,  method=method,
                            compile_mode=compile_mode)
                        self.assertAlmostEqual(covariance, 2.92593, places=5)

                        correlation = dcor.distance_correlation_sqr(
                            arr2, arr2,  method=method,
                            compile_mode=compile_mode)
                        self.assertAlmostEqual(correlation, 1, places=5)

                        stats = dcor.distance_stats_sqr(
                            arr1, arr2, method=method,
                            compile_mode=compile_mode)
                        np.testing.assert_allclose(
                            stats, (0.68519, 0.30661, 1.70679, 2.92593),
                            rtol=1e-4)

    def test_u_distance_covariance_avl_overflow(self):
        """Test potential overflow in fast distance correlation"""
        arr1 = np.concatenate((np.zeros(500, dtype=int),
                               np.ones(500, dtype=int)))
        covariance = dcor.u_distance_covariance_sqr(
            arr1, arr1,
            method='avl',
            compile_mode=dcor.CompileMode.NO_COMPILE)
        self.assertAlmostEqual(covariance, 0.25050, places=5)

    def _test_u_distance_correlation_vector_generic(self,
                                                    vector_type=None,
                                                    type_cov=None,
                                                    type_cor=None):
        """
        Auxiliar function for testing U-distance correlation in vectors.

        This function is provided to check that the results are the
        same with different dtypes, but that the dtype of the result is
        the right one.
        """
        if type_cov is None:
            type_cov = vector_type
        if type_cor is None:
            type_cor = vector_type

        arr1 = np.array([vector_type(1), vector_type(2), vector_type(3),
                         vector_type(4), vector_type(5), vector_type(6)])
        arr2 = np.array([vector_type(1), vector_type(7), vector_type(5),
                         vector_type(5), vector_type(6), vector_type(2)])

        covariance = dcor.u_distance_covariance_sqr(
            arr1, arr2, compile_mode=dcor.CompileMode.NO_COMPILE)
        self.assertIsInstance(covariance, type_cov)
        self.assertAlmostEqual(covariance, type_cov(-0.88889), places=5)

        correlation = dcor.u_distance_correlation_sqr(
            arr1, arr2, compile_mode=dcor.CompileMode.NO_COMPILE)
        self.assertIsInstance(correlation, type_cor)
        self.assertAlmostEqual(correlation, type_cor(-0.41613), places=5)

        covariance = dcor.u_distance_covariance_sqr(
            arr1, arr1, compile_mode=dcor.CompileMode.NO_COMPILE)
        self.assertIsInstance(covariance, type_cov)
        self.assertAlmostEqual(covariance, type_cov(1.5556), places=4)

        correlation = dcor.u_distance_correlation_sqr(
            arr1, arr1, compile_mode=dcor.CompileMode.NO_COMPILE)
        self.assertIsInstance(correlation, type_cor)
        self.assertAlmostEqual(correlation, type_cor(1), places=5)

    def test_u_distance_correlation_vector(self):
        """Check U-distance with vectors of float."""
        return self._test_u_distance_correlation_vector_generic(
            vector_type=float
        )

    def test_u_distance_correlation_vector_ints(self):
        """Check U-distance with vectors of integers."""
        return self._test_u_distance_correlation_vector_generic(
            vector_type=int,
            type_cov=float,
            type_cor=float
        )

    def test_u_distance_correlation_vector_fractions(self):
        """
        Check U-distance with vectors of fractions.

        Note that the correlation is given in floating point, as
        fractions can not generally represent a square root.
        """
        return self._test_u_distance_correlation_vector_generic(
            vector_type=Fraction,
            type_cor=float
        )

    def test_u_distance_correlation_vector_decimal(self):
        """Check U-distance with vectors of Decimal."""
        return self._test_u_distance_correlation_vector_generic(
            vector_type=Decimal
        )

    def _test_distance_correlation_vector_generic(self,
                                                  vector_type=None,
                                                  type_cov=None,
                                                  type_cor=None):
        """
        Auxiliar function for testing distance correlation in vectors.

        This function is provided to check that the results are the
        same with different dtypes, but that the dtype of the result is
        the right one.
        """
        if type_cov is None:
            type_cov = vector_type
        if type_cor is None:
            type_cor = vector_type

        arr1 = np.array([vector_type(1), vector_type(2), vector_type(3),
                         vector_type(4), vector_type(5), vector_type(6)])
        arr2 = np.array([vector_type(1), vector_type(7), vector_type(5),
                         vector_type(5), vector_type(6), vector_type(2)])

        covariance = dcor.distance_covariance_sqr(
            arr1, arr2, compile_mode=dcor.CompileMode.NO_COMPILE)
        self.assertIsInstance(covariance, type_cov)
        self.assertAlmostEqual(covariance, type_cov(0.6851851), places=6)

        correlation = dcor.distance_correlation_sqr(
            arr1, arr2, compile_mode=dcor.CompileMode.NO_COMPILE)
        self.assertIsInstance(correlation, type_cor)
        self.assertAlmostEqual(correlation, type_cor(0.3066099), places=6)

        print(covariance, correlation)

        covariance = dcor.distance_covariance_sqr(
            arr1, arr1, compile_mode=dcor.CompileMode.NO_COMPILE)
        self.assertIsInstance(covariance, type_cov)
        self.assertAlmostEqual(covariance, type_cov(1.706791), places=5)

        correlation = dcor.distance_correlation_sqr(
            arr1, arr1, compile_mode=dcor.CompileMode.NO_COMPILE)
        self.assertIsInstance(correlation, type_cor)
        self.assertAlmostEqual(correlation, type_cor(1), places=5)

        print(covariance, correlation)

    def test_distance_correlation_vector(self):
        """Check distance correlation with vectors of float."""
        return self._test_distance_correlation_vector_generic(
            vector_type=float
        )

    def test_distance_correlation_vector_ints(self):
        """Check distance correlation with vectors of integers."""
        return self._test_distance_correlation_vector_generic(
            vector_type=int,
            type_cov=float,
            type_cor=float
        )

    def test_distance_correlation_vector_fractions(self):
        """
        Check distance correlation with vectors of fractions.

        Note that the covariance and correlation are given in floating
        point, as fractions can not generally represent a square root.
        """
        return self._test_distance_correlation_vector_generic(
            vector_type=Fraction,
            type_cor=float
        )

    def test_distance_correlation_vector_decimal(self):
        """Check distance correlation with vectors of Decimal."""
        return self._test_distance_correlation_vector_generic(
            vector_type=Decimal,
        )

    def test_statistic(self):
        """Test that the fast and naive algorithms for biased dcor match"""
        for seed in range(5):

            random_state = np.random.RandomState(seed)

            for i in range(4, self.test_max_size + 1):
                arr1 = random_state.rand(i, 1)
                arr2 = random_state.rand(i, 1)

                stat = dcor.distance_correlation_sqr(
                    arr1, arr2, method='naive')

                for method in dcor.DistanceCovarianceMethod:
                    with self.subTest(method=method):
                        stat2 = dcor.distance_correlation_sqr(
                            arr1, arr2, method=method)

                        self.assertAlmostEqual(stat, stat2)

    def test_u_statistic(self):
        """Test that the fast and naive algorithms for unbiased dcor match"""
        for seed in range(5):

            random_state = np.random.RandomState(seed)

            for i in range(4, self.test_max_size + 1):
                arr1 = random_state.rand(i, 1)
                arr2 = random_state.rand(i, 1)

                u_stat = dcor.u_distance_correlation_sqr(
                    arr1, arr2, method='naive')
                for method in dcor.DistanceCovarianceMethod:
                    with self.subTest(method=method):
                        u_stat2 = dcor.u_distance_correlation_sqr(
                            arr1, arr2, method=method)

                        self.assertAlmostEqual(u_stat, u_stat2)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()  # pragma: no cover

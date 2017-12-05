"""Tests of the dcor package"""

from decimal import Decimal
from fractions import Fraction
import re
import unittest

import dcor
import dcor._dcor as dcor_internals
import numpy as np


class TestVersion(unittest.TestCase):
    """Tests of the version number."""

    def test_version(self):
        """Test that the version has the right format."""
        regex = re.compile(r"\d+\.\d+(\.\d+)?")
        self.assertTrue(regex.match(dcor.__version__))
        self.assertNotEqual(dcor.__version__, "0.0")


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

        gamma = dcor_internals._dyad_update(y, c)
        expected_gamma = [0., 4., 9.]

        np.testing.assert_allclose(gamma, expected_gamma)

    def test_partial_sum_2d(self):  # pylint:disable=no-self-use
        """Compare partial sum results with the original code in the paper."""
        x = [1, 2, 3]
        y = [4, 5, 6]
        c = [7, 8, 9]

        gamma = dcor_internals._partial_sum_2d(x, y, c)
        expected_gamma = [17., 16., 15.]

        np.testing.assert_allclose(gamma, expected_gamma)

    def test_distance_correlation_naive(self):
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

    def test_distance_correlation_fast(self):
        """Compare fast distance correlation with the energy package."""
        arr1 = np.array(((1,), (2,), (3,), (4,), (5,), (6,)))
        arr2 = np.array(((1,), (7,), (5,), (5,), (6,), (2,)))

        covariance = dcor_internals._u_distance_covariance_sqr_fast(
            arr1, arr2)
        self.assertAlmostEqual(covariance, -0.88889, places=5)

        correlation = dcor_internals._u_distance_correlation_sqr_fast(
            arr1, arr2)
        self.assertAlmostEqual(correlation, -0.41613, places=5)

        covariance = dcor_internals._u_distance_covariance_sqr_fast(
            arr1, arr1)
        self.assertAlmostEqual(covariance, 1.5556, places=4)

        correlation = dcor_internals._u_distance_correlation_sqr_fast(
            arr1, arr1)
        self.assertAlmostEqual(correlation, 1, places=5)

    def test_u_distance_covariance_fast_overflow(self):
        """Test potential overflow in fast distance correlation"""
        arr1 = np.concatenate((np.zeros(500, dtype=int),
                               np.ones(500, dtype=int)))
        covariance = dcor_internals._u_distance_covariance_sqr_fast(arr1, arr1)
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
            arr1, arr2)
        self.assertIsInstance(covariance, type_cov)
        self.assertAlmostEqual(covariance, type_cov(-0.88889), places=5)

        correlation = dcor.u_distance_correlation_sqr(
            arr1, arr2)
        self.assertIsInstance(correlation, type_cor)
        self.assertAlmostEqual(correlation, type_cor(-0.41613), places=5)

        covariance = dcor.u_distance_covariance_sqr(
            arr1, arr1)
        self.assertIsInstance(covariance, type_cov)
        self.assertAlmostEqual(covariance, type_cov(1.5556), places=4)

        correlation = dcor.u_distance_correlation_sqr(
            arr1, arr1)
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
            arr1, arr2)
        self.assertIsInstance(covariance, type_cov)
        self.assertAlmostEqual(covariance, type_cov(0.6851851), places=6)

        correlation = dcor.distance_correlation_sqr(
            arr1, arr2)
        self.assertIsInstance(correlation, type_cor)
        self.assertAlmostEqual(correlation, type_cor(0.3066099), places=6)

        print(covariance, correlation)

        covariance = dcor.distance_covariance_sqr(
            arr1, arr1)
        self.assertIsInstance(covariance, type_cov)
        self.assertAlmostEqual(covariance, type_cov(1.706791), places=5)

        correlation = dcor.distance_correlation_sqr(
            arr1, arr1)
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
            type_cov=float,
            type_cor=float
        )

    def test_distance_correlation_vector_decimal(self):
        """Check distance correlation with vectors of Decimal."""
        return self._test_distance_correlation_vector_generic(
            vector_type=Decimal,
        )

    def test_u_statistic(self):
        """Test that the fast and naive algorithms for unbiased dcor match"""
        for seed in range(5):

            random_state = np.random.RandomState(seed)

            for i in range(4, self.test_max_size + 1):
                arr1 = random_state.rand(i, 1)
                arr2 = random_state.rand(i, 1)

                u_stat = dcor_internals._u_distance_correlation_sqr_naive(
                    arr1, arr2)
                u_stat_fast = dcor_internals._u_distance_correlation_sqr_fast(
                    arr1, arr2)

                self.assertAlmostEqual(u_stat, u_stat_fast)


class TestEnergyTest(unittest.TestCase):
    """Tests for the homogeneity energy test function."""

    def test_same_distribution_same_parameters(self):
        """
        Test that the test works on equal distributions.

        As the distributions are the same, the test should not reject
        the null hypothesis.

        """
        vector_size = 10
        num_samples = 100
        mean = np.zeros(vector_size)
        cov = np.eye(vector_size)

        a = np.random.RandomState(0).multivariate_normal(mean=mean,
                                                         cov=cov,
                                                         size=num_samples)
        b = np.random.RandomState(0).multivariate_normal(mean=mean,
                                                         cov=cov,
                                                         size=num_samples)

        significance = 0.01
        num_resamples = int(3 / significance + 1)

        random_state = np.random.RandomState(0)

        result = dcor.homogeneity.energy_test(a, b,
                                              num_resamples=num_resamples,
                                              random_state=random_state)

        self.assertGreater(result.p_value, significance)

    def test_same_distribution_different_means(self):
        """
        Test that the test works on distributions with different means.

        As the distributions are not the same, the test should reject
        the null hypothesis.

        """
        vector_size = 10
        num_samples = 100
        mean_0 = np.zeros(vector_size)
        mean_1 = np.ones(vector_size)
        cov = np.eye(vector_size)

        a = np.random.RandomState(0).multivariate_normal(mean=mean_0, cov=cov,
                                                         size=num_samples)
        b = np.random.RandomState(0).multivariate_normal(mean=mean_1, cov=cov,
                                                         size=num_samples)

        significance = 0.01
        num_resamples = int(3 / significance + 1)

        result = dcor.homogeneity.energy_test(a, b,
                                              num_resamples=num_resamples,
                                              random_state=0)

        self.assertLess(result.p_value, significance)

    def test_same_distribution_different_covariances(self):
        """
        Test that the test works on distributions with different covariance.

        As the distributions are not the same, the test should reject
        the null hypothesis.

        """
        vector_size = 10
        num_samples = 100
        mean = np.zeros(vector_size)
        cov_0 = np.eye(vector_size)
        cov_1 = 3 * np.eye(vector_size)

        a = np.random.RandomState(0).multivariate_normal(mean=mean, cov=cov_0,
                                                         size=num_samples)
        b = np.random.RandomState(0).multivariate_normal(mean=mean, cov=cov_1,
                                                         size=num_samples)

        significance = 0.01
        num_resamples = int(3 / significance + 1)

        result = dcor.homogeneity.energy_test(a, b,
                                              num_resamples=num_resamples,
                                              random_state=0)

        self.assertLess(result.p_value, significance)

    def test_different_distributions(self):
        """
        Test that the test works on different distributions.

        As the distributions are not the same, the test should reject
        the null hypothesis.

        """
        num_samples = 100

        a = np.random.RandomState(0).standard_normal(size=(num_samples, 1))
        b = np.random.RandomState(0).standard_t(df=1, size=(num_samples, 1))

        significance = 0.01
        num_resamples = int(3 / significance + 1)

        result = dcor.homogeneity.energy_test(a, b,
                                              num_resamples=num_resamples,
                                              random_state=0)

        self.assertLess(result.p_value, significance)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()  # pragma: no cover

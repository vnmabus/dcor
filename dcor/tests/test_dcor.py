import re
import unittest

import dcor
import dcor.dcor as dcor_internals
import numpy as np


class TestVersion(unittest.TestCase):

    def test_version(self):
        regex = re.compile("\d+\.\d+(\.\d+)?")
        self.assertTrue(regex.match(dcor.__version__))
        self.assertNotEqual(dcor.__version__, "0.0")


class TestDistanceCorrelation(unittest.TestCase):

    def setUp(self):
        self.test_max_size = 10

    def tearDown(self):
        pass

    def test_double_centered(self):
        for i in range(1, self.test_max_size + 1):
            matrix = np.random.RandomState(i).rand(i, i)

            centered_matrix = dcor.double_centered(matrix)

            column_sum = np.sum(centered_matrix, 0)
            row_sum = np.sum(centered_matrix, 1)

            np.testing.assert_allclose(column_sum, np.zeros(i), atol=1e-8)
            np.testing.assert_allclose(row_sum, np.zeros(i), atol=1e-8)

    def test_average_product(self):
        size = 5

        matrix1 = np.random.RandomState(0).rand(size, size)
        matrix2 = np.zeros((size, size))

        result = dcor.average_product(matrix1, matrix2)

        self.assertAlmostEqual(result, 0)

    def test_dyad_update(self):
        Y = np.array([1, 2, 3])
        C = np.array([4, 5, 6])

        gamma = dcor_internals._dyad_update(Y, C)
        expected_gamma = [0., 4., 9.]

        np.testing.assert_allclose(gamma, expected_gamma)

    def test_partial_sum_2d(self):
        X = [1, 2, 3]
        Y = [4, 5, 6]
        C = [7, 8, 9]

        gamma = dcor_internals._partial_sum_2d(X, Y, C)
        expected_gamma = [17., 16., 15.]

        np.testing.assert_allclose(gamma, expected_gamma)

    def test_distance_correlation_naive(self):
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
        pass
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

    def test_distance_correlation_vector(self):
        pass
        arr1 = np.array((1, 2, 3, 4, 5, 6))
        arr2 = np.array((1, 7, 5, 5, 6, 2))

        covariance = dcor_internals.u_distance_covariance_sqr(
            arr1, arr2)
        self.assertAlmostEqual(covariance, -0.88889, places=5)

        correlation = dcor_internals.u_distance_correlation_sqr(
            arr1, arr2)
        self.assertAlmostEqual(correlation, -0.41613, places=5)

        covariance = dcor_internals.u_distance_covariance_sqr(
            arr1, arr1)
        self.assertAlmostEqual(covariance, 1.5556, places=4)

        correlation = dcor_internals.u_distance_correlation_sqr(
            arr1, arr1)
        self.assertAlmostEqual(correlation, 1, places=5)

    def test_u_statistic(self):

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

    def test_same_distribution_same_parameters(self):

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

        result = dcor.homogeneity.energy_test(a, b,
                                              num_resamples=num_resamples,
                                              random_state=0)

        self.assertGreater(result.p_value, significance)

    def test_same_distribution_different_means(self):

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
    unittest.main()

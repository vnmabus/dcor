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
            matrix = np.random.rand(i, i)

            centered_matrix = dcor.double_centered(matrix)

            column_sum = np.sum(centered_matrix, 0)
            row_sum = np.sum(centered_matrix, 1)

            np.testing.assert_allclose(column_sum, np.zeros(i), atol=1e-8)
            np.testing.assert_allclose(row_sum, np.zeros(i), atol=1e-8)

    def test_average_product(self):
        size = 5

        matrix1 = np.random.rand(size, size)
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


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()

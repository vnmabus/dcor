'''
Created on 10 de ene. de 2017

@author: Carlos Ramos Carreño
'''
import unittest

import dcor.dcor as dc
import numpy as np


class TestDistanceCorrelation(unittest.TestCase):

    def setUp(self):
        self.test_max_size = 10

    def tearDown(self):
        pass

    def test_double_centered(self):
        for i in range(1, self.test_max_size + 1):
            matrix = np.random.rand(i, i)

            centered_matrix = dc.double_centered(matrix)

            column_sum = np.sum(centered_matrix, 0)
            row_sum = np.sum(centered_matrix, 1)

            np.testing.assert_allclose(column_sum, np.zeros(i), atol=1e-8)
            np.testing.assert_allclose(row_sum, np.zeros(i), atol=1e-8)

    def test_average_product(self):
        size = 5

        matrix1 = np.random.rand(size, size)
        matrix2 = np.zeros((size, size))

        result = dc.average_product(matrix1, matrix2)

        self.assertAlmostEqual(result, 0)

    def test_dyad_update(self):
        Y = [1, 2, 3]
        C = [4, 5, 6]

        gamma = dc._dyad_update(Y, C)
        expected_gamma = [0., 4., 9.]

        np.testing.assert_allclose(gamma, expected_gamma)

    def test_partial_sum_2d(self):
        X = [1, 2, 3]
        Y = [4, 5, 6]
        C = [7, 8, 9]

        gamma = dc._partial_sum_2d(X, Y, C)
        expected_gamma = [17., 16., 15.]

        np.testing.assert_allclose(gamma, expected_gamma)

    def test_distance_correlation_naive(self):
        matrix1 = np.array(((1, 2, 3), (4, 5, 6), (7, 8, 9)))
        matrix2 = np.array(((7, 3, 6), (2, 1, 4), (3, 8, 1)))
        matrix3 = np.array(((1, 1, 1), (2, 1, 1), (1, 1, 1)))
        constant_matrix = np.ones((3, 3))

        correlation = dc.distance_correlation_sqr(
            matrix1, matrix1)
        self.assertAlmostEqual(correlation, 1)

        correlation = dc.distance_correlation_sqr(
            matrix1, constant_matrix)
        self.assertAlmostEqual(correlation, 0)

        correlation = dc.distance_correlation_sqr(
            matrix1, matrix2)
        self.assertAlmostEqual(correlation, 0.93387, places=5)

        correlation = dc.distance_correlation_sqr(
            matrix1, matrix3)
        self.assertAlmostEqual(correlation, 0.31623, places=5)

    def test_distance_correlation_fast(self):
        pass
        arr1 = np.array(((1,), (2,), (3,), (4,), (5,), (6,)))
        arr2 = np.array(((1,), (7,), (5,), (5,), (6,), (2,)))

        covariance = dc._u_distance_covariance_sqr_fast(
            arr1, arr2)
        self.assertAlmostEqual(covariance, -0.88889, places=5)

        correlation = dc._u_distance_correlation_sqr_fast(
            arr1, arr2)
        self.assertAlmostEqual(correlation, -0.41613, places=5)

        covariance = dc._u_distance_covariance_sqr_fast(
            arr1, arr1)
        self.assertAlmostEqual(covariance, 1.5556, places=4)

        correlation = dc._u_distance_correlation_sqr_fast(
            arr1, arr1)
        self.assertAlmostEqual(correlation, 1, places=5)

    def test_u_statistic(self):
        for i in range(4, self.test_max_size + 1):
            arr1 = np.random.rand(i, 1)
            arr2 = np.random.rand(i, 1)

            u_stat = dc._u_distance_correlation_sqr_naive(arr1, arr2)
            u_stat_fast = dc._u_distance_correlation_sqr_fast(arr1, arr2)

            self.assertAlmostEqual(u_stat, u_stat_fast)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()

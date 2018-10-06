"""Tests of the homogeneity module"""

import unittest

import dcor
import numpy as np


class TestDcovTest(unittest.TestCase):
    """Tests for the independence distance covariance test function."""

    def test_independent_variables(self):
        """
        Test that the test works on independent variables.

        As the variables are independent, the test should not reject
        the null hypothesis.

        """
        vector_size = 10
        num_samples = 100
        mean = np.zeros(vector_size)
        cov = np.eye(vector_size)

        random_state = np.random.RandomState(0)

        a = random_state.multivariate_normal(mean=mean,
                                             cov=cov,
                                             size=num_samples)
        b = random_state.multivariate_normal(mean=mean,
                                             cov=cov,
                                             size=num_samples)

        significance = 0.01
        num_resamples = int(3 / significance + 1)

        result = dcor.independence.distance_covariance_test(
            a, b, num_resamples=num_resamples, random_state=random_state)

        self.assertGreater(result.p_value, significance)

    def test_same_variable(self):
        """
        Test that the test works on the same variable.

        As the two variables are the same, the test should reject
        the null hypothesis.

        """
        vector_size = 10
        num_samples = 100
        mean = np.zeros(vector_size)
        cov = np.eye(vector_size)

        random_state = np.random.RandomState(0)

        a = random_state.multivariate_normal(mean=mean,
                                             cov=cov,
                                             size=num_samples)
        b = a

        significance = 0.01
        num_resamples = int(3 / significance + 1)

        result = dcor.independence.distance_covariance_test(
            a, b, num_resamples=num_resamples, random_state=random_state)

        self.assertLess(result.p_value, significance)

    def test_function_variable(self):
        """
        Test that the test works when the second variable is a function of the
        first one.

        As the two variables are dependent, the test should reject
        the null hypothesis.

        """
        vector_size = 10
        num_samples = 100
        mean = np.zeros(vector_size)
        cov = np.eye(vector_size)

        random_state = np.random.RandomState(0)

        a = random_state.multivariate_normal(mean=mean,
                                             cov=cov,
                                             size=num_samples)
        b = np.sin(a) + 3 * a ** 2

        significance = 0.01
        num_resamples = int(3 / significance + 1)

        result = dcor.independence.distance_covariance_test(
            a, b, num_resamples=num_resamples, random_state=random_state)

        self.assertLess(result.p_value, significance)

    def test_dependent_variables(self):
        """
        Test that the test works on dependent variables.

        As the variables are dependent, the test should reject
        the null hypothesis.

        """
        vector_size = 10
        num_samples = 100
        mean = np.zeros(vector_size)
        cov = np.eye(vector_size)

        random_state = np.random.RandomState(0)

        a = random_state.multivariate_normal(mean=mean,
                                             cov=cov,
                                             size=num_samples)
        b = random_state.multivariate_normal(mean=mean,
                                             cov=cov,
                                             size=num_samples) + np.sin(a)

        significance = 0.01
        num_resamples = int(3 / significance + 1)

        result = dcor.independence.distance_covariance_test(
            a, b, num_resamples=num_resamples, random_state=random_state)

        self.assertLess(result.p_value, significance)
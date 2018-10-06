"""Tests of the homogeneity module"""

import unittest

import dcor
import numpy as np


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

        random_state = np.random.RandomState(0)

        a = random_state.multivariate_normal(mean=mean,
                                             cov=cov,
                                             size=num_samples)
        b = random_state.multivariate_normal(mean=mean,
                                             cov=cov,
                                             size=num_samples)

        significance = 0.01
        num_resamples = int(3 / significance + 1)

        result = dcor.homogeneity.energy_test(
            a, b, num_resamples=num_resamples, random_state=random_state)

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

        random_state = np.random.RandomState(0)

        a = random_state.multivariate_normal(mean=mean_0, cov=cov,
                                             size=num_samples)
        b = random_state.multivariate_normal(mean=mean_1, cov=cov,
                                             size=num_samples)

        significance = 0.01
        num_resamples = int(3 / significance + 1)

        result = dcor.homogeneity.energy_test(
            a, b, num_resamples=num_resamples, random_state=random_state)

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

        random_state = np.random.RandomState(0)

        a = random_state.multivariate_normal(mean=mean, cov=cov_0,
                                             size=num_samples)
        b = random_state.multivariate_normal(mean=mean, cov=cov_1,
                                             size=num_samples)

        significance = 0.01
        num_resamples = int(3 / significance + 1)

        result = dcor.homogeneity.energy_test(
            a, b, num_resamples=num_resamples, random_state=random_state)

        self.assertLess(result.p_value, significance)

    def test_different_distributions(self):
        """
        Test that the test works on different distributions.

        As the distributions are not the same, the test should reject
        the null hypothesis.

        """
        num_samples = 100

        random_state = np.random.RandomState(0)

        a = random_state.standard_normal(size=(num_samples, 1))
        b = random_state.standard_t(df=1, size=(num_samples, 1))

        significance = 0.01
        num_resamples = int(3 / significance + 1)

        result = dcor.homogeneity.energy_test(
            a, b, num_resamples=num_resamples, random_state=random_state)

        self.assertLess(result.p_value, significance)

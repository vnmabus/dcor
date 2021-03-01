"""Tests of the homogeneity module"""

import time
import unittest
import subprocess
import sys
from pathlib import Path
import os

import numpy as np

import dcor


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

        np.random.seed(0)

        a = np.random.multivariate_normal(
            mean=mean,
            cov=cov,
            size=num_samples
        )
        b = np.random.multivariate_normal(
            mean=mean,
            cov=cov,
            size=num_samples
        )

        significance = 0.01
        num_resamples = int(3 / significance + 1)

        result = dcor.homogeneity.energy_test(
            a,
            b,
            num_resamples=num_resamples
        )

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

        np.random.seed(0)

        a = np.random.multivariate_normal(mean=mean_0, cov=cov,
                                          size=num_samples)
        b = np.random.multivariate_normal(mean=mean_1, cov=cov,
                                          size=num_samples)

        significance = 0.01
        num_resamples = int(3 / significance + 1)

        result = dcor.homogeneity.energy_test(
            a, b, num_resamples=num_resamples)

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

        np.random.seed(0)

        a = np.random.multivariate_normal(mean=mean, cov=cov_0,
                                          size=num_samples)
        b = np.random.multivariate_normal(mean=mean, cov=cov_1,
                                          size=num_samples)

        significance = 0.01
        num_resamples = int(3 / significance + 1)

        result = dcor.homogeneity.energy_test(
            a, b, num_resamples=num_resamples)

        self.assertLess(result.p_value, significance)

    def test_different_distributions(self):
        """
        Test that the test works on different distributions.

        As the distributions are not the same, the test should reject
        the null hypothesis.

        """
        num_samples = 100

        np.random.seed(0)

        a = np.random.standard_normal(size=(num_samples, 1))
        b = np.random.standard_t(df=1, size=(num_samples, 1))

        significance = 0.01
        num_resamples = int(3 / significance + 1)

        result = dcor.homogeneity.energy_test(
            a,
            b,
            num_resamples=num_resamples
        )

        self.assertLess(result.p_value, significance)

    def test_different_means_median(self):
        """
        Test that the test works on the same distribution with different means,
        using the median average.
        """
        num_samples = 100

        np.random.seed(0)

        a = np.random.normal(loc=0, size=(num_samples, 1))
        b = np.random.normal(loc=1, size=(num_samples, 1))

        significance = 0.01
        num_resamples = int(3 / significance + 1)

        median_result = dcor.homogeneity.energy_test(
            a,
            b,
            num_resamples=num_resamples,
            average='median'
        )

        mean_result = dcor.homogeneity.energy_test(
            a,
            b,
            num_resamples=num_resamples,
            average='mean'
        )

        # Check that we are actually using a different average
        self.assertNotAlmostEqual(
            mean_result.statistic,
            median_result.statistic
        )

        # Check that we detected the heterogeneity
        self.assertLess(median_result.p_value, significance)

    def test_different_distributions_median(self):
        """
        Test that the test works on different distributions using the median.
        """
        num_samples = 100

        np.random.seed(0)

        a = np.random.normal(loc=1, size=(num_samples, 1))
        b = np.random.exponential(scale=1, size=(num_samples, 1))

        significance = 0.01
        num_resamples = int(3 / significance + 1)

        result = dcor.homogeneity.energy_test(
            a,
            b,
            average='median',
            num_resamples=num_resamples
        )

        self.assertLess(result.p_value, significance)

    def test_speed_permutation(self):
        """
        Run the permutation with and without numba, and check that numba is
        indeed providing a speedup
        """
        start = time.time()
        here = Path(__file__).parent
        kwargs = dict(
            args=[
                sys.executable,
                str(here / 'speed_permutation.py')
            ],
            capture_output=True,
            check=True,
            cwd=here
        )
        numba_result = subprocess.run(**kwargs)
        mid = time.time()
        no_numba_result = subprocess.run(
            **kwargs,
            env={
                'NUMBA_DISABLE_JIT': '1',
                **os.environ
            }
        )
        end = time.time()

        self.assertEqual(
            numba_result.stdout,
            no_numba_result.stdout,
            msg='Numba JIT produced a different result to plain Python'
        )
        self.assertLess(
            10 * (mid - start),
            end - mid,
            msg='Numba JIT is not at least 10 times faster than plain Python'
        )

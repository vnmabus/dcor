"""Tests of the homogeneity module"""

import unittest

import array_api_strict
import numpy as np

import dcor


class TestEnergyTest(unittest.TestCase):
    """Tests for the homogeneity energy test function."""

    def test_same_distribution_same_parameters(self) -> None:
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

        a = random_state.multivariate_normal(
            mean=mean,
            cov=cov,
            size=num_samples,
        )
        b = random_state.multivariate_normal(
            mean=mean,
            cov=cov,
            size=num_samples,
        )

        significance = 0.01
        num_resamples = int(3 / significance + 1)

        result = dcor.homogeneity.energy_test(
            a,
            b,
            num_resamples=num_resamples,
            random_state=random_state,
        )

        self.assertGreater(result.pvalue, significance)

    def test_same_distribution_different_means(self) -> None:
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

        a = random_state.multivariate_normal(
            mean=mean_0,
            cov=cov,
            size=num_samples,
        )
        b = random_state.multivariate_normal(
            mean=mean_1,
            cov=cov,
            size=num_samples,
        )

        significance = 0.01
        num_resamples = int(3 / significance + 1)

        result = dcor.homogeneity.energy_test(
            a,
            b,
            num_resamples=num_resamples,
            random_state=random_state,
        )

        self.assertLess(result.pvalue, significance)

    def test_same_distribution_different_covariances(self) -> None:
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

        a = random_state.multivariate_normal(
            mean=mean,
            cov=cov_0,
            size=num_samples,
        )
        b = random_state.multivariate_normal(
            mean=mean,
            cov=cov_1,
            size=num_samples,
        )

        significance = 0.01
        num_resamples = int(3 / significance + 1)

        result = dcor.homogeneity.energy_test(
            a,
            b,
            num_resamples=num_resamples,
            random_state=random_state,
        )

        self.assertLess(result.pvalue, significance)

    def test_different_distributions(self) -> None:
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
            a,
            b,
            num_resamples=num_resamples,
            random_state=random_state,
        )

        self.assertLess(result.pvalue, significance)

    def test_different_means_median(self) -> None:
        """Check test works with different means, using the median average."""
        num_samples = 100

        random_state = np.random.RandomState(0)

        a = random_state.normal(loc=0, size=(num_samples, 1))
        b = random_state.normal(loc=1, size=(num_samples, 1))

        significance = 0.01
        num_resamples = int(3 / significance + 1)

        median_result = dcor.homogeneity.energy_test(
            a,
            b,
            num_resamples=num_resamples,
            random_state=random_state,
            average=np.median,
        )

        mean_result = dcor.homogeneity.energy_test(
            a,
            b,
            num_resamples=num_resamples,
            random_state=random_state,
            average=np.mean,
        )

        # Check that we are actually using a different average
        self.assertNotAlmostEqual(
            float(mean_result.statistic),
            float(median_result.statistic),
        )

        # Check that we detected the heterogeneity
        self.assertLess(median_result.pvalue, significance)

    def test_different_distributions_median(self) -> None:
        """Check test works on different distributions using the median."""
        num_samples = 100

        random_state = np.random.RandomState(0)

        a = random_state.normal(loc=1, size=(num_samples, 1))
        b = random_state.exponential(scale=1, size=(num_samples, 1))

        significance = 0.01
        num_resamples = int(3 / significance + 1)

        result = dcor.homogeneity.energy_test(
            a,
            b,
            average=np.median,
            num_resamples=num_resamples,
            random_state=random_state,
        )

        self.assertLess(result.pvalue, significance)


class TestEnergyArrayAPI(unittest.TestCase):
    """Check energy distance test works with the Array API standard."""

    def test_same_distribution_same_parameters(self) -> None:
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

        a = array_api_strict.asarray(
            random_state.multivariate_normal(
                mean=mean,
                cov=cov,
                size=num_samples,
            ),
        )

        b = array_api_strict.asarray(
            random_state.multivariate_normal(
                mean=mean,
                cov=cov,
                size=num_samples,
            ),
        )

        significance = 0.01
        num_resamples = int(3 / significance + 1)

        result = dcor.homogeneity.energy_test(
            a,
            b,
            num_resamples=num_resamples,
            random_state=random_state,
        )

        self.assertIsInstance(result.pvalue, float)
        self.assertIsInstance(result.statistic, type(a))

        self.assertGreater(result.pvalue, significance)

    def test_same_distribution_different_means(self) -> None:
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

        a = array_api_strict.asarray(
            random_state.multivariate_normal(
                mean=mean_0,
                cov=cov,
                size=num_samples,
            ),
        )
        b = array_api_strict.asarray(
            random_state.multivariate_normal(
                mean=mean_1,
                cov=cov,
                size=num_samples,
            ),
        )

        significance = 0.01
        num_resamples = int(3 / significance + 1)

        result = dcor.homogeneity.energy_test(
            a,
            b,
            num_resamples=num_resamples,
            random_state=random_state,
        )

        self.assertIsInstance(result.pvalue, float)
        self.assertIsInstance(result.statistic, type(a))

        self.assertLess(result.pvalue, significance)

"""Tests of the distance covariance and correlation."""
from __future__ import annotations

import math
import unittest
from decimal import Decimal
from fractions import Fraction
from typing import Any, Callable, Tuple, Type, TypeVar

import array_api_strict
import numpy as np

import dcor
from dcor._fast_dcov_avl import _dyad_update

T = TypeVar("T")


class TestDistanceCorrelation(unittest.TestCase):
    """Distance correlation tests."""

    def setUp(self) -> None:
        """Set the common parameters."""
        self.test_max_size = 10

    def test_double_centered(self) -> None:
        """
        Test that the double centering is right.

        Check that the sum of the rows and colums of a double centered
        matrix is 0.

        """
        for i in range(1, self.test_max_size + 1):
            matrix = np.random.RandomState(i).rand(i, i)
            matrix = matrix @ matrix.T

            centered_matrix = dcor.double_centered(matrix)

            column_sum = np.sum(centered_matrix, 0)
            row_sum = np.sum(centered_matrix, 1)

            np.testing.assert_allclose(column_sum, np.zeros(i), atol=1e-8)
            np.testing.assert_allclose(row_sum, np.zeros(i), atol=1e-8)

    def test_dyad_update(self) -> None:  # pylint:disable=no-self-use
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

        gamma = _dyad_update(
            y, c, gamma1, l_max, s, pos_sums)
        expected_gamma = [0, 4, 9]

        np.testing.assert_allclose(gamma, expected_gamma)

    def test_distance_correlation_multivariate(self) -> None:
        """Compare distance correlation with the energy package."""
        matrix1 = np.array(((1, 2, 3), (4, 5, 6), (7, 8, 9)))
        matrix2 = np.array(((7, 3, 6), (2, 1, 4), (3, 8, 1)))
        matrix3 = np.array(((1, 1, 1), (2, 1, 1), (1, 1, 1)))
        constant_matrix = np.ones((3, 3))

        correlation = dcor.distance_correlation_sqr(
            matrix1,
            matrix1,
        )
        self.assertAlmostEqual(float(correlation), 1)

        correlation = dcor.distance_correlation_sqr(
            matrix1,
            constant_matrix,
        )
        self.assertAlmostEqual(float(correlation), 0)

        correlation = dcor.distance_correlation_sqr(
            matrix1,
            matrix2,
        )
        self.assertAlmostEqual(float(correlation), 0.93387, places=5)

        correlation = dcor.distance_correlation_sqr(
            matrix1,
            matrix3,
        )
        self.assertAlmostEqual(float(correlation), 0.31623, places=5)

    def test_distance_correlation_comparison(self) -> None:
        """Compare all implementations of distance covariance/correlation."""
        arr1 = np.array(((1.0,), (2.0,), (3.0,), (4.0,), (5.0,), (6.0,)))
        arr2 = np.array(((1.0,), (7.0,), (5.0,), (5.0,), (6.0,), (2.0,)))

        for method in dcor.DistanceCovarianceMethod:

            compile_modes = [
                dcor.CompileMode.AUTO,
                dcor.CompileMode.NO_COMPILE,
            ]

            if method is not dcor.DistanceCovarianceMethod.NAIVE:
                compile_modes += [dcor.CompileMode.COMPILE_CPU]

            for compile_mode in compile_modes:
                with self.subTest(method=method, compile_mode=compile_mode):

                    # Unbiased versions

                    u_covariance = dcor.u_distance_covariance_sqr(
                        arr1,
                        arr2,
                        method=method,
                        compile_mode=compile_mode,
                    )
                    self.assertAlmostEqual(
                        float(u_covariance),
                        -0.88889,
                        places=5,
                    )

                    u_correlation = dcor.u_distance_correlation_sqr(
                        arr1,
                        arr2,
                        method=method,
                        compile_mode=compile_mode,
                    )
                    self.assertAlmostEqual(
                        float(u_correlation),
                        -0.41613,
                        places=5,
                    )

                    u_covariance = dcor.u_distance_covariance_sqr(
                        arr1,
                        arr1,
                        method=method,
                        compile_mode=compile_mode,
                    )
                    self.assertAlmostEqual(
                        float(u_covariance),
                        1.55556,
                        places=5,
                    )

                    u_correlation = dcor.u_distance_correlation_sqr(
                        arr1,
                        arr1,
                        method=method,
                        compile_mode=compile_mode,
                    )
                    self.assertAlmostEqual(
                        float(u_correlation),
                        1,
                        places=5,
                    )

                    u_covariance = dcor.u_distance_covariance_sqr(
                        arr2,
                        arr2,
                        method=method,
                        compile_mode=compile_mode,
                    )
                    self.assertAlmostEqual(
                        float(u_covariance),
                        2.93333,
                        places=5,
                    )

                    u_correlation = dcor.u_distance_correlation_sqr(
                        arr2,
                        arr2,
                        method=method,
                        compile_mode=compile_mode,
                    )
                    self.assertAlmostEqual(
                        float(u_correlation),
                        1,
                        places=5,
                    )

                    u_stats = dcor.u_distance_stats_sqr(
                        arr1,
                        arr2,
                        method=method,
                        compile_mode=compile_mode,
                    )
                    np.testing.assert_allclose(
                        tuple(u_stats),
                        (-0.88889, -0.41613, 1.55556, 2.93333),
                        rtol=1e-4,
                    )

                    # Biased

                    covariance = dcor.distance_covariance_sqr(
                        arr1,
                        arr2,
                        method=method,
                        compile_mode=compile_mode,
                    )
                    self.assertAlmostEqual(
                        float(covariance),
                        0.68519,
                        places=5,
                    )

                    correlation = dcor.distance_correlation_sqr(
                        arr1,
                        arr2,
                        method=method,
                        compile_mode=compile_mode,
                    )
                    self.assertAlmostEqual(
                        float(correlation),
                        0.30661,
                        places=5,
                    )

                    covariance = dcor.distance_covariance_sqr(
                        arr1,
                        arr1,
                        method=method,
                        compile_mode=compile_mode,
                    )
                    self.assertAlmostEqual(
                        float(covariance),
                        1.70679,
                        places=5,
                    )

                    correlation = dcor.distance_correlation_sqr(
                        arr1,
                        arr1,
                        method=method,
                        compile_mode=compile_mode,
                    )
                    self.assertAlmostEqual(
                        float(correlation),
                        1,
                        places=5,
                    )

                    covariance = dcor.distance_covariance_sqr(
                        arr2,
                        arr2,
                        method=method,
                        compile_mode=compile_mode,
                    )
                    self.assertAlmostEqual(
                        float(covariance),
                        2.92593,
                        places=5,
                    )

                    correlation = dcor.distance_correlation_sqr(
                        arr2,
                        arr2,
                        method=method,
                        compile_mode=compile_mode,
                    )
                    self.assertAlmostEqual(
                        float(correlation),
                        1,
                        places=5,
                    )

                    stats = dcor.distance_stats_sqr(
                        arr1,
                        arr2,
                        method=method,
                        compile_mode=compile_mode,
                    )
                    np.testing.assert_allclose(
                        tuple(stats),
                        (0.68519, 0.30661, 1.70679, 2.92593),
                        rtol=1e-4,
                    )

    def test_u_distance_covariance_avl_overflow(self) -> None:
        """Test potential overflow in fast distance correlation."""
        arr1 = np.concatenate((
            np.zeros(500, dtype=int),
            np.ones(500, dtype=int),
        ))
        covariance = dcor.u_distance_covariance_sqr(
            arr1,
            arr1,
            method='avl',
            compile_mode=dcor.CompileMode.NO_COMPILE,
        )
        self.assertAlmostEqual(covariance, 0.25050, places=5)

    def _test_u_distance_correlation_vector_generic(
        self,
        vector_type: Type[Any],
        type_cov: Type[Any] | None = None,
        type_cor: Type[Any] | None = None,
    ) -> None:
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

        arr1 = np.array([vector_type(i) for i in range(1, 7)])
        arr2 = np.array([vector_type(i) for i in (1, 7, 5, 5, 6, 2)])

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

    def test_u_distance_correlation_vector(self) -> None:
        """Check U-distance with vectors of float."""
        return self._test_u_distance_correlation_vector_generic(
            vector_type=float,
        )

    def test_u_distance_correlation_vector_ints(self) -> None:
        """Check U-distance with vectors of integers."""
        return self._test_u_distance_correlation_vector_generic(
            vector_type=int,
            type_cov=float,
            type_cor=float,
        )

    def test_u_distance_correlation_vector_fractions(self) -> None:
        """
        Check U-distance with vectors of fractions.

        Note that the correlation is given in floating point, as
        fractions can not generally represent a square root.
        """
        return self._test_u_distance_correlation_vector_generic(
            vector_type=Fraction,
            type_cor=float,
        )

    def test_u_distance_correlation_vector_decimal(self) -> None:
        """Check U-distance with vectors of Decimal."""
        return self._test_u_distance_correlation_vector_generic(
            vector_type=Decimal,
        )

    def _test_distance_correlation_vector_generic(
        self,
        vector_type: Type[Any],
        type_cov: Type[Any] | None = None,
        type_cor: Type[Any] | None = None,
    ) -> None:
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

        arr1 = np.array([vector_type(i) for i in range(1, 7)])
        arr2 = np.array([vector_type(i) for i in (1, 7, 5, 5, 6, 2)])

        covariance = dcor.distance_covariance_sqr(
            arr1,
            arr2,
            compile_mode=dcor.CompileMode.NO_COMPILE,
        )
        self.assertIsInstance(covariance, type_cov)
        self.assertAlmostEqual(covariance, type_cov(0.6851851), places=6)

        correlation = dcor.distance_correlation_sqr(
            arr1,
            arr2,
            compile_mode=dcor.CompileMode.NO_COMPILE,
        )
        self.assertIsInstance(correlation, type_cor)
        self.assertAlmostEqual(correlation, type_cor(0.3066099), places=6)

        covariance = dcor.distance_covariance_sqr(
            arr1,
            arr1,
            compile_mode=dcor.CompileMode.NO_COMPILE,
        )
        self.assertIsInstance(covariance, type_cov)
        self.assertAlmostEqual(covariance, type_cov(1.706791), places=5)

        correlation = dcor.distance_correlation_sqr(
            arr1,
            arr1,
            compile_mode=dcor.CompileMode.NO_COMPILE,
        )
        self.assertIsInstance(correlation, type_cor)
        self.assertAlmostEqual(correlation, type_cor(1), places=5)

    def test_distance_correlation_vector(self) -> None:
        """Check distance correlation with vectors of float."""
        return self._test_distance_correlation_vector_generic(
            vector_type=float,
        )

    def test_distance_correlation_vector_ints(self) -> None:
        """Check distance correlation with vectors of integers."""
        return self._test_distance_correlation_vector_generic(
            vector_type=int,
            type_cov=float,
            type_cor=float,
        )

    def test_distance_correlation_vector_fractions(self) -> None:
        """
        Check distance correlation with vectors of fractions.

        Note that the covariance and correlation are given in floating
        point, as fractions can not generally represent a square root.
        """
        return self._test_distance_correlation_vector_generic(
            vector_type=Fraction,
            type_cor=float,
        )

    def test_distance_correlation_vector_decimal(self) -> None:
        """Check distance correlation with vectors of Decimal."""
        return self._test_distance_correlation_vector_generic(
            vector_type=Decimal,
        )

    def _test_fast_naive_generic(
        self,
        function: Callable[..., np.typing.NDArray[float]],
    ) -> None:
        """Test that the fast and naive algorithms match."""
        for seed in range(5):

            random_state = np.random.RandomState(seed)

            for i in range(4, self.test_max_size + 1):
                arr1 = random_state.rand(i, 1)
                arr2 = random_state.rand(i, 1)

                stat = function(
                    arr1,
                    arr2,
                    method='naive',
                )

                for method in dcor.DistanceCovarianceMethod:
                    stat2 = function(
                        arr1,
                        arr2,
                        method=method,
                    )

                    self.assertAlmostEqual(float(stat), float(stat2))

    def test_fast_naive_statistic(self) -> None:
        """Test that the fast and naive algorithms for biased dcor match."""
        self._test_fast_naive_generic(dcor.distance_correlation_sqr)

    def test_u_statistic(self) -> None:
        """Test that the fast and naive algorithms for unbiased dcor match."""
        self._test_fast_naive_generic(dcor.u_distance_correlation_sqr)

    def test_dcor_constant(self) -> None:
        """Test that it works with constant random variables."""
        a = np.ones(100)

        cov = dcor.distance_covariance(a, a)
        self.assertAlmostEqual(cov, 0)

        corr = dcor.distance_correlation(a, a)
        self.assertAlmostEqual(corr, 0)

        corr_af_inv = dcor.distance_correlation_af_inv(a, a)
        self.assertAlmostEqual(corr_af_inv, 0)

    def test_integer_overflow(self) -> None:
        """Tests int overflow behavior detected in issue #59."""
        n_samples = 10000

        # some simple data
        arr1 = np.array([1, 2, 3] * n_samples, dtype=np.int64)
        arr2 = np.array([10, 20, 5] * n_samples, dtype=np.int64)

        int_int = dcor.distance_correlation(
            arr1,
            arr2,
            compile_mode=dcor.CompileMode.NO_COMPILE,
        )
        float_int = dcor.distance_correlation(
            arr1.astype(float),
            arr2,
            compile_mode=dcor.CompileMode.NO_COMPILE,
        )
        int_float = dcor.distance_correlation(
            arr1,
            arr2.astype(float),
            compile_mode=dcor.CompileMode.NO_COMPILE,
        )
        float_float = dcor.distance_correlation(
            arr1.astype(float),
            arr2.astype(float),
        )

        self.assertAlmostEqual(int_int, float_float)
        self.assertAlmostEqual(float_int, float_float)
        self.assertAlmostEqual(int_float, float_float)


class TestDcorArrayAPI(unittest.TestCase):
    """Check that the energy distance works with the Array API standard."""

    def setUp(self) -> None:
        """Initialize Array API arrays."""
        self.a = array_api_strict.asarray(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16],
            ],
            dtype=array_api_strict.float64,
        )
        self.b = array_api_strict.asarray(
            [
                [1],
                [0],
                [0],
                [1],
            ],
            dtype=array_api_strict.float64,
        )

    def _test_generic(
        self,
        function: Callable[[T, T], T],
        results: Tuple[float, float, float],
    ) -> None:
        """Check dcor functions."""
        fun_aa = function(self.a, self.a)
        fun_ab = function(self.a, self.b)
        fun_bb = function(self.b, self.b)

        self.assertIsInstance(fun_aa, type(self.a))
        self.assertIsInstance(fun_ab, type(self.a))
        self.assertIsInstance(fun_bb, type(self.a))

        self.assertAlmostEqual(float(fun_aa), results[0])
        self.assertAlmostEqual(float(fun_ab), results[1])
        self.assertAlmostEqual(float(fun_bb), results[2])

    def test_distance_covariance_sqr(self) -> None:
        """Basic check of energy_distance."""
        self._test_generic(
            dcor.distance_covariance_sqr,
            (52.0, 1.0, 0.25),
        )

    def test_u_distance_covariance_sqr(self) -> None:
        """Basic check of energy_distance."""
        self._test_generic(
            dcor.u_distance_covariance_sqr,
            (42.666666666, -2.666666666, 0.666666666),
        )

    def test_distance_covariance(self) -> None:
        """Basic check of energy_distance."""
        self._test_generic(
            dcor.distance_covariance,
            (7.211102550, 1.0, 0.5),
        )

    def test_distance_correlation_sqr(self) -> None:
        """Basic check of energy_distance."""
        self._test_generic(
            dcor.distance_correlation_sqr,
            (1.0, 0.277350098, 1.0),
        )

    def test_u_distance_correlation_sqr(self) -> None:
        """Basic check of energy_distance."""
        self._test_generic(
            dcor.u_distance_correlation_sqr,
            (1.0, -0.5, 1.0),
        )

    def test_distance_correlation(self) -> None:
        """Basic check of energy_distance."""
        self._test_generic(
            dcor.distance_correlation,
            (1.0, 0.526640387, 1.0),
        )

    def test_dcor_constant(self) -> None:
        """Test that it works with constant random variables."""
        a = array_api_strict.ones(100)

        cov = dcor.distance_covariance(a, a)
        self.assertIsInstance(cov, type(self.a))
        self.assertAlmostEqual(cov, 0)

        corr = dcor.distance_correlation(a, a)
        self.assertIsInstance(corr, type(self.a))
        self.assertAlmostEqual(corr, 0)

        corr_af_inv = dcor.distance_correlation_af_inv(a, a)
        self.assertIsInstance(corr_af_inv, type(self.a))
        self.assertAlmostEqual(corr_af_inv, 0)


if __name__ == "__main__":
    unittest.main()  # pragma: no cover

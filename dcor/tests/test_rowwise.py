import dcor
from dcor._utils import RowwiseMode
import unittest

import numpy as np


class TestRowwise(unittest.TestCase):

    def setUp(self):
        self.arr1 = np.array(((1.,), (2.,), (3.,), (4.,), (5.,), (6.,)))
        self.arr2 = np.array(((1.,), (7.,), (5.,), (5.,), (6.,), (2.,)))

        self.arr_list_1 = [self.arr1, self.arr1, self.arr2, self.arr2]
        self.arr_list_2 = [self.arr1, self.arr2, self.arr1, self.arr2, ]

        self.dcov_sqr_results = np.array([1.70679, 0.68519, 0.68519, 2.92593])
        self.u_dcov_sqr_results = np.array(
            [1.55556, -0.88889, -0.88889, 2.93333])
        self.dcov_results = np.sqrt(self.dcov_sqr_results)
        self.dcorr_sqr_results = np.array([1., 0.30661, 0.30661, 1.])
        self.u_dcorr_sqr_results = np.array([1., -0.41613, -0.41613, 1.])
        self.dcorr_results = np.sqrt(self.dcorr_sqr_results)

    def test_naive(self):

        dcov_sqr = dcor.rowwise(dcor.distance_covariance_sqr,
                                self.arr_list_1, self.arr_list_2,
                                rowwise_mode=dcor.RowwiseMode.NAIVE)

        np.testing.assert_allclose(dcov_sqr, self.dcov_sqr_results, rtol=1e-5)

    def test_rowwise(self):

        for rowwise_mode in dcor.RowwiseMode:
            with self.subTest(rowwise_mode=rowwise_mode):

                compile_modes = [
                    dcor.CompileMode.AUTO,
                    dcor.CompileMode.COMPILE_CPU]

                if rowwise_mode != RowwiseMode.NAIVE:
                    compile_modes += [dcor.CompileMode.COMPILE_PARALLEL]

                for compile_mode in compile_modes:

                    with self.subTest(compile_mode=compile_mode):

                        dcov_sqr = dcor.rowwise(
                            dcor.distance_covariance_sqr,
                            self.arr_list_1, self.arr_list_2,
                            rowwise_mode=rowwise_mode,
                            compile_mode=compile_mode)

                        np.testing.assert_allclose(
                            dcov_sqr, self.dcov_sqr_results, rtol=1e-5)

                        u_dcov_sqr = dcor.rowwise(
                            dcor.u_distance_covariance_sqr,
                            self.arr_list_1, self.arr_list_2,
                            rowwise_mode=rowwise_mode,
                            compile_mode=compile_mode)

                        np.testing.assert_allclose(
                            u_dcov_sqr, self.u_dcov_sqr_results, rtol=1e-5)

                        dcov = dcor.rowwise(
                            dcor.distance_covariance,
                            self.arr_list_1, self.arr_list_2,
                            rowwise_mode=rowwise_mode,
                            compile_mode=compile_mode)

                        np.testing.assert_allclose(
                            dcov, self.dcov_results, rtol=1e-5)

                        dcorr_sqr = dcor.rowwise(
                            dcor.distance_correlation_sqr,
                            self.arr_list_1, self.arr_list_2,
                            rowwise_mode=rowwise_mode,
                            compile_mode=compile_mode)

                        np.testing.assert_allclose(
                            dcorr_sqr, self.dcorr_sqr_results, rtol=1e-5)

                        u_dcorr_sqr = dcor.rowwise(
                            dcor.u_distance_correlation_sqr,
                            self.arr_list_1, self.arr_list_2,
                            rowwise_mode=rowwise_mode,
                            compile_mode=compile_mode)

                        np.testing.assert_allclose(
                            u_dcorr_sqr, self.u_dcorr_sqr_results, rtol=1e-4)

                        dcorr = dcor.rowwise(
                            dcor.distance_correlation,
                            self.arr_list_1, self.arr_list_2,
                            rowwise_mode=rowwise_mode,
                            compile_mode=compile_mode)

                        np.testing.assert_allclose(
                            dcorr, self.dcorr_results, rtol=1e-5)

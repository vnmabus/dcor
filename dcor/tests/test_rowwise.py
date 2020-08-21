import dcor
import unittest

import numpy as np


class TestRowwise(unittest.TestCase):

    def setUp(self):
        self.arr1 = np.array(((1.,), (2.,), (3.,), (4.,), (5.,), (6.,)))
        self.arr2 = np.array(((1.,), (7.,), (5.,), (5.,), (6.,), (2.,)))

        self.arr_list_1 = [self.arr1, self.arr1, self.arr2, self.arr2]
        self.arr_list_2 = [self.arr1, self.arr2, self.arr1, self.arr2, ]

        self.dcov_sqr_results = np.array([1.70679, 0.68519, 0.68519, 2.92593])

    def test_naive(self):

        dcov_sqr = dcor.rowwise(dcor.distance_covariance_sqr,
                                self.arr_list_1, self.arr_list_2,
                                rowwise_mode=dcor.RowwiseMode.NAIVE)

        np.testing.assert_allclose(dcov_sqr, self.dcov_sqr_results, rtol=1e-5)

    def test_optimized_avl(self):

        compile_modes = [  # dcor.CompileMode.AUTO,
            dcor.CompileMode.COMPILE_CPU,
            dcor.CompileMode.COMPILE_PARALLEL]

        for compile_mode in compile_modes:

            with self.subTest(compile_mode=compile_mode):

                dcov_sqr = dcor.rowwise(
                    dcor.distance_covariance_sqr,
                    self.arr_list_1, self.arr_list_2,
                    rowwise_mode=dcor.RowwiseMode.OPTIMIZED,
                    compile_mode=compile_mode)

                np.testing.assert_allclose(
                    dcov_sqr, self.dcov_sqr_results, rtol=1e-5)

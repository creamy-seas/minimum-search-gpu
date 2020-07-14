import unittest
from unittest.mock import Mock
from unittest.mock import patch

from kernels.potential_evaluator import PotentialEvaluator
from template import mock_potential_function_cuda
import numpy as np
from numba import cuda

class TestPotentialEvaluator(unittest.TestCase):
    def setUp(self):

        self.NUMBER_OF_PHI_POINTS = 3
        self.phi_array = np.array([1, 2, 3])
        self.NUMBER_OF_FIELD_POINTS = 1
        self.lr_array = np.array([0])
        self.alpha = 7

        self.DEVICE_out_array = cuda.device_array(
            shape=(
                self.NUMBER_OF_FIELD_POINTS,
                self.NUMBER_OF_FIELD_POINTS,
                self.NUMBER_OF_PHI_POINTS,
                self.NUMBER_OF_PHI_POINTS,
                self.NUMBER_OF_PHI_POINTS,
            ),
            dtype=np.int32,
        )

        self.sut = PotentialEvaluator(
            self.NUMBER_OF_FIELD_POINTS,
            self.NUMBER_OF_PHI_POINTS,
            mock_potential_function_cuda,
        )

    def tearDown(self):
        pass

    def test(self):
        THREADS_PER_BLOCK = self.sut.allocate_max_threads()

        self.sut.kernel[
            (self.NUMBER_OF_FIELD_POINTS, self.NUMBER_OF_FIELD_POINTS),
            THREADS_PER_BLOCK,
        ](
            cuda.to_device(self.phi_array),
            cuda.to_device(self.lr_array),
            self.alpha,
            self.DEVICE_out_array
        )
        expected = np.array([[[[[711100, 711200, 711300],
                [712100, 712200, 712300],
                [713100, 713200, 713300]],
               [[721100, 721200, 721300],
                [722100, 722200, 722300],
                [723100, 723200, 723300]],
               [[731100, 731200, 731300],
                [732100, 732200, 732300],
                [733100, 733200, 733300]]]]])

        self.assertAlmostEqual(
            expected[0][0][1][0][0],
            self.DEVICE_out_array.copy_to_host()[0][0][1][0][0]
        )

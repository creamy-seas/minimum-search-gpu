import unittest
from unittest.mock import Mock
from unittest.mock import patch

from kernels.potential_evaluator import PotentialEvaluator
from template import mock_potential_function_cuda
from utils.info import allocate_max_threads
import numpy as np
from numba import cuda


class TestPotentialEvaluator(unittest.TestCase):
    def setUp(self):

        self.NUMBER_OF_PHI_POINTS = 3
        self.phi_array = np.array([1, 2, 3])
        self.NUMBER_OF_FIELD_POINTS_PER_RUN = 1
        self.lr_array = np.array([0, 1, 2])

        self.alpha = 7

        self.DEVICE_out_array = cuda.device_array(
            shape=(
                self.NUMBER_OF_FIELD_POINTS_PER_RUN,
                self.NUMBER_OF_FIELD_POINTS_PER_RUN,
                self.NUMBER_OF_PHI_POINTS,
                self.NUMBER_OF_PHI_POINTS,
                self.NUMBER_OF_PHI_POINTS,
            ),
            dtype=np.int32,
        )

        self.sut = PotentialEvaluator(
            self.NUMBER_OF_PHI_POINTS, mock_potential_function_cuda,
        )

    def tearDown(self):
        pass

    def test(self):
        THREADS_PER_BLOCK = allocate_max_threads()

        self.sut.kernel[
            (self.NUMBER_OF_FIELD_POINTS_PER_RUN, self.NUMBER_OF_FIELD_POINTS_PER_RUN),
            THREADS_PER_BLOCK,
        ](
            cuda.to_device(self.phi_array),
            cuda.to_device(self.lr_array),
            0,
            0,
            self.alpha,
            self.DEVICE_out_array,
        )
        expected = np.array(
            [
                [
                    [
                        [
                            [711100, 711200, 711300],
                            [712100, 712200, 712300],
                            [713100, 713200, 713300],
                        ],
                        [
                            [721100, 721200, 721300],
                            [722100, 722200, 722300],
                            [723100, 723200, 723300],
                        ],
                        [
                            [731100, 731200, 731300],
                            [732100, 732200, 732300],
                            [733100, 733200, 733300],
                        ],
                    ]
                ]
            ]
        )
        np.all(expected == self.DEVICE_out_array.copy_to_host())

    def test_offset(self):
        THREADS_PER_BLOCK = allocate_max_threads()

        self.sut.kernel[
            (self.NUMBER_OF_FIELD_POINTS_PER_RUN, self.NUMBER_OF_FIELD_POINTS_PER_RUN),
            THREADS_PER_BLOCK,
        ](
            cuda.to_device(self.phi_array),
            cuda.to_device(self.lr_array),
            1,
            2,
            self.alpha,
            self.DEVICE_out_array,
        )
        expected = np.array(
            [
                [
                    [
                        [
                            [711112, 711212, 711312],
                            [712112, 712212, 712312],
                            [713112, 713212, 713312],
                        ],
                        [
                            [721112, 721212, 721312],
                            [722112, 722212, 722312],
                            [723112, 723212, 723312],
                        ],
                        [
                            [731112, 731212, 731312],
                            [732112, 732212, 732312],
                            [733112, 733212, 733312],
                        ],
                    ]
                ]
            ]
        )
        self.assertTrue(np.all(expected == self.DEVICE_out_array.copy_to_host()))

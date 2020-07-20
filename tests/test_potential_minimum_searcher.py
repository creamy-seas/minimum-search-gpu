import unittest
from unittest.mock import Mock
from unittest.mock import patch

from kernels.potential_minimum_searcher import PotentialMinimumSearcher
import numpy as np
from numba import cuda


class TestPotentialMinimumSearcher(unittest.TestCase):
    def setUp(self):

        self.NUMBER_OF_PHI_POINTS = 2
        self.potential_array = np.array(
            [
                [
                    [
                        [
                            [10, 11],  # along phi03 - 2 values
                            [12, 13],  # along phi03 - 2 values
                        ],  # along phi02 - 2 values
                        [[14, 15], [16, 17]],  # along phi02 - 2 values
                    ],  # along phi01 - 2 values
                    [
                        [[22, 23], [24, 25]],
                        [[18, 19], [20, 21]],
                    ],  # along phi01 - 2 values
                ],  # along R - 2 values
                [
                    [[[38, 39], [40, 41]], [[35, 34], [36, 37]]],
                    [[[33, 32], [31, 30]], [[29, 28], [27, 26]]],
                ],  # along R - 2 values
            ],  # along L - 2 values
            dtype=np.float32,
        )

        self.sut = PotentialMinimumSearcher(self.NUMBER_OF_PHI_POINTS)

    def tearDown(self):
        pass

    def test(self):
        NUMBER_OF_FIELD_POINTS = 2
        THREADS_PER_BLOCK = self.sut.allocate_max_threads()
        BLOCKS_PER_GRID = (
            NUMBER_OF_FIELD_POINTS,
            NUMBER_OF_FIELD_POINTS,
        )
        DEVICE_potential_array = cuda.to_device(self.potential_array)
        DEVICE_out_array = cuda.device_array(
            shape=(NUMBER_OF_FIELD_POINTS, NUMBER_OF_FIELD_POINTS, 4), dtype=np.float32,
        )

        self.sut.kernel[BLOCKS_PER_GRID, THREADS_PER_BLOCK](
            DEVICE_potential_array, DEVICE_out_array,
        )

        expected_array = np.array(
            [[[10, 0, 0, 0], [18, 1, 0, 0]], [[34, 1, 0, 1], [26, 1, 1, 1]],]
        )
        np.testing.assert_almost_equal(
            expected_array, DEVICE_out_array.copy_to_host(), decimal=0
        )

import unittest
from unittest.mock import Mock
from unittest.mock import patch

from kernels.potential_minimum_searcher import PotentialMinimumSearcher
import numpy as np
from numba import cuda


class TestPotentialMinimumSearcher(unittest.TestCase):
    def tearDown(self):
        pass

    def test(self):
        NUMBER_OF_PHI_POINTS = 2
        NUMBER_OF_FIELD_POINTS = 2

        sut = PotentialMinimumSearcher(NUMBER_OF_PHI_POINTS)
        THREADS_PER_BLOCK = sut.allocate_max_threads()
        BLOCKS_PER_GRID = (
            NUMBER_OF_FIELD_POINTS,
            NUMBER_OF_FIELD_POINTS,
        )
        DEVICE_potential_array = cuda.to_device(
            np.array(
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
        )
        DEVICE_out_array = cuda.device_array(
            shape=(NUMBER_OF_FIELD_POINTS, NUMBER_OF_FIELD_POINTS, 4), dtype=np.float32,
        )

        sut.kernel[BLOCKS_PER_GRID, THREADS_PER_BLOCK](
            DEVICE_potential_array, DEVICE_out_array,
        )

        expected_array = np.array(
            [[[10, 0, 0, 0], [18, 1, 0, 0]], [[34, 1, 0, 1], [26, 1, 1, 1]],]
        )

        self.assertTrue(np.all(DEVICE_out_array.copy_to_host() == expected_array))

    def test_alternative(self):
        NUMBER_OF_PHI_POINTS = 5
        NUMBER_OF_FIELD_POINTS = 1

        sut = PotentialMinimumSearcher(NUMBER_OF_PHI_POINTS)
        THREADS_PER_BLOCK = (2, 2)
        BLOCKS_PER_GRID = (
            NUMBER_OF_FIELD_POINTS,
            NUMBER_OF_FIELD_POINTS,
        )
        DEVICE_potential_array = cuda.to_device(
            np.array(
                [
                    [
                        [
                            [
                                [5.0, 5.0, 5.0, 5.0, 5.0],
                                [6.5, 4.5, 4.5, 6.5, 6.5],
                                [8.0, 6.0, 4.0, 6.0, 8.0],
                                [6.5, 6.5, 4.5, 4.5, 6.5],
                                [5.0, 5.0, 5.0, 5.0, 5.0],
                            ],
                            [
                                [5.0, 5.0, 5.0, 5.0, 5.0],
                                [4.5, 2.5, 2.5, 4.5, 4.5],
                                [6.0, 4.0, 2.0, 4.0, 6.0],
                                [6.5, 6.5, 4.5, 4.5, 6.5],
                                [5.0, 5.0, 5.0, 5.0, 5.0],
                            ],
                            [
                                [5.0, 5.0, 5.0, 5.0, 5.0],
                                [4.5, 2.5, 2.5, 4.5, 4.5],
                                [4.0, 2.0, 0.0, 2.0, 4.0],
                                [4.5, 4.5, 2.5, 2.5, 4.5],
                                [5.0, 5.0, 5.0, 5.0, 5.0],
                            ],
                            [
                                [5.0, 5.0, 5.0, 5.0, 5.0],
                                [6.5, 4.5, 4.5, 6.5, 6.5],
                                [6.0, 4.0, 2.0, 4.0, 6.0],
                                [4.5, 4.5, 2.5, 2.5, 4.5],
                                [5.0, 5.0, 5.0, 5.0, 5.0],
                            ],
                            [
                                [5.0, 5.0, 5.0, 5.0, 5.0],
                                [6.5, 4.5, 4.5, 6.5, 6.5],
                                [8.0, 6.0, 4.0, 6.0, 8.0],
                                [6.5, 6.5, 4.5, 4.5, 6.5],
                                [5.0, 5.0, 5.0, 5.0, 5.0],
                            ],
                        ]
                    ]
                ]
            )
        )
        DEVICE_out_array = cuda.device_array(
            shape=(NUMBER_OF_FIELD_POINTS, NUMBER_OF_FIELD_POINTS, 4), dtype=np.float32,
        )

        sut.kernel[BLOCKS_PER_GRID, THREADS_PER_BLOCK](
            DEVICE_potential_array, DEVICE_out_array,
        )

        expected_array = np.array([[[0, 2, 2, 2]]])

        self.assertTrue(np.all(DEVICE_out_array.copy_to_host() == expected_array))

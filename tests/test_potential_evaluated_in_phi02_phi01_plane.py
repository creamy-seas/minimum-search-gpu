import unittest
from unittest.mock import Mock
from unittest.mock import patch

from kernels.potential_evaluated_in_phi02_phi01_plane import (
    PotentialEvaluatedInPhi02Phi02Plane,
)
from template import mock_potential_function_cuda
import numpy as np
from numba import cuda


class TestPotentialEvaluatedInPhi02Phi02Plane(unittest.TestCase):
    def setUp(self):

        self.NUMBER_OF_PHI_POINTS = 3

        self.sut = PotentialEvaluatedInPhi02Phi02Plane(
            self.NUMBER_OF_PHI_POINTS, mock_potential_function_cuda,
        )

    def tearDown(self):
        pass

    def test(self):
        phi01_array = np.array([4, 5, 6])
        phi02_array = np.array([4, 5, 6])
        phi03_array = np.array([4, 5, 6])
        alpha = 7
        DEVICE_out_array = cuda.device_array(
            shape=(self.NUMBER_OF_PHI_POINTS, self.NUMBER_OF_PHI_POINTS,),
            dtype=np.float32,
        )

        self.sut.kernel[(self.NUMBER_OF_PHI_POINTS, self.NUMBER_OF_PHI_POINTS), 10,](
            cuda.to_device(phi01_array),
            cuda.to_device(phi01_array),
            cuda.to_device(phi01_array),
            0,
            0,
            alpha,
            DEVICE_out_array,
        )
        expected = np.array(
            [
                [744400, 745400, 746400],
                [754400, 755400, 756400],
                [764400, 765400, 766400],
            ]
        )
        np.all(expected == DEVICE_out_array.copy_to_host())

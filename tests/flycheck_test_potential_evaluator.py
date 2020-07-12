import unittest
from unittest.mock import Mock
from unittest.mock import patch

from numba import cuda
from kernels.potential_evaluator import PotentialEvaluator


class TestPotentialEvaluator(unittest.TestCase):
    def setUp(self):
        self.NUMBER_OF_PHI_POINTS = 10
        self.NUMBER_OF_FIELD_POINTS = 10

        def dummy_potential_function(phi_array, L, R, alpha):
            return phi_array[0] + phi_array[1] + phi_array[2] + 10 * (L + R) + 100 * alpha
        self.dummy_potential_function = dummy_potential_function

        self.sut = PotentialEvaluator(
            self.NUMBER_OF_FIELD_POINTS,
            self.NUMBER_OF_PHI_POINTS,
        )

    def tearDown(self):
        pass


    def test(self):
        pass

import unittest
from unittest.mock import Mock
from unittest.mock import patch

from numba import cuda
import numpy as np

from functions.potential import potential_function, potential_function_cuda
from template import mock_potential_function_cuda


class TestPotentialFunction(unittest.TestCase):
    def setUp(self):
        @cuda.jit
        def kernel_wrapped_func(
            phi01: float,
            phi02: float,
            phi03: float,
            L: float,
            R: float,
            alpha: float,
            store_value,
        ):
            store_value[0] = potential_function_cuda((phi01, phi02, phi03), L, R, alpha)

        self.sut = kernel_wrapped_func

    def tearDown(self):
        pass

    def test_potential_function_cuda__mocked(self):
        """Tests the patched function that we use in other tests"""
        @cuda.jit
        def kernel_wrapped_func(
            phi01: float,
            phi02: float,
            phi03: float,
            L: float,
            R: float,
            alpha: float,
            store_value,
        ):
            store_value[0] = mock_potential_function_cuda((phi01, phi02, phi03), L, R, alpha)

        store_result = cuda.device_array(shape=(1), dtype=np.float32)
        test_result = kernel_wrapped_func[1, 1](1, 2, 3, 4, 5, 6, store_result)
        self.assertEqual(store_result.copy_to_host()[0], 612345)

    def test_potential_function_cuda(self):
        store_result = cuda.device_array(shape=(1), dtype=np.float32)

        test_result = self.sut[1, 1](0, 0, 0, 0, 0, 1, store_result)
        self.assertAlmostEqual(store_result.copy_to_host()[0], 0)

        test_result = self.sut[1, 1](0, np.pi / 2, 0, 0, 0, 1, store_result)
        self.assertAlmostEqual(store_result.copy_to_host()[0], 3, f"🐳 Expected a 3")

    def test_potential_function(self):
        test_result = potential_function((0, np.pi / 2, 0), 0, 0, 1)
        self.assertAlmostEqual(test_result, 3)

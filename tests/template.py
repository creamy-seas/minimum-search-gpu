from typing import List, Tuple
from numba import cuda


def mock_potential_function(phi_array: Tuple, L: float, R: float, alpha: float):
    return (
        alpha * 10 ** 5
        + phi_array[0] * 10 ** 4
        + phi_array[1] * 10 ** 3
        + phi_array[2] * 10 ** 2
        + L * 10 ** 1
        + R * 10 ** 0
    )

mock_potential_function_cuda = cuda.jit(mock_potential_function, device=True)

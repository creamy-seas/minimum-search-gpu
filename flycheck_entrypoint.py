import math

pi = math.pi

import numpy as np
from numba import cuda
from numba.cuda.cudadrv.devicearray import DeviceNDArray

from functions.potential import potential_function_cuda
from kernels.potential_evaluator import PotentialEvaluator

# Parameters for simulation ###################################################
NUMBER_OF_PHI_POINTS = 100
NUMBER_OF_FIELD_POINTS = 50
ALPHA = 1
LOWER = -0.5
UPPER = 1.5

lr_array = np.linspace(LOWER * 2 * pi, UPPER * 2 * pi, NUMBER_OF_FIELD_POINTS)
phixx_array = np.linspace(-pi, pi, NUMBER_OF_PHI_POINTS)

# Kernels #####################################################################
potential_evaluator = PotentialEvaluator(
    NUMBER_OF_FIELD_POINTS, NUMBER_OF_PHI_POINTS, potential_function_cuda
)
THREADS_PER_BLOCK = potential_evaluator.allocate_max_threads()
BLOCKS_PER_GRID = (NUMBER_OF_FIELD_POINTS, NUMBER_OF_FIELD_POINTS)
potential_evaluator.verify_blocks_per_grid(BLOCKS_PER_GRID)
# Execution ###################################################################
DEVICE_lr_array = cuda.to_device(lr_array)
DEVICE_phixx_array = cuda.to_device(phixx_array)
DEVICE_potential_array = cuda.device_array(
    shape=(
        NUMBER_OF_FIELD_POINTS,
        NUMBER_OF_FIELD_POINTS,
        NUMBER_OF_PHI_POINTS,
        NUMBER_OF_PHI_POINTS,
        NUMBER_OF_PHI_POINTS,
    ),
    dtype=np.float32,
)

# potential_evaluator.kernel[BLOCKS_PER_GRID, THREADS_PER_BLOCK](
#     DEVICE_phixx_array, DEVICE_lr_array, ALPHA, DEVICE_potential_array
# )
# print(DEVICE_potential_array.copy_to_host().shape)

@cuda.jit(device=True)
def dummy_potential_function(phi_array, L: float, R: float, alpha: float):
    return (
        alpha * 10 ** 5
        + phi_array[0] * 10 ** 4
        + phi_array[1] * 10 ** 3
        + phi_array[2] * 10 ** 2
        + L * 10 ** 1
        + R * 10 ** 0
    )

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
    store_value[0] = dummy_potential_function(
        (phi01, phi02, phi03), L, R, alpha
    )

import numpy as np
store_result = cuda.device_array(shape=(1), dtype=np.float32)
kernel_wrapped_func[1,1](1, 2, 3, 4, 5, 6, store_result)


assert store_result.copy_to_host()[0] == 612345

#     kernel[NUMBER_OF_FIELD_POINTS, THREADS_PER_BLOCK](
#         , DEVICE_potential_array
#     )
#     # potential_array =
#     # DEVICE_potential_array_to_minimize = DEVICE_potential_array.copy_to_host()
#     print("Original")
#
#     print(DEVICE_potential_array.copy_to_host()[0][0][1])

import math

pi = math.pi

import numpy as np
from numba import cuda
from numba.cuda.cudadrv.devicearray import DeviceNDArray

from functions.potential import potential_function_cuda
from kernels.potential_evaluator import PotentialEvaluator
from utils.info import gpu_check

gpu_info = gpu_check()
max_shared_memory_per_block = gpu_info["max_shared_memory_per_block"]

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

potential_evaluator.kernel[NUMBER_OF_FIELD_POINTS, THREADS_PER_BLOCK](
    DEVICE_phixx_array, DEVICE_lr_array, ALPHA, DEVICE_potential_array
)
print(DEVICE_potential_array.copy_to_host()[0][0][0])

#     kernel[NUMBER_OF_FIELD_POINTS, THREADS_PER_BLOCK](
#         , DEVICE_potential_array
#     )
#     # potential_array =
#     # DEVICE_potential_array_to_minimize = DEVICE_potential_array.copy_to_host()
#     print("Original")
#
#     print(DEVICE_potential_array.copy_to_host()[0][0][1])

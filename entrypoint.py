import math
import itertools

pi = math.pi

import numpy as np
from numba import cuda
from numba.cuda.cudadrv.devicearray import DeviceNDArray

from functions.potential import potential_function_cuda
from kernels.potential_evaluator import PotentialEvaluator
from utils.array_stacker import ArrayStacker

# Parameters for simulation ###################################################
NUMBER_OF_PHI_POINTS = 100
NUMBER_OF_FIELD_POINTS = 40
NUMBER_OF_FIELD_POINTS_PER_RUN = 20
NUMBER_OF_FIELD_RUNS = (
    NUMBER_OF_FIELD_POINTS - 1
) // NUMBER_OF_FIELD_POINTS_PER_RUN + 1
ALPHA = 1
LOWER = -0.5
UPPER = 1.5

lr_array = np.linspace(LOWER * 2 * pi, UPPER * 2 * pi, NUMBER_OF_FIELD_POINTS)
phixx_array = np.linspace(-pi, pi, NUMBER_OF_PHI_POINTS)

# Kernels #####################################################################
potential_evaluator = PotentialEvaluator(NUMBER_OF_PHI_POINTS, potential_function_cuda)
THREADS_PER_BLOCK = potential_evaluator.allocate_max_threads(8)
BLOCKS_PER_GRID = (NUMBER_OF_FIELD_POINTS_PER_RUN, NUMBER_OF_FIELD_POINTS_PER_RUN)
potential_evaluator.verify_blocks_per_grid(BLOCKS_PER_GRID)

# Execution ###################################################################
DEVICE_lr_array = cuda.to_device(lr_array)
DEVICE_phixx_array = cuda.to_device(phixx_array)
DEVICE_potential_array = cuda.device_array(
    shape=(
        NUMBER_OF_FIELD_POINTS_PER_RUN,
        NUMBER_OF_FIELD_POINTS_PER_RUN,
        NUMBER_OF_PHI_POINTS,
        NUMBER_OF_PHI_POINTS,
        NUMBER_OF_PHI_POINTS,
    ),
    dtype=np.float32,
)

# Go through teach of the field section and evaluate ##########################
FIELD_SECTIONS = [[None] * NUMBER_OF_FIELD_RUNS for i in range(0, NUMBER_OF_FIELD_RUNS)]
for (L_RUN, R_RUN) in itertools.product(
    range(0, NUMBER_OF_FIELD_RUNS), range(0, NUMBER_OF_FIELD_RUNS)
):
    print(
        f"🦑 Running (L={L_RUN}/{NUMBER_OF_FIELD_RUNS - 1}), (R={R_RUN}/{NUMBER_OF_FIELD_RUNS - 1})"
    )
    L_OFFSET = int(L_RUN * NUMBER_OF_FIELD_POINTS_PER_RUN)
    R_OFFSET = int(R_RUN * NUMBER_OF_FIELD_POINTS_PER_RUN)
    potential_evaluator.kernel[BLOCKS_PER_GRID, THREADS_PER_BLOCK](
        DEVICE_phixx_array,
        DEVICE_lr_array,
        L_OFFSET,
        R_OFFSET,
        ALPHA,
        DEVICE_potential_array,
    )

    FIELD_SECTIONS[L_RUN][R_RUN] = DEVICE_potential_array.copy_to_host()

TOTAL_FIELD = ArrayStacker.stack_into_square(FIELD_SECTIONS)

print(TOTAL_FIELD)

from typing import List, Tuple
import math

import numpy as np
from numba import cuda
import numba as nb
from numba.cuda.cudadrv.devicearray import DeviceNDArray

from cuda_support_functions.cuda_device_parameters import gpu_check

FLUX = float
FLUX_NUMBER = int
cos = math.cos
sin = math.sin
pi = math.pi


def potential_function(
    phi_array: Tuple[FLUX], L: FLUX, R: FLUX, alpha: float,
):
    """Order of the flux array is [phi01, phi02, phi03]"""

    alpha = float(alpha)
    (L, R) = (float(L), float(R))
    (phi01, phi02, phi03) = (
        float(phi_array[0]),
        float(phi_array[1]),
        float(phi_array[2]),
    )

    return (
        4
        + alpha
        - (
            alpha * cos(phi02)
            + cos(phi01)
            + cos(phi03)
            + cos(phi02 - phi01 - L)
            + cos(phi02 - phi03 + R)
        )
    )


potential_function_cuda = cuda.jit(potential_function, device=True)

dimension_dict = gpu_check()
print(f"🦑 {dimension_dict['max_shared_memory_per_block'] / nb.float32.bitwidth}")
NUMBER_OF_PHI_POINTS = 100


@cuda.jit(device=True)
def potential_minimum_search(
    potential_grid: cuda.shared.array, info_array: DeviceNDArray
):
    ###########################################################################
    #                      Find minimal point in 3D grid                      #
    #       threads must be syncrhonized to ensure grid does not change       #
    ###########################################################################
    # Project from cube to plane (go down each vertical column) ###############
    phi01_idx = cuda.threadIdx.x
    phi02_idx = cuda.threadIdx.y
    min_phi03_grid = cuda.shared.array(
        shape=(NUMBER_OF_PHI_POINTS, NUMBER_OF_PHI_POINTS), dtype=nb.int16
    )

    while phi01_idx < NUMBER_OF_PHI_POINTS:
        while phi02_idx < NUMBER_OF_PHI_POINTS:
            (min_potential, min_phi03_idx) = (None, None)
            for phi03_idx in range(0, NUMBER_OF_PHI_POINTS):
                potential = potential_grid[phi01_idx][phi02_idx][phi03_idx]
                if min_potential is None or potential < min_potential:
                    min_potential = potential
                    min_phi03_idx = phi03_idx

            potential_grid[phi01_idx][phi02_idx][0] = min_potential
            min_phi03_grid[phi01_idx][phi02_idx] = min_phi03_idx
            # info_array[0][0] = min_phi03_idx
            phi02_idx += cuda.blockDim.y
        phi02_idx = cuda.threadIdx.y
        phi01_idx += cuda.blockDim.x

    cuda.syncthreads()

    # for i in range(0, len(min_phi03_grid)):
    #     for j in range(0, len(min_phi03_grid[0])):
    #         info_array[i][j] = min_phi03_grid[i][j]


@cuda.jit(device=True)
def dump_to_cell(
    x: int, y: int, value, array_to_dump_to: DeviceNDArray,
):
    for xidx in range(x - 1, x + 2):
        if xidx >= 0 and xidx < len(array_to_dump_to):
            for yidx in range(y - 1, y + 2):
                if yidx >= 0 and yidx < len(array_to_dump_to):
                    array_to_dump_to[xidx][yidx] = 88
    array_to_dump_to[x][y] = value


@cuda.jit(device=True)
def dump_thread_information(array_to_dump_to: DeviceNDArray):
    dump_to_cell(1, 3, cuda.threadIdx.x, array_to_dump_to)
    dump_to_cell(3, 3, cuda.threadIdx.y, array_to_dump_to)
    dump_to_cell(5, 3, cuda.threadIdx.z, array_to_dump_to)
    dump_to_cell(1, 1, cuda.blockDim.x, array_to_dump_to)
    dump_to_cell(3, 1, cuda.blockDim.y, array_to_dump_to)
    dump_to_cell(5, 1, cuda.blockDim.z, array_to_dump_to)


@cuda.jit
def kernel(
    phixx_array: List[float],
    lr_array: List[float],
    alpha: float,
    array_out: DeviceNDArray,
    info_array: DeviceNDArray,
):
    # shared memory in a single block - must be known at compile time
    potential_grid = cuda.shared.array(
        shape=(NUMBER_OF_PHI_POINTS, NUMBER_OF_PHI_POINTS, NUMBER_OF_PHI_POINTS),
        dtype=nb.int16,
    )

    phi01_idx = cuda.threadIdx.x
    phi02_idx = cuda.threadIdx.y
    phi03_idx = cuda.threadIdx.z

    # Traverse over the full grid
    while phi01_idx < NUMBER_OF_PHI_POINTS:
        while phi02_idx < NUMBER_OF_PHI_POINTS:
            while phi03_idx < NUMBER_OF_PHI_POINTS:
                potential_grid[phi01_idx][phi02_idx][
                    phi03_idx
                ] = potential_function_cuda(
                    (
                        phixx_array[phi01_idx],
                        phixx_array[phi02_idx],
                        phixx_array[phi03_idx],
                    ),
                    0,
                    0,
                    alpha,
                )

                phi03_idx += cuda.blockDim.z
            phi03_idx = cuda.threadIdx.z
            phi02_idx += cuda.blockDim.y
        phi02_idx = cuda.threadIdx.y
        phi01_idx += cuda.blockDim.x

    # cuda.syncthreads()

    # dump_thread_information(info_array)
    # dump_to_cell(1, 5, phi01_idx, info_array)
    # dump_to_cell(3, 5, phi02_idx, info_array)
    # dump_to_cell(5, 5, phi03_idx, info_array)

    # potential_minimum_search(potential_grid, info_array)

    # # Project from plane to line (go across each row in the plane) ############
    # phi01_idx = cuda.threadIdx.x
    # min_phi02_grid = cuda.shared.array(shape=NUMBER_OF_PHI_POINTS, dtype=nb.int16)

    # while phi01_idx < NUMBER_OF_PHI_POINTS:
    #     (min_potential, min_phi02_idx) = (None, None)
    #     for phi02_idx in range(0, NUMBER_OF_PHI_POINTS):
    #         potential = potential_grid[phi01_idx][phi02_idx][0]
    #         if min_potential is None or potential < min_potential:
    #             min_potential = potential
    #             min_phi02_idx = phi02_idx

    #     potential_grid[phi01_idx][0][0] = min_potential
    #     # min_phi03_grid[phi01_idx][0] = min_phi03_grid[phi01_idx][min_phi02_idx]
    #     array_out[phi01_idx][phi02_idx] = phi01_idx
    #     min_phi02_grid[phi01_idx] = min_phi02_idx
    #     phi01_idx += cuda.blockDim.x
    # cuda.syncthreads()

    # # Project from line to single point #######################################
    # (min_potential, min_phi01_idx) = (None, None)

    # for phi01_idx in range(0, NUMBER_OF_PHI_POINTS):
    #     potential = potential_grid[phi01_idx][0][0]
    #     if min_potential is None or potential < min_potential:
    #         min_potential = potential
    #         min_phi01_idx = phi01_idx

    # min_phi03_idx = min_phi03_grid[min_phi01_idx][0]
    # min_phi02_idx = min_phi02_grid[min_phi01_idx]


NUMBER_OF_PHI_POINTS = 100
NUMBER_OF_FIELD_POINTS = 2
ALPHA = 1
LOWER = -0.5
UPPER = 1.5

BLOCKS_PER_GRID = 1
THREADS_PER_BLOCK = (2, 2, 2)
# NUMBER_OF_PHI_POINTS, NUMBER_OF_PHI_POINTS, NUMBER_OF_PHI_POINTS

lr_array = np.linspace(LOWER * 2 * pi, UPPER * 2 * pi, NUMBER_OF_FIELD_POINTS)
phixx_array = np.linspace(-pi, pi, NUMBER_OF_PHI_POINTS)

DEVICE_lr_array = cuda.to_device(lr_array)
DEVICE_phixx_array = cuda.to_device(phixx_array)
DEVICE_out = cuda.device_array(
    shape=(NUMBER_OF_PHI_POINTS, NUMBER_OF_PHI_POINTS), dtype=np.float32
)
DEVICE_info = cuda.device_array(
    shape=(NUMBER_OF_PHI_POINTS, NUMBER_OF_PHI_POINTS), dtype=np.float32
)

kernel[BLOCKS_PER_GRID, THREADS_PER_BLOCK](
    DEVICE_phixx_array, DEVICE_lr_array, ALPHA, DEVICE_out, DEVICE_info
)
cuda.synchronize()

# print(DEVICE_out.copy_to_host())
print(DEVICE_info.copy_to_host())

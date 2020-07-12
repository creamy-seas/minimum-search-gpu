import numpy as np
from numba import cuda
import numba as nb
from cuda_support_functions.cuda_device_parameters import gpu_check
from cuda_support_functions.cuda_print_to_array import (
    dump_to_cell,
    dump_thread_information,
)
from functions.potential import potential_function_cuda



# Preparation steps ###########################################################


# Block allocation ############################################################
# BLOCKS_PER_GRID = (
#     NUMBER_OF_FIELD_POINTS
#     # min(gpu_info["max_grid_dim_x"], NUMBER_OF_FIELD_POINTS),
#     # min(gpu_info["max_grid_dim_y"], NUMBER_OF_FIELD_POINTS),
# )
# max_blocks_for_3d_thread_grid = int(1024 ** (1 / 3))
# THREADS_PER_BLOCK = (
#     min(max_blocks_for_3d_thread_grid, NUMBER_OF_PHI_POINTS),
#     min(max_blocks_for_3d_thread_grid, NUMBER_OF_PHI_POINTS),
#     min(max_blocks_for_3d_thread_grid, NUMBER_OF_PHI_POINTS),
# )
# TODO: hardcoded, otherwise an error is raised
# THREADS_PER_BLOCK=(8,10,10)
# print(
#     f"""🦑 Allocated
# {'Blocks:':<10}{BLOCKS_PER_GRID}
# {'Threads per block:':<10}{THREADS_PER_BLOCK}"""
# )




"""
# @cuda.jit(device=True)
@cuda.jit
def potential_minimum_search(
        potential_grid: "4D-array-of-potentials: [R, phi01, phi02, phi03]",
        min_phi03_grid: "2D array of min phi03 values",
        min_phi02_grid: "1D array of min phi02 values",
):
    ###########################################################################
    #                      Find minimal point in 3D grid                      #
    #       threads must be syncrhonized to ensure grid does not change       #
    ###########################################################################
    # Project from cube to plane (go down each vertical column) ###############
    R = cuda.blockIdx.x
    phi01_idx = cuda.threadIdx.x
    phi02_idx = cuda.threadIdx.y

    while phi01_idx < NUMBER_OF_PHI_POINTS:
        while phi02_idx < NUMBER_OF_PHI_POINTS:
            for (phi03_idx, potential) in enumerate(
                    potential_grid[R][phi01_idx][phi02_idx]
            ):
                if potential < potential_grid[R][phi01_idx][phi02_idx][0]:
                    potential_grid[R][phi01_idx][phi02_idx][0] = potential
                    min_phi03_grid[phi01_idx][phi02_idx] = phi03_idx

            phi02_idx = phi02_idx + cuda.blockDim.y
        phi02_idx = cuda.threadIdx.y
        phi01_idx += cuda.blockDim.x
    cuda.syncthreads()

    # Project from plane to line (go across each row in the plane) ############
    phi01_idx = cuda.blockIdx.x

    while phi01_idx < NUMBER_OF_PHI_POINTS:
        for (phi02_idx, potential) in enumerate(
                potential_grid[R][phi01_idx][:][0]
            ):
            if potential < potential_grid[R][phi01_idx][0][0]:
                potential_grid[R][phi01_idx][0][0] = potential
                min_phi03_grid[phi01_idx][0] = min_phi03_grid[phi01_idx][phi02_idx]
                min_phi02_grid[phi01_idx] = phi02_idx

        phi01_idx += cuda.blockDim.x



###############################################################################
#                                  Execution                                  #
###############################################################################
DEVICE_min_phi03_grid = cuda.device_array(
        shape=(
            NUMBER_OF_PHI_POINTS,
            NUMBER_OF_PHI_POINTS,
        ),
        dtype=np.int16,
    )
DEVICE_min_phi02_grid = cuda.device_array(
        shape=(
            NUMBER_OF_PHI_POINTS,
        ),
        dtype=np.int16,
    )
DEVICE_phi01 = None



    potential_minimum_search[NUMBER_OF_FIELD_POINTS,
                             (30,
                              30)](
                                  DEVICE_potential_array, DEVICE_min_phi03_grid, DEVICE_min_phi02_grid
    )

    # print(f"NUMBER_of_PHI: {DEVICE_min_phi03_grid.copy_to_host()[0][0]}")
    # print(f"phi01: {DEVICE_min_phi03_grid.copy_to_host()[0][1]}")
    # print(f"phi02: {DEVICE_min_phi03_grid.copy_to_host()[0][2]}")
    print("Condensed")
    print(DEVICE_potential_array.copy_to_host()[0][0][0])
    print(DEVICE_potential_array.copy_to_host()[0][0][1])
    print(DEVICE_potential_array.copy_to_host()[0][0][2])
    print(DEVICE_potential_array.copy_to_host()[0][0][3])
    print(DEVICE_potential_array.copy_to_host()[0][0][4])
    print(DEVICE_min_phi03_grid.copy_to_host())
    # print(DEVICE_min_phi02_grid.copy_to_host())
    # THREADS_PER_BLOCK = (100)
    # DEVICE_potential_array = cuda.to_device(potential_array)
    # potential_minimum_search_from2d_to_1d[BLOCKS_PER_GRID, THREADS_PER_BLOCK](
        # DEVICE_potential_array, DEVICE_min_phi03_grid, DEVICE_min_phi02_grid
    # )

    # (min_potential, min_phi01_idx) = (None, None)

    # for phi01_idx in range(0, NUMBER_OF_PHI_POINTS):
    #     potential = potential_grid[phi01_idx][0][0]
    #     if min_potential is None or potential < min_potential:
    #         min_potential = potential
    #         min_phi01_idx = phi01_idx

    # min_phi03_idx = min_phi03_grid[min_phi01_idx][0]
    # min_phi02_idx = min_phi02_grid[min_phi01_idx]
    # min_phi01 = min_phi01_idx

    # print(DEVICE_potential_array.copy_to_host())
"""

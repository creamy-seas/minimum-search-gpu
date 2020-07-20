"""Class that takes an array of potentials and finds the minimum point
"""
from typing import List, Callable, Tuple, Optional

from numba import cuda
from numba.cuda.cudadrv.devicearray import DeviceNDArray
import numpy as np

from utils.info import gpu_check


class PotentialMinimumSearcher:
    def __init__(self, number_of_phi_points: int):
        self.NUMBER_OF_PHI_POINTS = number_of_phi_points
        self.kernel = self.kernel_wrapper()
        self.gpu_info = gpu_check()

    def allocate_max_threads(
            self, user_defined_number: Optional[int] = None,
            verbose=False
    ) -> Tuple[int, int]:
        if verbose:
            print(
                f"""Thread parameters:
        > Max threads per block: {self.gpu_info['max_threads_per_block']}
        > Max threads in x: {self.gpu_info['max_block_dim_x']}
        > Max threads in y: {self.gpu_info['max_block_dim_y']}
        > Max threads in z: {self.gpu_info['max_block_dim_z']}"""
            )
        max_threads_approximation = int(
            self.gpu_info["max_threads_per_block"] ** (1 / 2)
        )
        if user_defined_number is not None:
            max_threads_approximation = user_defined_number

        max_thread_allocation = (
            min(max_threads_approximation, self.gpu_info["max_block_dim_x"]),
            min(max_threads_approximation, self.gpu_info["max_block_dim_x"]),
        )
        print(f"🐳 {'Allocating':<20} THREADS_PER_BLOCK = {max_thread_allocation}")

        return max_thread_allocation

    def kernel_wrapper(self):
        NUMBER_OF_PHI_POINTS = self.NUMBER_OF_PHI_POINTS

        @cuda.jit
        def kernel(
            potential_grid: DeviceNDArray, array_out: DeviceNDArray,
        ):
            """Take a 5D grid loaded into memory and find the minimum for each L-R point

            potential_grid:     array with the evaluated potential values at each L,R point
            array_out:          array with a [min_potential, min_phi01, min_phi02, min_phi03] for
                                each L,R point
            """

            L = cuda.blockIdx.x
            R = cuda.blockIdx.y

            # Project from cube to plane (go down each vertical column) ###############
            min_phi03_grid = cuda.shared.array(
                shape=(NUMBER_OF_PHI_POINTS, NUMBER_OF_PHI_POINTS), dtype=np.int16
            )
            phi01_idx = cuda.threadIdx.x
            phi02_idx = cuda.threadIdx.y

            while phi01_idx < NUMBER_OF_PHI_POINTS:
                while phi02_idx < NUMBER_OF_PHI_POINTS:
                    for (phi03_idx, potential) in enumerate(
                        potential_grid[L][R][phi01_idx][phi02_idx]
                    ):
                        if potential < potential_grid[L][R][phi01_idx][phi02_idx][0]:
                            potential_grid[L][R][phi01_idx][phi02_idx][0] = potential
                            min_phi03_grid[phi01_idx][phi02_idx] = phi03_idx
                    phi02_idx += cuda.blockDim.y
                phi02_idx = cuda.threadIdx.y
                phi01_idx += cuda.blockDim.x
            cuda.syncthreads()

            # Project from plane to line (go across each row in the plane) ############
            min_phi02_grid = cuda.shared.array(
                shape=(NUMBER_OF_PHI_POINTS), dtype=np.int16
            )
            phi01_idx = cuda.threadIdx.x
            cum = 0

            while phi01_idx < NUMBER_OF_PHI_POINTS:
                for (phi02_idx, potential) in enumerate(
                    potential_grid[L][R][phi01_idx][:, 0]
                ):
                    if potential < potential_grid[L][R][phi01_idx][0][0]:
                        potential_grid[L][R][phi01_idx][0][0] = potential
                        min_phi03_grid[phi01_idx][0] = min_phi03_grid[phi01_idx][
                            phi02_idx
                        ]
                        min_phi02_grid[phi01_idx] = phi02_idx
                phi01_idx += cuda.blockDim.x
            cuda.syncthreads()

            # Project from line to points (go across the line) ################
            for (phi01_idx, potential) in enumerate(potential_grid[L][R][:, 0, 0]):
                array_out[L][R][0] = potential_grid[L][R][0][0][0]
                if potential < potential_grid[L][R][0][0][0]:
                    array_out[L][R][0] = potential
                    array_out[L][R][1] = phi01_idx
                    array_out[L][R][2] = min_phi02_grid[phi01_idx]
                    array_out[L][R][3] = min_phi03_grid[phi01_idx][0]

        return kernel

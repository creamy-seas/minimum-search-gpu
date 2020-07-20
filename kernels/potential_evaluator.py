from typing import List, Callable, Tuple, Optional

from numba import cuda
from numba.cuda.cudadrv.devicearray import DeviceNDArray

from utils.info import gpu_check


class PotentialEvaluator:
    def __init__(
        self, number_of_phi_points: int, potential_function_cuda: Callable,
    ):
        self.NUMBER_OF_PHI_POINTS = number_of_phi_points
        self.potential_function_cuda = potential_function_cuda

        self.kernel = self.kernel_wrapper()
        self.gpu_info = gpu_check()

    def allocate_max_threads(
            self, user_defined_number: Optional[int] = None,
            verbose=False
    ) -> Tuple[int, int, int]:
        if verbose:
            print(
                f"""Thread parameters:
        > Max threads per block: {self.gpu_info['max_threads_per_block']}
        > Max threads in x: {self.gpu_info['max_block_dim_x']}
        > Max threads in y: {self.gpu_info['max_block_dim_y']}
        > Max threads in z: {self.gpu_info['max_block_dim_z']}"""
            )
        max_threads_approximation = int(
            self.gpu_info["max_threads_per_block"] ** (1 / 3)
        )
        if user_defined_number is not None:
            max_threads_approximation = user_defined_number

        max_thread_allocation = (
            min(max_threads_approximation, self.gpu_info["max_block_dim_x"]),
            min(max_threads_approximation, self.gpu_info["max_block_dim_y"]),
            min(max_threads_approximation, self.gpu_info["max_block_dim_z"]),
        )
        print(f"🐳 {'Allocating':<20} THREADS_PER_BLOCK = {max_thread_allocation}")

        return max_thread_allocation

    def verify_blocks_per_grid(self, blocks_per_grid: Tuple, verbose=False) -> bool:
        if verbose:
            print(
                f"""Block parameters:
        > Max blocks in x: {self.gpu_info['max_grid_dim_x']}
        > Max blocks in y: {self.gpu_info['max_grid_dim_y']}
        > Max blocks in z: {self.gpu_info['max_grid_dim_z']}"""
            )
        for (block_dim, max_dim) in zip(
            blocks_per_grid,
            [
                self.gpu_info["max_grid_dim_x"],
                self.gpu_info["max_grid_dim_y"],
                self.gpu_info["max_grid_dim_z"],
            ],
        ):
            if block_dim > max_dim:
                print("🦑 Allocating too many blocks")
                return False
        print(f"🐳 {'Verified':<20} BLOCKS_PER_GRID={blocks_per_grid}")
        return True

    def kernel_wrapper(self):
        NUMBER_OF_PHI_POINTS = self.NUMBER_OF_PHI_POINTS
        potential_function_cuda = self.potential_function_cuda

        @cuda.jit
        def kernel(
            phixx_array: List[float],
            lr_array: List[float],
            L_offset: int,
            R_offset: int,
            alpha: float,
            array_out: DeviceNDArray,
        ):
            """
            phixx_array:        array of the values that phi01, phi02, phi03
            lr_array:           array of the values for phil and phir
            L_offset, R_offset: because of finite memory on device, grid search is performed
                                on separate qudrants of the field.
                                In order to a global lr_array, this offset
                                if introduced to access elements for the different quadrants

            alpha:              variables parametr
            array_out:          allocate either with cuda.device_array or passing in a numpy array

            We perfrom evaluate of the potential
            """

            phi01_idx = cuda.threadIdx.x
            phi02_idx = cuda.threadIdx.y
            phi03_idx = cuda.threadIdx.z
            L = cuda.blockIdx.x
            R = cuda.blockIdx.y
            L_offset = int(L + L_offset)
            R_offset = int(R + R_offset)

            # Traverse over the full grid
            while phi01_idx < NUMBER_OF_PHI_POINTS:
                while phi02_idx < NUMBER_OF_PHI_POINTS:
                    while phi03_idx < NUMBER_OF_PHI_POINTS:
                        L_FIELD = lr_array[L_offset]
                        R_FIELD = lr_array[R_offset]
                        array_out[L][R][phi01_idx][phi02_idx][
                            phi03_idx
                        ] = potential_function_cuda(
                            (
                                phixx_array[phi01_idx],
                                phixx_array[phi02_idx],
                                phixx_array[phi03_idx],
                            ),
                            L_FIELD,
                            R_FIELD,
                            alpha,
                        )

                        phi03_idx += cuda.blockDim.z
                    phi03_idx = cuda.threadIdx.z
                    phi02_idx += cuda.blockDim.y
                phi02_idx = cuda.threadIdx.y
                phi01_idx += cuda.blockDim.x
            cuda.syncthreads()

        return kernel

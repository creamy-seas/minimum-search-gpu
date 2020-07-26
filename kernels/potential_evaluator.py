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
                        array_out[L][R][phi01_idx][phi02_idx][
                            phi03_idx
                        ] = potential_function_cuda(
                            (
                                phixx_array[phi01_idx],
                                phixx_array[phi02_idx],
                                phixx_array[phi03_idx],
                            ),
                            lr_array[L_offset],
                            lr_array[R_offset],
                            alpha,
                        )

                        phi03_idx += cuda.blockDim.z
                    phi03_idx = cuda.threadIdx.z
                    phi02_idx += cuda.blockDim.y
                phi02_idx = cuda.threadIdx.y
                phi01_idx += cuda.blockDim.x
            cuda.syncthreads()

        return kernel

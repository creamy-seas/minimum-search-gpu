from typing import List, Callable

from numba import cuda
from numba.cuda.cudadrv.devicearray import DeviceNDArray

from functions.potential import potential_function_cuda


class PotentialEvaluator:
    def __init__(
        self,
        number_of_field_points: int,
        number_of_phi_points: int,
        potential_function_cuda: Callable,
    ):
        self.NUMBER_OF_PHI_POINTS = number_of_phi_points
        self.NUMBER_OF_FIELD_POINTS = number_of_field_points
        self.kernel = self.kernel_wrapper()

    def kernel_wrapper(self):
        NUMBER_OF_FIELD_POINTS = self.NUMBER_OF_FIELD_POINTS
        NUMBER_OF_PHI_POINTS = self.NUMBER_OF_PHI_POINTS

        @cuda.jit
        def kernel(
            phixx_array: List[float],
            lr_array: List[float],
            R: int,
            alpha: float,
            array_out: DeviceNDArray,
        ):
            """
            phixx_array:        array of the values that phi01, phi02, phi03
            lr_array:           array of the values for phil and phir
            L:                  as we are memory bounded, fix L for given iteration
            alpha:              variables parametr
            array_out:          allocate either with cuda.device_array or passing in a numpy array

            We perfrom evaluate of the potential
            """

            phi01_idx = cuda.threadIdx.x
            phi02_idx = cuda.threadIdx.y
            phi03_idx = cuda.threadIdx.z
            L = cuda.blockIdx.x
            R = R

            # Traverse over the full grid
            while phi01_idx < NUMBER_OF_PHI_POINTS:
                while phi02_idx < NUMBER_OF_PHI_POINTS:
                    while phi03_idx < NUMBER_OF_PHI_POINTS:
                        array_out[R][phi01_idx][phi02_idx][
                            phi03_idx
                        ] = potential_function_cuda(
                            (
                                phixx_array[phi01_idx],
                                phixx_array[phi02_idx],
                                phixx_array[phi03_idx],
                            ),
                            lr_array[L],
                            lr_array[R],
                            alpha,
                        )

                        phi03_idx += cuda.blockDim.z
                    phi03_idx = cuda.threadIdx.z
                    phi02_idx += cuda.blockDim.y
                phi02_idx = cuda.threadIdx.y
                phi01_idx += cuda.blockDim.x
            cuda.syncthreads()

        return kernel

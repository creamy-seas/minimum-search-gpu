"""Potential is plotted for
- Fixed Phi L
- Fixed Phi R
- Minimised phi03

and for a range of phi02-phi01 values
"""

from typing import List, Callable, Tuple, Optional

from numba import cuda
from numba.cuda.cudadrv.devicearray import DeviceNDArray
import numpy as np


class PotentialEvaluatedInPhi02Phi02Plane:
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
            phi01_array: List[float],
            phi02_array: List[float],
            phi03_array: List[float],
            phi_l: float,
            phi_r: float,
            alpha: float,
            array_out: DeviceNDArray,
        ):
            """
            phixx_array:        array of the values for phi03 - float32
            phi_l, phi_r:       fixed values for the externally applied phase
            alpha:              variables parameter
            array_out:          allocate either with cuda.device_array or passing in a numpy array - floa32
            """

            phi01_idx = cuda.blockIdx.x
            phi02_idx = cuda.blockIdx.y
            phi03_idx = cuda.threadIdx.x
            potential_at_phi03 = cuda.shared.array(
                shape=(NUMBER_OF_PHI_POINTS), dtype=np.float32
            )

            # Traverse over all the phi03 values
            while phi03_idx < NUMBER_OF_PHI_POINTS:
                potential_at_phi03[phi03_idx] = potential_function_cuda(
                    (
                        phi01_array[phi01_idx],
                        phi02_array[phi02_idx],
                        phi03_array[phi03_idx],
                    ),
                    phi_l,
                    phi_r,
                    alpha,
                )
                phi03_idx += cuda.blockDim.x
            cuda.syncthreads()

            # Then step through and find minimal value
            min_potential = potential_at_phi03[0]
            for potential in potential_at_phi03[1:]:
                if potential < min_potential:
                    min_potential = potential

            array_out[phi01_idx][phi02_idx] = min_potential

        return kernel

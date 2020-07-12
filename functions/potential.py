import math
from typing import List, Tuple

from numba import cuda

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

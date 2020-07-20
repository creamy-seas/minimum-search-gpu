from typing import List

from numba import cuda
import numba as nb
import numpy as np
from scipy import optimize

FLUX = float
FLUX_NUMBER = int
cos = np.cos
sin = np.sin
pi = np.pi


def potential_function(phi_array: List[FLUX], L: FLUX, R: FLUX, alpha: float):
    """
    Potential to numerically minimize.
    Order of the flux array is [phi01, phi02, phi03]
    """
    (phi01, phi02, phi03) = (phi_array[0], phi_array[1], phi_array[2])

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


class Miminize:
    def __init__(self):
        self.minimize_vectorized = np.vectorize(
            self.minimize, excluded={"alpha", "number_of_phi_points"}
        )

    @staticmethod
    def minimize(L: FLUX, R: FLUX, alpha: float, number_of_phi_points: int = 100):
        phi_interval = (-pi, pi)

        (minimized_phi, min_potential, _, _) = optimize.brute(
            func=potential_function,
            ranges=(phi_interval, phi_interval, phi_interval),
            args=(L, R, alpha),
            full_output=True,
            Ns=number_of_phi_points,
            # finish=optimize.fmin,
            workers=16,
        )

        (phi01, phi02, phi03) = (minimized_phi[0], minimized_phi[1], minimized_phi[2])

        return (phi01, phi02, phi03, min_potential)


###############################################################################
#                              Common parameters                              #
###############################################################################
NUMBER_OF_PHI_POINTS = 10
NROWSCOLS = 10
ALPHA = 1
LOWER = -0.5
UPPER = 1.5

grid = np.linspace(LOWER * 2 * pi, UPPER * 2 * pi, NROWSCOLS)
(grid_x, grid_y) = np.meshgrid(grid, grid)

(phi01_grid, phi02_grid, phi03_grid, potential_grid,) = Miminize().minimize_vectorized(
    L=grid_x, R=grid_y, alpha=ALPHA, number_of_phi_points=NUMBER_OF_PHI_POINTS,
)

from copy import deepcopy
from typing import Union

import numpy as np
from femflow.numerics.linear_algebra import fast_diagonal_inverse
from scipy.sparse.csr import csr_matrix

from .integrator import Integrator


class ExplicitCentralDifferenceMethod(Integrator):
    def __init__(
        self,
        dt: float,
        point_mass: float,
        stiffness: Union[np.ndarray, csr_matrix],
        initial_displacement: np.ndarray,
        initial_force: np.ndarray,
    ):
        super(ExplicitCentralDifferenceMethod, self).__init__()
        self.dt = dt

        self.point_mass = point_mass

        self.stiffness = stiffness

        self.initial_displacement = initial_displacement
        self.initial_force = initial_force

        self.mass_matrix = csr_matrix(np.eye(stiffness.shape[0]))
        self.inverse_mass_matrix = deepcopy(self.mass_matrix)
        self.inverse_mass_matrix = fast_diagonal_inverse(self.inverse_mass_matrix)

        self.a0 = 1 / dt ** 2
        self.a1 = 1 / 2 * dt
        self.a2 = 2 * self.a0
        self.a3 = 1 / self.a2

    def integrate(self, forces: np.ndarray) -> np.ndarray:
        return np.array([])

    def _compute_effective_mass_matrix(self):
        pass

    def _compute_rayleigh_damping(self, mu: float, lambda_: float):
        self.damping_matrix = mu * self.mass_matrix + lambda_ * self.stiffness

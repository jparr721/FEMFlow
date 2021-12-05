from copy import deepcopy
from typing import Union

import numpy as np
from femflow.numerics.linear_algebra import fast_diagonal_inverse
from scipy.sparse.csr import csr_matrix
from scipy.sparse.linalg import inv

from .integrator import Integrator


class ExplicitCentralDifferenceMethod(Integrator):
    def __init__(
        self,
        dt: float,
        point_mass: Union[float, csr_matrix],
        stiffness: Union[np.ndarray, csr_matrix],
        initial_displacement: np.ndarray,
        initial_force: np.ndarray,
        *,
        rayleigh_lambda=0.5,
        rayleigh_mu=0.5
    ):
        super(ExplicitCentralDifferenceMethod, self).__init__()
        self.dt = dt

        self.point_mass = point_mass

        self.stiffness = stiffness

        self.initial_displacement = initial_displacement
        self.initial_force = initial_force

        if type(point_mass) != csr_matrix:
            self.mass_matrix = csr_matrix(np.eye(stiffness.shape[0]))
        else:
            self.mass_matrix = point_mass

        self.inverse_mass_matrix = deepcopy(self.mass_matrix)
        self.inverse_mass_matrix = fast_diagonal_inverse(self.inverse_mass_matrix)

        self.a0 = 1 / dt ** 2
        self.a1 = 1 / 2 * dt
        self.a2 = 2 * self.a0
        self.a3 = 1 / self.a2

        self.velocity = np.zeros(initial_displacement.size)
        self.acceleration = self.inverse_mass_matrix * initial_force

        self.rayleigh_lambda = rayleigh_lambda
        self.rayleigh_mu = rayleigh_mu

        self._set_last_position(initial_displacement)
        self._compute_rayleigh_damping()
        self._compute_effective_mass_matrix()

    def integrate(self, forces: np.ndarray, displacements: np.ndarray) -> np.ndarray:
        effective_load = (
            forces
            - (self.stiffness - self.a2 * self.mass_matrix) * displacements
            - (self.a0 * self.mass_matrix - self.a1 * self.damping_matrix) * self.previous_position
        )

        next_displacements = self.effective_mass_matrix * effective_load

        self.acceleration = self.a0 * (self.previous_position - 2 * displacements + next_displacements)
        self.velocity = self.a1 * (-self.previous_position + next_displacements)
        self.previous_position = displacements

        return next_displacements

    def _set_last_position(self, positions: np.ndarray):
        self.previous_position = positions - self.dt * self.velocity + self.a3 * self.acceleration

    def _compute_effective_mass_matrix(self):
        self.effective_mass_matrix = self.a0 * self.mass_matrix + self.a1 * self.damping_matrix
        self.effective_mass_matrix = inv(self.effective_mass_matrix)

    def _compute_rayleigh_damping(self):
        self.damping_matrix = self.rayleigh_mu * self.mass_matrix + self.rayleigh_lambda * self.stiffness

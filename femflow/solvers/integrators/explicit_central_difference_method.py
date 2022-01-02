from copy import deepcopy

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import inv

from femflow.numerics.linear_algebra import fast_diagonal_inverse


class ExplicitCentralDifferenceMethod(object):
    def __init__(
        self,
        dt: float,
        mass_matrix: csr_matrix,
        stiffness: csr_matrix,
        initial_displacement: np.ndarray,
        initial_force: np.ndarray,
        *,
        rayleigh_lambda=0.5,
        rayleigh_mu=0.5,
    ):
        self.dt = dt

        self.stiffness = stiffness

        self.initial_displacement = initial_displacement
        self.initial_force = initial_force

        self.mass_matrix = mass_matrix
        if not isinstance(self.mass_matrix, csr_matrix):
            raise TypeError("Mass matrix must be sparse")

        self.inverse_mass_matrix = deepcopy(self.mass_matrix)
        fast_diagonal_inverse(self.inverse_mass_matrix)

        self.a0 = 1 / dt ** 2
        self.a1 = 1 / 2 * dt
        self.a2 = 2 * self.a0
        self.a3 = 1 / self.a2

        self.velocity = np.zeros(initial_displacement.size)
        self.acceleration = self.inverse_mass_matrix.dot(initial_force)

        self.rayleigh_lambda = rayleigh_lambda
        self.rayleigh_mu = rayleigh_mu

        self.a0mass_matrix = self.a0 * self.mass_matrix
        self.a2mass_matrix = self.a2 * self.mass_matrix

        self._set_last_position(initial_displacement)
        self._compute_rayleigh_damping()
        self._compute_effective_mass_matrix()

        self.a1damping_matrix = self.a1 * self.damping_matrix

        self.stiff_mass_diff = self.stiffness - self.a2mass_matrix
        self.a0mass_damping_diff = self.a0mass_matrix - self.a1damping_matrix

    def integrate(self, forces: np.ndarray, displacements: np.ndarray) -> np.ndarray:
        v1 = self.stiff_mass_diff.dot(displacements.T)
        v2 = self.a0mass_damping_diff.dot(self.previous_position.T)

        effective_load = forces - v1 - v2

        next_displacements = self.effective_mass_matrix * effective_load

        self.acceleration = self.a0 * (
            self.previous_position - 2 * displacements + next_displacements
        )
        self.velocity = self.a1 * (-self.previous_position + next_displacements)
        self.previous_position = displacements

        return next_displacements

    def _set_last_position(self, positions: np.ndarray):
        self.previous_position = (
            positions - self.dt * self.velocity + self.a3 * self.acceleration
        )

    def _compute_effective_mass_matrix(self):
        if self.rayleigh_mu == 0 and self.rayleigh_lambda == 0:
            self.effective_mass_matrix = self.a0mass_matrix.copy()
            fast_diagonal_inverse(self.effective_mass_matrix)
        else:
            self.effective_mass_matrix = csr_matrix(
                self.a0 * self.mass_matrix + self.a1 * self.damping_matrix
            )
            self.effective_mass_matrix = inv(self.effective_mass_matrix)

    def _compute_rayleigh_damping(self):
        self.damping_matrix = (
            self.rayleigh_mu * self.mass_matrix + self.rayleigh_lambda * self.stiffness
        )

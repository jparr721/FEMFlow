import time
from copy import deepcopy
from typing import Union

import numpy as np
from loguru import logger
from numerics.linear_algebra import fast_diagonal_inverse
from scipy.sparse.csc import csc_matrix
from scipy.sparse.linalg import inv

from .integrator import Integrator


class ExplicitCentralDifferenceMethod(Integrator):
    def __init__(
        self,
        dt: float,
        point_mass: Union[float, csc_matrix],
        stiffness: Union[np.ndarray, csc_matrix],
        initial_displacement: np.ndarray,
        initial_force: np.ndarray,
        *,
        rayleigh_lambda=0.5,
        rayleigh_mu=0.5,
    ):
        super(ExplicitCentralDifferenceMethod, self).__init__()
        self.dt = dt

        self.point_mass = point_mass

        self.stiffness = stiffness

        self.initial_displacement = initial_displacement
        self.initial_force = initial_force

        if type(point_mass) != csc_matrix:
            self.mass_matrix = csc_matrix(np.eye(stiffness.shape[0])) * point_mass
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
        # The shapes don't work right due to scipy's ordering, so we transpose and remove the dimension
        v1 = np.array((self.stiffness - self.a2 * self.mass_matrix).dot(displacements)).T.reshape(-1)
        v2 = np.array(
            (self.a0 * self.mass_matrix - self.a1 * self.damping_matrix).dot(self.previous_position)
        ).T.reshape(-1)

        effective_load = forces - v1 - v2

        next_displacements = self.effective_mass_matrix * effective_load

        self.acceleration = self.a0 * (self.previous_position - 2 * displacements + next_displacements)
        self.velocity = self.a1 * (-self.previous_position + next_displacements)
        self.previous_position = displacements

        return next_displacements

    def _set_last_position(self, positions: np.ndarray):
        self.previous_position = positions - self.dt * self.velocity + self.a3 * self.acceleration

    def _compute_effective_mass_matrix(self):
        start = time.time()
        self.effective_mass_matrix = csc_matrix(self.a0 * self.mass_matrix + self.a1 * self.damping_matrix)
        if self.rayleigh_mu == 0 and self.rayleigh_lambda == 0:
            self.effective_mass_matrix = fast_diagonal_inverse(self.effective_mass_matrix)
        else:
            self.effective_mass_matrix = inv(self.effective_mass_matrix)
        end = time.time()
        logger.debug(f"It took {end - start}s to compute the emm")

    def _compute_rayleigh_damping(self):
        self.damping_matrix = self.rayleigh_mu * self.mass_matrix + self.rayleigh_lambda * self.stiffness

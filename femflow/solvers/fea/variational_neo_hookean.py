import numpy as np

from .boundary_conditions import BoundaryConditions


class VariationalNeoHookean(object):
    def __init__(self, v: np.ndarray, f: np.ndarray, t: np.ndarray, E: float, nu: float):
        # Geometry
        self.v = v
        self.f = f
        self.t = t

        self.E = E
        self.nu = nu
        self.lambda_ = (
            0.5 * (self.E * self.nu) / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        )
        self.mu = 0.5 * self.E / (2.0 * (1.0 + self.nu))

        self.k_selected = 1e5

    def simulate(self, q: np.ndarray, qdot: np.ndarray, dt: float, t: float):
        pass


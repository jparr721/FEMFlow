from typing import List

import numpy as np

from femflow.solvers.mpm import three_d
from femflow.solvers.mpm.particle import Particle


def make_mls_mpm_coefficients(lenx: int, dim: int):
    v = np.zeros((lenx, dim), dtype=np.float64)
    F = np.array([np.eye(dim, dtype=np.float64) for _ in range(lenx)])
    C = np.zeros((lenx, dim, dim), dtype=np.float64)
    Jp = np.ones((lenx, 1), dtype=np.float64)

    return v, F, C, Jp


# self.mass = mass
# self.volume = volume
# self.hardening = hardening

# self.E = E
# self.nu = nu

# self.mu_0 = E / (2 * (1 + nu))
# self.lambda_0 = E * nu / ((1 + nu) * (1 - 2 * nu))

# self.gravity = gravity
# self.dt = dt
# self.grid_resolution = grid_resolution

# # dx is always 1 / grid_resolution
# self.dx = 1 / grid_resolution
# self.inv_dx = 1 / self.dx
# self.model = model

# self.tightening_coeff = tightening_coeff


def solve_mls_mpm_3d(
    res: int,
    inv_dx: float,
    hardening: float,
    dx: float,
    dt: float,
    volume: float,
    gravity: float,
    particles: List[Particle],
    v: np.ndarray,
    F: np.ndarray,
    C: np.ndarray,
    Jp: np.ndarray,
):
    dres = res + 1
    grid_velocity = np.zeros((dres, dres, dres, 3))
    grid_mass = np.zeros((dres, dres, dres, 1))

    model = "neo_hookean"
    three_d.p2g(
        inv_dx,
        hardening,
        dx,
        dt,
        volume,
        grid_velocity,
        grid_mass,
        particles,
        v,
        F,
        C,
        Jp,
        model,
    )

    three_d.grid_op(
        res, dx, dt, gravity, grid_velocity, grid_mass,
    )

    three_d.g2p(inv_dx, dt, grid_velocity, particles, v, F, C, Jp, model)

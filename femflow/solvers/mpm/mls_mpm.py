import numpy as np
from loguru import logger

from femflow.solvers.mpm import three_d
from femflow.solvers.mpm.two_d import g2p, grid_op, p2g

from .parameters import Parameters


def solve_mls_mpm_2d(
    params: Parameters,
    x: np.ndarray,
    v: np.ndarray,
    F: np.ndarray,
    C: np.ndarray,
    Jp: np.ndarray,
):
    grid_velocity = np.zeros((params.grid_resolution + 1, params.grid_resolution + 1, 2))
    grid_mass = np.zeros((params.grid_resolution + 1, params.grid_resolution + 1, 1))

    p2g(
        params.inv_dx,
        params.hardening,
        params.mu_0,
        params.lambda_0,
        params.mass,
        params.dx,
        params.dt,
        params.volume,
        grid_velocity,
        grid_mass,
        x,
        v,
        F,
        C,
        Jp,
        params.model,
    )
    grid_op(params.grid_resolution, params.dt, params.gravity, grid_velocity, grid_mass)
    g2p(params.inv_dx, params.dt, grid_velocity, x, v, F, C, Jp, params.model)


def solve_mls_mpm_3d(
    params: Parameters,
    x: np.ndarray,
    v: np.ndarray,
    F: np.ndarray,
    C: np.ndarray,
    Jp: np.ndarray,
):
    grid_velocity = np.zeros(
        (
            params.grid_resolution + 1,
            params.grid_resolution + 1,
            params.grid_resolution + 1,
            3,
        )
    )

    grid_mass = np.zeros(
        (
            params.grid_resolution + 1,
            params.grid_resolution + 1,
            params.grid_resolution + 1,
            1,
        )
    )

    three_d.p2g(
        params.inv_dx,
        params.hardening,
        params.mu_0,
        params.lambda_0,
        params.mass,
        params.dx,
        params.dt,
        params.volume,
        grid_velocity,
        grid_mass,
        x,
        v,
        F,
        C,
        Jp,
        params.model,
    )

    three_d.grid_op(
        params.grid_resolution, params.dt, params.gravity, grid_velocity, grid_mass
    )

    three_d.g2p(params.inv_dx, params.dt, grid_velocity, x, v, F, C, Jp, params.model)

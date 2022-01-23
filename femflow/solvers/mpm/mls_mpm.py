import numpy as np

from femflow.solvers.mpm import three_d
from femflow.solvers.mpm.two_d import g2p, grid_op, p2g

from .parameters import Parameters


def make_mls_mpm_coefficients(lenx: int, dim: int):
    v = np.zeros((lenx, dim), dtype=np.float64)
    F = np.array([np.eye(dim, dtype=np.float64) for _ in range(lenx)])
    C = np.zeros((lenx, dim, dim), dtype=np.float64)
    Jp = np.ones((lenx, 1), dtype=np.float64)

    return v, F, C, Jp


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
        params.grid_resolution,
        params.dx,
        params.dt,
        params.gravity,
        grid_velocity,
        grid_mass,
    )

    three_d.g2p(params.inv_dx, params.dt, grid_velocity, x, v, F, C, Jp, params.model)

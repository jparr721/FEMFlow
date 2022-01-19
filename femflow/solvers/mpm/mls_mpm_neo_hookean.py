import numba as nb
import numpy as np

from .parameters import Parameters


@nb.njit
def sqr(v: np.ndarray):
    return np.power(v, 2)


@nb.njit
def _vec(dim: int, v):
    return np.ones(dim) * v


@nb.njit
def _mat(dim: int, v):
    return np.eye(dim) * v


@nb.njit
def _p2g_2d(
    dx: float,
    inv_dx: float,
    mu_0: float,
    lambda_0: float,
    mass: float,
    dt: float,
    volume: float,
    grid: np.ndarray,
    x: np.ndarray,
    v: np.ndarray,
    F: np.ndarray,
    C: np.ndarray,
):
    for p in range(len(x)):
        # Compute the neo-hookean stress
        # P = mu * (F - F^-T) + lambda * log(J) * F^-T
        F_ = F[p]
        J = np.linalg.det(F_)

        # Volume is a function of the jacobian
        volume = volume * J

        F_inv_T = np.linalg.inv(F_.T)
        P = mu_0 * (F_ - F_inv_T) + lambda_0 * (np.log(J) * F_inv_T)

        # Neo-hookean cauchy stress
        stress = (1.0 / J) * (P @ F_.T)

        # Fused momentum and force for the APIC MLS-MPM step with quadric weights
        affine = -volume * 4 * stress * dt

        # Quadric weight function
        base_coord = (x[p] * inv_dx - _vec(2, 0.5)).astype(np.int64)
        fx = (x[p] * inv_dx - base_coord).astype(np.float64)
        w = [0.5 * sqr(1.5 - fx), 0.75 - sqr(fx - 1.0), 0.5 * sqr(fx - 0.5)]

        for i in range(3):
            for j in range(3):
                dpos = (np.array((i, j)) - fx) * dx
                Q = C[p] @ dpos

                weight = w[i][0] * w[j][1]
                wm = weight * mass

                grid[base_coord[0] + i, base_coord[1] + j][:2] += wm * (v[p] + Q)
                grid[base_coord[0] + i, base_coord[1] + j][:2] += (affine * weight) @ dpos
                grid[base_coord[0] + i, base_coord[1] + j][2] += wm


@nb.njit(parallel=True)
def _gv_2d(grid_resolution: int, dt: float, gravity: float, grid: np.ndarray):
    boundary = 0.05
    for i in range(grid_resolution + 1):
        for j in range(grid_resolution + 1):
            if grid[i, j][2] > 0:
                grid[i, j] /= grid[i, j][2]
                grid[i, j][1] += dt * gravity
                x = i / grid_resolution
                y = j / grid_resolution
                if x < boundary or x > 1 - boundary or y > 1 - boundary:
                    grid[i, j] = 0
                if y < boundary:
                    grid[i, j][1] = max(0.0, grid[i, j][1])


@nb.njit
def _g2p_2d(
    inv_dx: float,
    dt: float,
    grid: np.ndarray,
    x: np.ndarray,
    v: np.ndarray,
    F: np.ndarray,
    C: np.ndarray,
):
    for p in range(len(x)):
        v[p] = 0

        # Quadric weight function
        base_coord = (x[p] * inv_dx - _vec(2, 0.5)).astype(np.int64)
        fx = x[p] * inv_dx - base_coord.astype(np.float64)
        w = [0.5 * sqr(1.5 - fx), 0.75 - sqr(fx - 1.0), 0.5 * sqr(fx - 0.5)]

        B = np.zeros((2, 2))
        for gx in range(3):
            for gy in range(3):
                weight = w[gx][0] * w[gy][1]
                dpos = np.array((gx, gy)) - fx
                grid_v = grid[base_coord[0] + gx, base_coord[1] + gy][:2]
                v[p] += grid_v * weight
                B += np.outer(weight * grid_v, dpos)
        C[p] = B * 4

        # Advection
        x[p] += v[p] * dt

        # Clamp the values
        # x[p] = np.clip(x[p], 0.05, 0.95)
        F[p] = (_mat(2, 1) + dt * C[p]) @ F[p]


def solve_mls_mpm_neo_hookean_2d(
    params: Parameters, x: np.ndarray, v: np.ndarray, F: np.ndarray, C: np.ndarray,
):
    grid = np.zeros((params.grid_resolution + 1, params.grid_resolution + 1, 3))
    _p2g_2d(
        params.dx,
        params.inv_dx,
        params.mu_0,
        params.lambda_0,
        params.mass,
        params.dt,
        params.volume,
        grid,
        x,
        v,
        F,
        C,
    )
    _gv_2d(params.grid_resolution, params.dt, params.gravity, grid)
    _g2p_2d(params.inv_dx, params.dt, grid, x, v, F, C)

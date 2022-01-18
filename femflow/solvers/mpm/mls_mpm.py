import functools
from typing import List

import numba as nb
import numpy as np
from scipy.linalg import polar

from .parameters import Parameters
from .particle import Particle


def sqr(v):
    return pow(v, 2)


def _vec(dim: int, v):
    return np.ones(dim) * v


def _mat(dim: int, v):
    return np.eye(dim) * v


@nb.njit(parallel=True)
def _gv_2d(grid_resolution: int, dt: float, gravity: float, grid: np.ndarray):
    for i in nb.prange(grid_resolution + 1):
        for j in nb.prange(grid_resolution + 1):
            if grid[i, j][2] > 0:
                grid[i, j] /= grid[i, j][2]
                grid[i, j] += dt * np.array((0, gravity, 0))
                boundary = 0.05
                x = i / grid_resolution
                y = j / grid_resolution
                if x < boundary or x > 1 - boundary or y > 1 - boundary:
                    grid[i, j] = np.zeros(3)
                if y < boundary:
                    grid[i, j][1] = max(0.0, grid[i, j][1])


@nb.njit(parallel=True)
def _gv_3d(grid_resolution: int, dt: float, gravity: float, grid: np.ndarray):
    for i in nb.prange(grid_resolution + 1):
        for j in nb.prange(grid_resolution + 1):
            for k in nb.prange(grid_resolution + 1):
                if grid[i, j, k][3] > 0:
                    grid[i, j, k] /= grid[i, j, k][3]
                    grid[i, j, k] += dt * np.array((0, gravity, 0, 0))
                    boundary = 0.05
                    x = i / grid_resolution
                    y = j / grid_resolution
                    z = k / grid_resolution
                    if (
                        x < boundary
                        or x > 1 - boundary
                        or y > 1 - boundary
                        or z < boundary
                        or z > 1 - boundary
                    ):
                        grid[i, j, k] = np.zeros(4)
                    if y < boundary:
                        grid[i, j, k][1] = max(0.0, grid[i, j, k][1])


def _p2g_2d(
    inv_dx: float,
    hardening: float,
    mu_0: float,
    lambda_0: float,
    mass: float,
    dx: float,
    dt: float,
    volume: float,
    grid: np.ndarray,
    x: np.ndarray,
    v: np.ndarray,
    F: np.ndarray,
    C: np.ndarray,
    Jp: np.ndarray,
):
    for p in range(len(x)):
        base_coord = (x[p] * inv_dx - _vec(2, 0.5)).astype(int)
        fx = (x[p] * inv_dx - base_coord).astype(float)

        w = np.array(
            [
                _vec(2, 0.5) * sqr(_vec(2, 1.5) - fx),
                _vec(2, 0.75) - sqr(fx - _vec(2, 1.0)),
                _vec(2, 0.5) * sqr(fx - _vec(2, 0.5)),
            ]
        )

        e = np.exp(hardening * (1 - Jp[p]))
        mu = mu_0 * e
        lambda_ = lambda_0 * e

        J = np.linalg.det(F[p])
        r, _ = polar(F[p])

        D_inv = 4 * inv_dx * inv_dx

        PF = np.matmul(2 * mu * (F[p] - r), F[p].T) + lambda_ * (J - 1) * J
        stress = -(dt * volume) * (D_inv * PF)

        affine = stress + mass * C[p]

        for i in range(3):
            for j in range(3):
                dpos = (np.array((i, j)) - fx) * dx
                mass_x_velocity = np.array((*(v[p] * mass), mass))
                weight = w[i][0] * w[j][1]
                grid[base_coord[0] + i, base_coord[1] + j] += weight * (
                    mass_x_velocity + np.array((*(np.matmul(affine, dpos)), 0))
                )


def _g2p_2d(
    inv_dx: float,
    dt: float,
    grid: np.ndarray,
    x: np.ndarray,
    v: np.ndarray,
    F: np.ndarray,
    C: np.ndarray,
    Jp: np.ndarray,
):
    for p in range(len(x)):
        base_coord = (x[p] * inv_dx - _vec(2, 0.5)).astype(int)
        fx = x[p] * inv_dx - base_coord.astype(float)
        w = np.array(
            [
                _vec(2, 0.5) * sqr(_vec(2, 1.5) - fx),
                _vec(2, 0.75) - sqr(fx - _vec(2, 1.0)),
                _vec(2, 0.5) * sqr(fx - _vec(2, 0.5)),
            ]
        )
        C[p] = _mat(2, 0.0)
        v[p] = _vec(2, 0.0)

        for i in range(3):
            for j in range(3):
                dpos = np.array((i, j)) - fx
                grid_v = grid[base_coord[0] + i, base_coord[1] + j][:2]
                weight = w[i][0] * w[j][1]
                v[p] += weight * grid_v
                C[p] += 4 * inv_dx * np.outer(weight * grid_v, dpos)

        x[p] += dt * v[p]
        F_ = (_mat(2, 1) + dt * C[p]) * F[p]

        U, sig, V = np.linalg.svd(F_)
        sig = np.eye(2) * sig

        for i in range(2):
            sig[i, i] = np.clip(sig[i, i], 1.0 - 2.5e-2, 1.0 + 7.5e-3)

        old_J = np.linalg.det(F_)
        F_ = U @ sig @ V.T

        Jp_new = np.clip(Jp[p] * old_J / np.linalg.det(F_), 0.6, 20.0)
        Jp[p] = Jp_new
        F[p] = F_


def solve_mls_mpm_2d(
    params: Parameters,
    x: np.ndarray,
    v: np.ndarray,
    F: np.ndarray,
    C: np.ndarray,
    Jp: np.ndarray,
):
    grid = np.zeros((params.grid_resolution + 1, params.grid_resolution + 1, 3))

    # for p in particles:
    #     base_coord = (p.x * params.inv_dx - vec(0.5)).astype(int)
    #     fx = (p.x * params.inv_dx - base_coord).astype(float)

    #     w = np.array(
    #         [
    #             vec(0.5) * sqr(vec(1.5) - fx),
    #             vec(0.75) - sqr(fx - vec(1.0)),
    #             vec(0.5) * sqr(fx - vec(0.5)),
    #         ]
    #     )

    #     e = np.exp(params.hardening * (1 - p.Jp))
    #     mu = params.mu_0 * e
    #     lambda_ = params.lambda_0 * e

    #     J = np.linalg.det(p.F)
    #     r, _ = polar(p.F)

    #     D_inv = 4 * params.inv_dx * params.inv_dx

    #     PF = np.matmul(2 * mu * (p.F - r), p.F.T) + lambda_ * (J - 1) * J
    #     stress = -(params.dt * params.volume) * (D_inv * PF)

    #     affine = stress + params.mass * p.C

    #     for i in range(3):
    #         for j in range(3):
    #             dpos = (np.array((i, j)) - fx) * params.dx
    #             mass_x_velocity = np.array((*(p.v * params.mass), params.mass))
    #             weight = w[i][0] * w[j][1]
    #             grid[base_coord[0] + i, base_coord[1] + j] += weight * (
    #                 mass_x_velocity + np.array((*(np.matmul(affine, dpos)), 0))
    #             )

    _p2g_2d(
        params.inv_dx,
        params.hardening,
        params.mu_0,
        params.lambda_0,
        params.mass,
        params.dx,
        params.dt,
        params.volume,
        grid,
        x,
        v,
        F,
        C,
        Jp,
    )
    _gv_2d(params.grid_resolution, params.dt, params.gravity, grid)
    _g2p_2d(params.inv_dx, params.dt, grid, x, v, F, C, Jp)

    # for p in particles:
    #     base_coord = (p.x * params.inv_dx - vec(0.5)).astype(int)
    #     fx = p.x * params.inv_dx - base_coord.astype(float)
    #     w = np.array(
    #         [
    #             vec(0.5) * sqr(vec(1.5) - fx),
    #             vec(0.75) - sqr(fx - vec(1.0)),
    #             vec(0.5) * sqr(fx - vec(0.5)),
    #         ]
    #     )
    #     p.C = mat(0.0)
    #     p.v = vec(0.0)

    #     for i in range(3):
    #         for j in range(3):
    #             dpos = np.array((i, j)) - fx
    #             grid_v = grid[base_coord[0] + i, base_coord[1] + j][:2]
    #             weight = w[i][0] * w[j][1]
    #             p.v += weight * grid_v
    #             p.C += 4 * params.inv_dx * np.outer(weight * grid_v, dpos)

    #     p.x += params.dt * p.v
    #     F = (mat(1) + params.dt * p.C) * p.F

    #     U, sig, V = np.linalg.svd(F)
    #     sig = np.eye(2) * sig

    #     for i in range(2):
    #         sig[i, i] = np.clip(sig[i, i], 1.0 - 2.5e-2, 1.0 + 7.5e-3)

    #     old_J = np.linalg.det(F)
    #     F = U @ sig @ V.T

    #     Jp_new = np.clip(p.Jp * old_J / np.linalg.det(F), 0.6, 20.0)
    #     p.Jp = Jp_new
    #     p.F = F


def solve_mls_mpm_3d(params: Parameters, particles: List[Particle]):
    vec = functools.partial(_vec, 3)
    mat = functools.partial(_mat, 3)

    grid = np.zeros(
        (
            params.grid_resolution + 1,
            params.grid_resolution + 1,
            params.grid_resolution + 1,
            4,
        )
    )

    for p in particles:
        base_coord = (p.x * params.inv_dx - vec(0.5)).astype(int)
        fx = (p.x * params.inv_dx - base_coord).astype(float)

        w = np.array(
            [
                vec(0.5) * sqr(vec(1.5) - fx),
                vec(0.75) - sqr(fx - vec(1.0)),
                vec(0.5) * sqr(fx - vec(0.5)),
            ]
        )

        e = np.exp(params.hardening * (1 - p.Jp))
        mu = params.mu_0 * e
        lambda_ = params.lambda_0 * e

        J = np.linalg.det(p.F)
        r, _ = polar(p.F)

        D_inv = 4 * params.inv_dx * params.inv_dx

        PF = np.matmul(2 * mu * (p.F - r), p.F.T) + lambda_ * (J - 1) * J
        stress = -(params.dt * params.volume) * (D_inv * PF)

        affine = stress + params.mass * p.C

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    dpos = (np.array((i, j, k)) - fx) * params.dx
                    mass_x_velocity = np.array((*(p.v * params.mass), params.mass))
                    weight = w[i][0] * w[j][1] * w[k][2]
                    grid[
                        base_coord[0] + i, base_coord[1] + j, base_coord[2] + k
                    ] += weight * (mass_x_velocity + np.array((*(affine @ dpos), 0.0)))

    _gv_3d(params.grid_resolution, params.dt, params.gravity, grid)

    for p in particles:
        base_coord = (p.x * params.inv_dx - vec(0.5)).astype(int)
        fx = p.x * params.inv_dx - base_coord.astype(float)
        w = np.array(
            [
                vec(0.5) * sqr(vec(1.5) - fx),
                vec(0.75) - sqr(fx - vec(1.0)),
                vec(0.5) * sqr(fx - vec(0.5)),
            ]
        )
        p.C = mat(0.0)
        p.v = vec(0.0)

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    dpos = np.array((i, j, k)) - fx
                    grid_v = grid[
                        base_coord[0] + i, base_coord[1] + j, base_coord[2] + k
                    ][:3]
                    weight = w[i][0] * w[j][1] * w[k][2]
                    p.v += weight * grid_v
                    p.C += 4 * params.inv_dx * np.outer(weight * grid_v, dpos)

        p.x += params.dt * p.v
        F = (mat(1) + params.dt * p.C) * p.F

        U, sig, V = np.linalg.svd(F)
        sig = np.eye(3) * sig

        for i in range(3):
            sig[i, i] = np.clip(sig[i, i], 1.0 - 2.5e-2, 1.0 + 7.5e-3)

        old_J = np.linalg.det(F)
        F = U @ sig @ V.T

        Jp_new = np.clip(p.Jp * old_J / np.linalg.det(F), 0.6, 20.0)
        p.Jp = Jp_new
        p.F = F

import functools
from typing import List

import numba as nb
import numpy as np
from scipy.linalg import polar

from femflow.numerics.linear_algebra import polar_decomp_2d

from .parameters import Parameters
from .particle import Particle


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
def _gv_2d(grid_resolution: int, dt: float, gravity: float, grid: np.ndarray):
    for i in range(grid_resolution + 1):
        for j in range(grid_resolution + 1):
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


@nb.njit
def _gv_3d(
    grid_resolution: int,
    dt: float,
    gravity: float,
    grid_velocity: np.ndarray,
    grid_mass: np.ndarray,
):
    for i in range(grid_resolution + 1):
        for j in range(grid_resolution + 1):
            for k in range(grid_resolution + 1):
                if grid_mass[i, j, k] > 0:
                    grid_velocity[i, j, k] /= grid_mass[i, j, k]
                    grid_velocity[i, j, k] += dt * np.array((0, gravity, 0))
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
                        grid_velocity[i, j, k] = np.zeros(3)
                    if y < boundary:
                        grid_velocity[i, j, k][1] = max(0.0, grid_velocity[i, j, k][1])


@nb.njit
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
    model: str = "neo_hookean",
):
    for p in range(len(x)):
        base_coord = (x[p] * inv_dx - _vec(2, 0.5)).astype(np.int64)
        fx = (x[p] * inv_dx - base_coord).astype(np.float64)

        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]

        e = np.exp(hardening * (1 - Jp[p]))[0]
        if model == "neo_hookean":
            e = 0.3
        mu = mu_0 * e
        lambda_ = lambda_0 * e

        J = np.linalg.det(F[p])
        r, _ = polar_decomp_2d(F[p])

        D_inv = 4 * inv_dx * inv_dx

        PF = (2 * mu * (F[p] - r) @ F[p].T) + lambda_ * (J - 1) * J
        stress = -(dt * volume) * (D_inv * PF)

        affine = stress + mass * C[p]

        for i in range(3):
            for j in range(3):
                dpos = (np.array((i, j)) - fx) * dx
                mv = v[p] * mass
                mv = np.array((mv[0], mv[1], mass))

                weight = w[i][0] * w[j][1]

                adpos = affine @ dpos
                adpos = np.array((adpos[0], adpos[1], 0))
                grid[base_coord[0] + i, base_coord[1] + j] += weight * (mv + adpos)


@nb.jit
def _g2p_2d(
    inv_dx: float,
    dt: float,
    grid: np.ndarray,
    x: np.ndarray,
    v: np.ndarray,
    F: np.ndarray,
    C: np.ndarray,
    Jp: np.ndarray,
    model: str = "neo_hookean",
):
    for p in range(len(x)):
        base_coord = (x[p] * inv_dx - _vec(2, 0.5)).astype(np.int64)
        fx = x[p] * inv_dx - base_coord.astype(np.float64)

        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]

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
        F_ = (_mat(2, 1) + dt * C[p]) @ F[p]

        U, sig, V = np.linalg.svd(F_)
        if model == "snow":
            sig = np.clip(sig, 1.0 - 2.5e-2, 1.0 + 7.5e-3)
        sig = np.eye(2) * sig

        old_J = np.linalg.det(F_)
        F_ = U @ sig @ V.T

        det = np.linalg.det(F_)
        Jp_new = np.clip(Jp[p] * old_J / det, 0.6, 20.0)
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
        params.model,
    )
    _gv_2d(params.grid_resolution, params.dt, params.gravity, grid)
    _g2p_2d(params.inv_dx, params.dt, grid, x, v, F, C, Jp, params.model)


def solve_mls_mpm_3d(params: Parameters, particles: List[Particle]):
    vec = functools.partial(_vec, 3)
    mat = functools.partial(_mat, 3)

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
                    mv = p.v * params.mass

                    weight = w[i][0] * w[j][1] * w[k][2]

                    grid_velocity[
                        base_coord[0] + i, base_coord[1] + j, base_coord[2] + k
                    ] += weight * (mv + affine @ dpos)
                    grid_mass[
                        base_coord[0] + i, base_coord[1] + j, base_coord[2] + k
                    ] += (weight * params.mass)

    _gv_3d(params.grid_resolution, params.dt, params.gravity, grid_velocity, grid_mass)

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
                    grid_v = grid_velocity[
                        base_coord[0] + i, base_coord[1] + j, base_coord[2] + k
                    ]
                    weight = w[i][0] * w[j][1] * w[k][2]
                    p.v += weight * grid_v
                    p.C += 4 * params.inv_dx * np.outer(weight * grid_v, dpos)

        p.x += params.dt * p.v
        F = (mat(1) + params.dt * p.C) @ p.F

        U, sig, V = np.linalg.svd(F)
        sig = np.clip(sig, 1.0 - 2.5e-2, 1.0 + 7.5e-3)
        sig = np.eye(3) * sig

        # for i in range(3):
        #     sig[i, i] = np.clip(sig[i, i], 1.0 - 2.5e-2, 1.0 + 7.5e-3)

        old_J = np.linalg.det(F)
        F = U @ sig @ V.T

        Jp_new = np.clip(p.Jp * old_J / np.linalg.det(F), 0.6, 20.0)
        p.Jp = Jp_new
        p.F = F

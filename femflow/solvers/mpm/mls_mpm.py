import numpy as np
from scipy.linalg import polar

from .particle import Particle


def sqr(v):
    return pow(v, 2)


def vec(v):
    return np.array([v, v])


def mat(v):
    return np.eye(2) * v


def advance(particles: List[Particle]):
    grid = np.zeros((n + 1, n + 1, 3))

    for p in particles:
        base_coord = (p.x * inv_dx - vec(0.5)).astype(int)
        fx = (p.x * inv_dx - base_coord).astype(float)

        w = np.array(
            [
                vec(0.5) * sqr(vec(1.5) - fx),
                vec(0.75) - sqr(fx - vec(1.0)),
                vec(0.5) * sqr(fx - vec(0.5)),
            ]
        )

        e = np.exp(hardening * (1 - p.Jp))
        mu = mu_0 * e
        lambda_ = lambda_0 * e

        J = np.linalg.det(p.F)
        r, _ = polar(p.F)

        D_inv = 4 * inv_dx * inv_dx

        PF = np.matmul(2 * mu * (p.F - r), p.F.T) + lambda_ * (J - 1) * J
        stress = -(dt * vol) * (D_inv * PF)

        affine = stress + particle_mass * p.C

        for i in range(3):
            for j in range(3):
                dpos = (np.array((i, j)) - fx) * dx
                mass_x_velocity = np.array((*(p.v * particle_mass), particle_mass))
                weight = w[i][0] * w[j][1]
                grid[base_coord[0] + i, base_coord[1] + j] += weight * (
                    mass_x_velocity + np.array((*(np.matmul(affine, dpos)), 0))
                )

    for i in range(n + 1):
        for j in range(n + 1):
            if grid[i, j][2] > 0:
                grid[i, j] /= grid[i, j][2]
                grid[i, j] += dt * np.array((0, -200, 0))
                boundary = 0.05
                x = i / n
                y = j / n
                if x < boundary or x > 1 - boundary or y > 1 - boundary:
                    grid[i, j] = np.zeros(3)
                if y < boundary:
                    grid[i, j][1] = max(0.0, grid[i, j][1])

    for p in particles:
        base_coord = (p.x * inv_dx - vec(0.5)).astype(int)
        fx = p.x * inv_dx - base_coord.astype(float)
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
                dpos = np.array((i, j)) - fx
                grid_v = grid[base_coord[0] + i, base_coord[1] + j][:2]
                weight = w[i][0] * w[j][1]
                p.v += weight * grid_v
                p.C += 4 * inv_dx * np.outer(weight * grid_v, dpos)

        p.x += dt * p.v
        F = (mat(1) + dt * p.C) * p.F

        U, sig, V = svd(F)

        for i in range(2):
            sig[i, i] = np.clip(sig[i, i], 1.0 - 2.5e-2, 1.0 + 7.5e-3)

        old_J = np.linalg.det(F)
        F = U @ sig @ V.T

        Jp_new = np.clip(p.Jp * old_J / np.linalg.det(F), 0.6, 20.0)
        p.Jp = Jp_new
        p.F = F

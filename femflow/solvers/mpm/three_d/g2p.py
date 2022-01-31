from typing import List

import numba as nb
import numpy as np
from numba import typed

from femflow.solvers.mpm.particle import Particle
from femflow.solvers.mpm.utils import oob


@nb.njit
def g2p(
    inv_dx: float,
    dt: float,
    grid_velocity: np.ndarray,
    particles: typed.List[Particle],
    v: np.ndarray,
    F: np.ndarray,
    C: np.ndarray,
    Jp: np.ndarray,
    model: str = "neo_hookean",
):
    for p, particle in enumerate(particles):
        base_coord = (particle.pos * inv_dx - 0.5).astype(np.int64)
        if oob(base_coord, grid_velocity.shape[0]):
            raise RuntimeError
        fx = particle.pos * inv_dx - (base_coord).astype(np.float64)

        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]

        C[p] = 0.0
        v[p] = 0.0

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if oob(base_coord, grid_velocity.shape[0], np.array((i, j, k))):
                        raise RuntimeError
                    dpos = np.array((i, j, k)) - fx
                    grid_v = grid_velocity[
                        base_coord[0] + i, base_coord[1] + j, base_coord[2] + k
                    ]
                    weight = w[i][0] * w[j][1] * w[k][2]
                    v[p] += weight * grid_v
                    C[p] += 4 * inv_dx * np.outer(weight * grid_v, dpos)

        particle.pos += dt * v[p]
        F_ = (np.eye(3) + dt * C[p]) @ F[p]

        if model == "snow":
            U, sig, V = np.linalg.svd(F_)
            sig = np.clip(sig, 1.0 - 2.5e-2, 1.0 + 7.5e-3)
            sig = np.eye(3) * sig

            old_J = np.linalg.det(F_)

            F_ = U @ sig @ V.T

            det = np.linalg.det(F_) + 1e-10
            Jp[p] = np.clip(Jp[p] * old_J / det, 0.6, 20.0)
        F[p] = F_

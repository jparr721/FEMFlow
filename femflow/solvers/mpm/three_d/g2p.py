from typing import Tuple

import numba as nb
import numpy as np

from femflow.solvers.mpm.utils import out_of_bounds_3d


@nb.njit
def g2p(
    inv_dx: float,
    dt: float,
    grid_velocity: np.ndarray,
    x: np.ndarray,
    v: np.ndarray,
    F: np.ndarray,
    C: np.ndarray,
    Jp: np.ndarray,
    model: str = "neo_hookean",
    # boundary_forces: Tuple[np.ndarray, set[int]] = (np.zeros(3), {0}),
):
    for p in range(len(x)):
        base_coord = (x[p] * inv_dx - 0.5).astype(np.int64)
        if out_of_bounds_3d(base_coord, grid_velocity.shape[0]):
            raise RuntimeError
        fx = (x[p] * inv_dx - base_coord).astype(np.float64)

        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]

        C[p] = 0.0
        v[p] = 0.0

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if out_of_bounds_3d(
                        base_coord, grid_velocity.shape[0], np.array([i, j, k])
                    ):
                        raise RuntimeError
                    dpos = np.array((i, j, k)) - fx
                    grid_v = grid_velocity[
                        base_coord[0] + i, base_coord[1] + j, base_coord[2] + k
                    ]
                    weight = w[i][0] * w[j][1] * w[k][2]
                    v[p] += weight * grid_v
                    C[p] += 4 * inv_dx * np.outer(weight * grid_v, dpos)

        # if len(boundary_forces[1]) > 0:
        #     if p in boundary_forces[1]:
        #         v[p] += boundary_forces[0]

        x[p] += dt * v[p]
        F_ = (np.eye(3) + dt * C[p]) @ F[p]

        U, sig, V = np.linalg.svd(F_)
        if model == "snow":
            sig = np.clip(sig, 1.0 - 2.5e-2, 1.0 + 7.5e-3)
        sig = np.eye(3) * sig

        old_J = np.linalg.det(F_)

        if model == "snow":
            F_ = U @ sig @ V.T

        det = np.linalg.det(F_) + 1e-10
        Jp[p] = np.clip(Jp[p] * old_J / det, 0.6, 20.0)
        F[p] = F_

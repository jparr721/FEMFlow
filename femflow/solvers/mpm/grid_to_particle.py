import functools
from typing import List

import numpy as np

from .parameters import MPMParameters
from .particle import NeoHookeanParticle
from .utils import quadric_kernel


def grid_to_particle(
    params: MPMParameters, particles: List[NeoHookeanParticle], grid: np.ndarray
):
    nvec = functools.partial(np.full, params.dimensions)

    for p in particles:
        cell_index = np.floor(p.position * params.grid_resolution - nvec(0.5)).astype(
            np.int32
        )

        # fx
        cell_difference = p.position * params.grid_resolution - cell_index

        # Weight value w_p
        weights = quadric_kernel(cell_difference)

        # We re-compute our momentum and velocity at each step.
        p.velocity = np.zeros(params.dimensions)
        p.affine_momentum = np.zeros((params.dimensions, params.dimensions))

        for i in range(3):
            for j in range(3):
                # (x_i - x_p)^T
                dpos = np.array((i, j)) - cell_difference

                grid_velocity = grid[cell_index[0] + i, cell_index[1] + j]

                weight = weights[i][0] * weights[j][1]

                # Update the velocity term, v_p
                # Eqn 175. v_p = sum(w_i_p * v_i)
                p.velocity = weight * grid_velocity

                # Update the affine motion term via B_p
                # Eqn 176. B_p = sum(w_i_p * v_i * (x_i - x_p)^T
                inv_dx = params.grid_resolution
                p.affine_momentum = 4 * inv_dx * np.outer(weight * grid_velocity, dpos)

        # Advection step
        p.position += params.dt * p.velocity

        # MLS-MPM F-Update
        F = np.matmul(
            np.eye(params.dimensions) + params.dt * p.affine_momentum,
            p.deformation_gradient,
        )

        U, sig, V = np.linalg.svd(F)

        # Snow plasticity
        for i in range(params.dimensions):
            sig[i, i] = np.clip(sig[i, i], 1.0 - 2.5e-2, 1.0 + 7.5e-3)

        oldJ = np.linalg.det(F)
        F = np.matmul(np.matmul(U, sig), V.T)

        J_p_new = np.clip(p.volume * oldJ / np.linalg.det(F), 0.6, 20.0)

        p.volume = J_p_new
        p.deformation_gradient = F


import numba as nb
import numpy as np
from numba import typed

from femflow.solvers.mpm.particle import Particle
from femflow.solvers.mpm.utils import (
    constant_hardening,
    fixed_corotated_stress_3d,
    oob,
    snow_hardening,
)


@nb.njit
def p2g(
    inv_dx: float,
    hardening: float,
    dx: float,
    dt: float,
    volume: float,
    grid_velocity: np.ndarray,
    grid_mass: np.ndarray,
    particles: typed.List[Particle],
    v: np.ndarray,
    F: np.ndarray,
    C: np.ndarray,
    Jp: np.ndarray,
    model: str = "neo_hookean",
):
    """Particle scatter to grid phase via eulerian operations.

    Args:
        inv_dx (float): inv_dx
        hardening (float): hardening
        mu_0 (float): mu_0
        lambda_0 (float): lambda_0
        mass (float): mass
        dx (float): dx
        dt (float): dt
        volume (float): volume
        grid (np.ndarray): grid
        x (np.ndarray): x
        v (np.ndarray): v
        F (np.ndarray): F
        C (np.ndarray): C
        Jp (np.ndarray): Jp
        model (str): model
    """
    for p, particle in enumerate(particles):
        base_coord = (particle.pos * inv_dx - 0.5).astype(np.int64)
        if oob(base_coord, grid_velocity.shape[0]):
            raise RuntimeError
        fx = (particle.pos * inv_dx - base_coord).astype(np.float64)

        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]

        mu, lambda_ = (
            constant_hardening(particle.mu_0, particle.lambda_0, hardening)
            if model == "neo_hookean"
            else snow_hardening(particle.mu_0, particle.lambda_0, hardening, Jp[p])
        )

        affine = fixed_corotated_stress_3d(
            F[p], inv_dx, mu, lambda_, dt, volume, particle.mass, C[p]
        )

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if oob(base_coord, grid_velocity.shape[0], np.array((i, j, k))):
                        raise RuntimeError
                    dpos = (np.array((i, j, k)) - fx) * dx
                    weight = w[i][0] * w[j][1] * w[k][2]

                    grid_velocity[
                        base_coord[0] + i, base_coord[1] + j, base_coord[2] + k
                    ] += weight * (v[p] * particle.mass + affine @ dpos)
                    grid_mass[
                        base_coord[0] + i, base_coord[1] + j, base_coord[2] + k
                    ] += (weight * particle.mass)

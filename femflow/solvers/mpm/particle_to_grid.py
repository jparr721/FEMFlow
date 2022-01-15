import functools
from typing import List

import numpy as np
from loguru import logger

from femflow.numerics.linear_algebra import polar_decomp

from .parameters import MPMParameters
from .particle import NeoHookeanParticle
from .utils import quadric_kernel


def hardening(params: MPMParameters, particle: NeoHookeanParticle):
    """Compute the hardening of the plasticity model as

    F^n+1 = F^n+1_E + F_n+1_P

    Where each component is the elastic and plastic components of the hardening model.

    This is simplified as:
    mu(F_P) = mu_0 * e^epsilon(1 - J_p)
    lambda(F_P) = lambda_0 * e^epsilon(1 - J_p)

    J_p (volume) is provided by the particle, so we just compute the value of e and
    multiply through in this implementation.

    Args:
        params (MPMParameters): The simulation parameters
        particle (NeoHookeanParticle): The simulation particle
    """
    e = np.exp(params.hardening * (1.0 - particle.volume))
    particle.mu = params.mu_0 * e
    particle.lambda_ = params.lambda_0 * e


def fixed_corotated_stress(params: MPMParameters, particle: NeoHookeanParticle):
    """Computes the fixed corotated stress of the particle following snow plasticity.

    It utilizes the following formula:

    P(F) = grad(strain_energy_density) with respect to deformation gradient;
    This translates to 2 * mu * (F - R) + lambda * (J - 1) * J * F^-T

    Where mu and lambda are our neo-hookean material coefficients, J is the
    jacobian, F is the deformation gradient, and R is the rotation component
    from the polar decomposition of the deformation gradient. This gives us out
    co-rotated stress responses in the snow plasticity model.

    Args:
        params (MPMParameters): The simulation parameters
        particle (NeoHookeanParticle): The simulation particle

    Returns:
        np.ndarray: The affine particle-in-a-cell momentum fused with cauchy stress
    """
    current_volume = np.linalg.det(particle.deformation_gradient)

    # F = r, s; Rotation Matrix and Symmetric Matrix
    r, _ = polar_decomp(particle.deformation_gradient)

    # Cauchy stress
    PF = np.matmul(
        2 * particle.mu * (particle.deformation_gradient - r),
        particle.deformation_gradient.T
        + particle.lambda_ * (current_volume - 1) * current_volume,
    )

    # params.dx is 1 / grid res so inv dx is grid res
    inv_dx = params.grid_resolution

    # Inverse density is our constant scaling factor for our APIC momentum/stres
    inverse_density = 4 * inv_dx * inv_dx

    # Scaled cauchy stress
    stress = -(params.dt * params.volume) * (inverse_density * PF)

    # Fused APIC momentum + MLS-MPM stress contribution
    # See http://taichi.graphics/wp-content/uploads/2019/03/mls-mpm-cpic.pdf
    # Eqn 29
    return stress + particle.mass * particle.affine_momentum


def particle_to_grid(
    params: MPMParameters, particles: List[NeoHookeanParticle], grid: np.ndarray
):
    nvec = functools.partial(np.full, params.dimensions)

    # Clear the grid for APIC step.
    grid[:, :, :] = 0

    for p in particles:
        if params.debug:
            logger.debug(f"Before: {p}")
            logger.debug(
                f"Grid Stats Before: {grid.max()} {len(grid.nonzero())} {grid.min()}"
            )
        cell_index = (p.position * params.grid_resolution - nvec(0.5)).astype(np.int64)

        # fx
        cell_difference = p.position * params.grid_resolution - cell_index.astype(
            np.float64
        )

        # Weight value w_p
        weights = quadric_kernel(cell_difference)

        # Compute the hardened coefficients and apply directly to the particle
        hardening(params, p)

        # Compute the model stress
        affine = fixed_corotated_stress(params, p)

        # For all particles, map to grid and compute mass and momentum
        try:
            for i in range(3):
                for j in range(3):
                    # w_i_p The weight for the grid cell of this particle
                    weight = weights[i][0] * weights[j][1]

                    dpos = (np.array((i, j)) - cell_difference) * params.dx
                    vxy = p.velocity * p.mass

                    assert len(vxy) == params.dimensions

                    # Compute the translational momentum of the particle
                    mv = np.array((vxy[0], vxy[1], p.mass))

                    affinexdx = np.dot(affine, dpos)

                    # Compute the density for this particle in relation to the others
                    grid[cell_index[0] + i, cell_index[1] + j] += weight * (
                        mv + np.array((*np.dot(affine, dpos), 0))
                    )
        except Exception as e:
            logger.error(f"Found an error: {e}")
            logger.debug(f"Cell index: {cell_index}")
            logger.debug(f"Particle: {p}")
            exit(1)
        if params.debug:
            logger.debug(f"After: {p}")
            logger.debug(
                f"Grid Stats After: {grid.max()} {len(grid.nonzero())} {grid.min()}"
            )

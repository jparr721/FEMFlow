import os
from typing import List

import numpy as np

from ..grid_to_particle import grid_to_particle
from ..grid_velocity import grid_velocity
from ..parameters import MPMParameters
from ..particle import NeoHookeanParticle, make_particle
from ..particle_to_grid import particle_to_grid

E = 1e4
nu = 0.2

params = MPMParameters(
    mass=1,
    volume=1,
    hardening=10,
    E=E,
    v=nu,
    mu_0=E / (2 * (1 + nu)),
    lambda_0=E * nu / ((1 + nu) * (1 - 2 * nu)),
    gravity=-200.0,
    dt=0.0001,
    dx=1 / 80,
    grid_resolution=80,
    dimensions=2,
)


def test_grid_to_particle():
    def add_object(center: np.ndarray, particles: List[NeoHookeanParticle]):
        particles.append(make_particle(center, np.zeros(2), np.zeros(3)))

    particles = []
    add_object(np.array((0.55, 0.45)), particles)
    add_object(np.array((0.45, 0.65)), particles)
    add_object(np.array((0.55, 0.85)), particles)
    grid = np.zeros((params.grid_resolution + 1, params.grid_resolution + 1, 3))

    particle_to_grid(params, particles, grid)
    grid_velocity(params, grid)
    grid_to_particle(params, particles, grid)

    compare_particles = [
        NeoHookeanParticle(
            position=np.array([0.55, 0.449998]),
            mass=1,
            velocity=np.array([0.0, -0.02]),
            affine_momentum=np.array(
                [[0.00000000e00, 0.00000000e00], [-2.98023e-08, 0.00000000e00]]
            ),
            deformation_gradient=np.array(
                [[1.00000000e00, 0], [-2.98023e-12, 1.00000000e00]]
            ),
            volume=1.0,
            color=np.array([0.0, 0.0, 0.0]),
            lambda_=2777.777777777778,
            mu=4166.666666666667,
        ),
        NeoHookeanParticle(
            position=np.array([0.45, 0.649998]),
            mass=1,
            velocity=np.array([0.0, -0.02]),
            affine_momentum=np.array(
                [[0.00000000e00, 0.00000000e00], [-2.98023e-08, 0.00000000e00]]
            ),
            deformation_gradient=np.array(
                [[1.00000000e00, 0], [-2.98023e-12, 1.00000000e00]]
            ),
            volume=1.0,
            color=np.array([0.0, 0.0, 0.0]),
            lambda_=2777.777777777778,
            mu=4166.666666666667,
        ),
        NeoHookeanParticle(
            position=np.array([0.55, 0.849998]),
            mass=1,
            velocity=np.array([0.0, -0.02]),
            affine_momentum=np.array(
                [[0.00000000e00, 0.00000000e00], [-2.98023e-08, 0.00000000e00]]
            ),
            deformation_gradient=np.array(
                [[1.00000000e00, 0], [-2.98023e-12, 1.00000000e00]]
            ),
            volume=1.0,
            color=np.array([0.0, 0.0, 0.0]),
            lambda_=2777.777777777778,
            mu=4166.666666666667,
        ),
    ]

    for real_particle, compare_particle in zip(particles, compare_particles):
        assert np.isclose(real_particle.position, compare_particle.position).all()
        assert np.isclose(real_particle.velocity, compare_particle.velocity).all()
        assert np.isclose(
            real_particle.affine_momentum, compare_particle.affine_momentum, atol=0.000001
        ).all()
        assert np.isclose(
            real_particle.deformation_gradient, compare_particle.deformation_gradient
        ).all()
        assert np.isclose(real_particle.volume, compare_particle.volume)
        assert np.isclose(real_particle.volume, compare_particle.volume)

import os
from typing import List

import numpy as np

from ..grid_to_particle import grid_to_particle
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
        for _ in range(1000):
            pos = (np.random.rand(2) * 2.0 - np.ones(2)) * 0.08 + center
            particles.append(make_particle(pos, np.zeros(2), np.zeros(3)))

    particles = []
    add_object(np.array((0.55, 0.45)), particles)
    add_object(np.array((0.45, 0.65)), particles)
    add_object(np.array((0.55, 0.85)), particles)
    grid = np.zeros((params.grid_resolution + 1, params.grid_resolution + 1, 3))

    particle_to_grid(params, particles, grid)
    grid_to_particle(params, particles, grid)

    assert False

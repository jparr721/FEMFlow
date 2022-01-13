import os
from collections import Counter
from typing import List

import numpy as np

from ..parameters import MPMParameters
from ..particle import NeoHookeanParticle, make_particle
from ..particle_to_grid import *

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


def test_hardening():
    particle = make_particle(np.random.rand(2), np.zeros(2), np.zeros(3))
    hardening(params, particle)

    c_lambda = 2777.777777777778
    c_mu = 4166.666666666667

    assert np.isclose(c_lambda, particle.lambda_)
    assert np.isclose(c_mu, particle.mu)


def test_fixed_corotated_stress():
    particle = make_particle(np.random.rand(2), np.zeros(2), np.zeros(3))
    hardening(params, particle)

    particle.deformation_gradient = np.array([[1, 0], [2.51457e-12, 1]])
    particle.affine_momentum = np.array([[0, 0], [2.51457e-08, -5.96046e-08]])

    c_affine = np.array([[0, -2.68221e-08], [-1.67638e-09, -5.96046e-08]])
    affine = fixed_corotated_stress(params, particle)
    assert np.isclose(affine, c_affine).all()


def test_particle_to_grid():
    def add_object(center: np.ndarray, particles: List[NeoHookeanParticle]):
        particles.append(make_particle(center, np.zeros(2), np.zeros(3)))

    particles = []
    add_object(np.array((0.55, 0.45)), particles)
    add_object(np.array((0.45, 0.65)), particles)
    add_object(np.array((0.55, 0.85)), particles)

    assert len(particles) == 3

    this_dir = os.path.dirname(os.path.abspath(__file__))
    ground_truth_file_path = os.path.join(this_dir, "ground_truth_grid.txt")
    ground_truth_grid = np.loadtxt(ground_truth_file_path)

    grid = np.zeros((params.grid_resolution + 1, params.grid_resolution + 1, 3))

    particle_to_grid(params, particles, grid)

    r, c, _ = grid.nonzero()

    # Counters are kinda not ideal here, but the grid nonzeros return in a different
    # order than the ground truth is listed, so restructuring would be annoying, this
    # was independently verified to be correct though.
    c = Counter(grid[r, c].reshape(-1))
    cc = Counter(ground_truth_grid.reshape(-1))

    assert c == cc

import os

import numpy as np

from ..parameters import MPMParameters
from ..particle import make_particle
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


def read_outfile() -> np.ndarray:
    values = []

    with open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "out.txt"), "r"
    ) as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line.startswith("value:"):
                ln = lines[i + 1]
                group_start = ln.index("(") + 1
                group_end = ln.index(")")
                ln = ln[group_start:group_end]

                x, y, z = ln.split()
                values.append(np.array((x, y, z), dtype=np.float32))

    return np.array(values)


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
    values = read_outfile()

    p1pos = np.array((0.607899, 0.387073))
    p1 = make_particle(p1pos, np.zeros(2), np.zeros(3))

    p2pos = np.array((0.524773, 0.894981))
    p2 = make_particle(p2pos, np.zeros(2), np.zeros(3))

    particles = [p1, p2]

    grid = np.zeros((params.grid_resolution + 1, params.grid_resolution + 1, 3))

    particle_to_grid(params, particles, grid)

    rows, cols, _ = grid.nonzero()
    print(grid[rows, cols])

    assert np.isclose(grid[rows, cols], values).all()

from typing import List

import numba as nb
import numpy as np
from numba import float64

from femflow.numerics.linear_algebra import matrix_to_vector


@nb.experimental.jitclass(
    [
        ("pos", float64[::1]),
        ("mass", float64),
        ("force", float64),
        ("lambda_", float64),
        ("mu", float64),
    ]
)
class Particle(object):
    def __init__(
        self, pos: np.ndarray, mass: float, force: float, lambda_: float, mu: float,
    ):
        self.pos = pos
        self.mass = mass
        self.force = force
        self.lambda_ = lambda_
        self.mu = mu


def map_particles_to_pos(particles: List[Particle], coeff: float):
    return matrix_to_vector(
        np.array(list(map(lambda p: p.pos.copy() / coeff, particles)))
    )


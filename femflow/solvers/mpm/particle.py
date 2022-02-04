from typing import List

import numba as nb
import numpy as np
from numba import float64

from femflow.numerics.linear_algebra import matrix_to_vector


@nb.experimental.jitclass(
    [
        ("pos", float64[::1]),
        ("force", float64),
        ("mass", float64),
        ("lambda_0", float64),
        ("mu_0", float64),
    ]
)
class Particle(object):
    def __init__(
        self, pos: np.ndarray, force: float, mass: float, lambda_: float, mu: float,
    ):
        self.pos = pos
        self.force = force
        self.mass = mass
        self.lambda_0 = lambda_
        self.mu_0 = mu


def map_particles_to_pos(particles: List[Particle], coeff: float):
    return matrix_to_vector(
        np.array(list(map(lambda p: p.pos.copy() / coeff, particles)))
    )


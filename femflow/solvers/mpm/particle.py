import numpy as np


class Particle(object):
    def __init__(self, x: np.ndarray, c: int):
        dim = x.shape[0]
        assert dim == 2 or dim == 3, f"Shape {x.shape} is invalid"

        self.x = x
        self.c = c
        self.v = np.zeros(dim, dtype=float)
        self.F = np.eye(dim)
        self.C = np.zeros((dim, dim))
        self.Jp = 1

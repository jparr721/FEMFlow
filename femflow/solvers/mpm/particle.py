import numpy as np


class Particle(object):
    def __init__(self, x: np.ndarray, c: int, v=np.zeros(2, dtype=np.float64)):
        self.x = x
        self.c = c
        self.v = v
        self.F = np.eye(2)
        self.C = np.zeros(2)
        self.Jp = 1

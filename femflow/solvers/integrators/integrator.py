import numpy as np


class Integrator(object):
    def __init__(self):
        pass

    def integrate(self, forces: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

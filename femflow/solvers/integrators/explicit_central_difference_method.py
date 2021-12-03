import numpy as np

from .integrator import Integrator


class ExplicitCentralDifferenceMethod(Integrator):
    def __init__(self):
        super(ExplicitCentralDifferenceMethod, self).__init__()

    def integrate(forces: np.ndarray) -> np.ndarray:
        return np.array([])

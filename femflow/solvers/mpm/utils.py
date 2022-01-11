import functools

import numpy as np


def quadric_kernel(fx: np.ndarray) -> np.ndarray:
    nvec = functools.partial(np.full, fx.size)
    weights = np.zeros(3)
    weights[0] = nvec(0.5) * pow(nvec(1.5) - fx, 2)
    weights[1] = nvec(0.75) - pow(fx - nvec(1), 2)
    weights[2] = nvec(0.5) * pow(fx - nvec(0.5), 2)
    return weights

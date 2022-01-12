import numpy as np

from ..utils import *


def test_quadric_kernel():
    fx = np.array((0.631893, 0.965839))

    weights_compare = np.array(
        [[0.37680488, 0.14266399], [0.61449724, 0.74883303], [0.00869788, 0.10850299]]
    )

    weights = quadric_kernel(fx)

    assert np.isclose(weights, weights_compare).all()

import numpy as np

from ..linear_tetrahedron import linear_tetrahedron


def test_linear_tetrahedron():
    v = np.array([[0, 0, 0], [0, 1, 0], [0, 1, 1], [1, 1, 1]])
    element = np.array([0, 1, 2, 3])
    x = np.array([0.5, 0.5, 0.5])
    compare = np.array([0.5, 1.11022e-16, -3.88578e-16, 0.5])

    phi = linear_tetrahedron(v, element, x)
    assert np.isclose(compare, phi).all()

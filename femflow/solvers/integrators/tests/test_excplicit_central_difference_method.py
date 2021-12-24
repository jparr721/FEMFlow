import numpy as np
from scipy.sparse import csr_matrix

from ..explicit_central_difference_method import ExplicitCentralDifferenceMethod


def test_constructor():
    stiffness_matrix = csr_matrix(np.array([[6, -2], [-2, 4]]))
    initial_displacement = np.zeros(2)
    initial_forces = np.array([0, 10])

    e = ExplicitCentralDifferenceMethod(0.28, 1, stiffness_matrix, initial_displacement, initial_forces)

    assert np.array_equal(e.acceleration, np.array([0, 10]))


def test_solver():
    displacement = np.zeros(2)
    forces = np.array([0, 10])
    stiffness_matrix = csr_matrix(np.array([[6, -2], [-2, 4]]))
    mass_matrix = csr_matrix(np.array([[2, 0], [0, 1]]))
    e = ExplicitCentralDifferenceMethod(
        0.28, mass_matrix, stiffness_matrix, displacement, forces, rayleigh_lambda=0, rayleigh_mu=0
    )

    for i in range(12):
        displacement = e.integrate(forces, displacement)
        if i == 0:
            assert np.all(np.isclose(np.array([0, 0.392]), displacement))

    assert np.all(np.isclose(np.array([1.0223, 2.60083]), displacement))

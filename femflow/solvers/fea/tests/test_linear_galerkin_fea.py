import numpy as np
from femflow.solvers.material import hookes_law_isotropic_constitutive_matrix

from ..linear_galerkin_nondynamic import LinearGalerkinNonDynamic

E = 210e6
nu = 0.3


def make_mesh():
    v = np.array(
        [
            [0, 0, 0],
            [0.025, 0, 0],
            [0, 0.5, 0],
            [0.025, 0.5, 0],
            [0, 0, 0.25],
            [0.025, 0, 0.25],
            [0, 0.5, 0.25],
            [0.025, 0.5, 0.25],
        ]
    )

    t = np.array([[0, 1, 3, 5], [0, 3, 2, 6], [5, 4, 6, 0], [5, 6, 7, 3], [0, 5, 3, 6]])

    return v, t


def test_static_form_solve():
    v, t = make_mesh()
    v = v.reshape(-1)
    t = t.reshape(-1)

    boundary_conditions = dict()
    boundary_conditions[2] = np.array([0, 3.125, 0])
    boundary_conditions[3] = np.array([0, 6.25, 0])
    boundary_conditions[6] = np.array([0, 6.25, 0])
    boundary_conditions[7] = np.array([0, 3.125, 0])
    D = hookes_law_isotropic_constitutive_matrix(np.array([E, nu]))

    solver = LinearGalerkinNonDynamic(boundary_conditions, D, v, t)
    solver.solve_static()
    # NOTE: The division by 1e-5 is only for making the comparison here easier.
    U_compare = np.array(
        [
            0,
            0,
            0,
            0,
            0,
            0,
            -0.0004,
            0.6082,
            0.0090,
            -0.0127,
            0.6078,
            0.0056,
            0,
            0,
            0,
            0,
            0,
            0,
            0.0127,
            0.6078,
            -0.0056,
            0.0004,
            0.6082,
            -0.0090,
        ]
    ).reshape((24, 1))

    assert np.all(np.isclose(solver.U.toarray() / 1e-5, U_compare, atol=0.0001))

import numpy as np

from ..linear_galerkin_fea import *

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
    boundary_conditions = {
        2: np.array([0, 3.125, 0]),
        3: np.array([0, 6.25, 0]),
        6: np.array([0, 6.25, 0]),
        7: np.array([0, 3.125, 0]),
    }
    D = isotropic_constitutive_matrix(E, nu)

    element_stiffnesses = []
    for row in t:
        B = assemble_shape_fn_matrix(*v[row])
        element_stiffnesses.append(assemble_element_stiffness_matrix(row, v, B, D))

    K = assemble_global_stiffness_matrix(element_stiffnesses, 3 * len(v))
    K_e, F_e = assemble_boundary_forces(K, boundary_conditions)

    # NOTE: The division by 1e-5 is only for making the comparison here easier.
    U = compute_U(K, K_e, F_e, boundary_conditions) / 1e-5

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
    )

    assert np.all(np.isclose(U, U_compare, atol=0.0001))

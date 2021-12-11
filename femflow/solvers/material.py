import numpy as np


def hookes_law_isotropic_constitutive_matrix(coefficients: np.ndarray) -> np.ndarray:
    assert len(coefficients) == 2, "Too many coefficients! AHH"
    E, v = coefficients
    return (E / ((1 + v) * (1 - 2 * v))) * np.array(
        [
            [1 - v, v, v, 0, 0, 0],
            [v, 1 - v, v, 0, 0, 0],
            [v, v, 1 - v, 0, 0, 0],
            [0, 0, 0, (1 - 2 * v) / 2, 0, 0],
            [0, 0, 0, 0, (1 - 2 * v) / 2, 0],
            [0, 0, 0, 0, 0, (1 - 2 * v) / 2],
        ]
    )


def hookes_law_orthotropic_constitutive_matrix(coefficients: np.ndarray) -> np.ndarray:
    assert len(coefficients) == 12, "Too many coefficients! AHH"
    E_x, E_y, E_z, G_yz, G_zx, G_xy, v_yx, v_zx, v_zy, v_xy, v_xz, v_yz = coefficients
    delta = (1 - v_xy * v_yx - v_yz * v_zy - v_zx * v_xz - 2 * (v_xy * v_yz * v_zx)) / (np.prod(E_x, E_y, E_z))

    constitutive_matrix = np.zeros((6, 6))
    constitutive_matrix[0, 0] = (1 - v_yz * v_zy) / (E_y * E_z * delta)
    constitutive_matrix[0, 1] = (v_yx + v_zx * v_yz) / (E_y * E_z * delta)
    constitutive_matrix[0, 2] = (v_zx + v_yx * v_zy) / (E_y * E_z * delta)

    constitutive_matrix[1, 0] = (v_xy + v_xz * v_zy) / (E_z * E_x * delta)
    constitutive_matrix[1, 1] = (1 - v_zx * v_xz) / (E_z * E_x * delta)
    constitutive_matrix[1, 2] = (v_zy + v_zx * v_xy) / (E_z * E_x * delta)

    constitutive_matrix[2, 0] = (v_xz + v_xy * v_yz) / (E_x * E_y * delta)
    constitutive_matrix[2, 1] = (v_yz + v_xz * v_yx) / (E_x * E_y * delta)
    constitutive_matrix[2, 2] = (1 - v_xy * v_yx) / (E_x * E_y * delta)

    # NOTE: These parameters differ from the isotropic form, more details here if this causes problems:
    # https://www.efunda.com/formulae/solid_mechanics/mat_mechanics/hooke_orthotropic.cfm
    constitutive_matrix[3, 3] = 2 * G_yz
    constitutive_matrix[4, 4] = 2 * G_zx
    constitutive_matrix[5, 5] = 2 * G_xy
    return constitutive_matrix

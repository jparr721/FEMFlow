import copy
from collections import namedtuple
from typing import Dict, List, Tuple

import numpy as np
from scipy.sparse import csr_matrix

ElementStiffness = namedtuple("ElementStiffness", ["stiffness", "element"])


def index_slice(X: np.ndarray, R: np.ndarray, C: np.ndarray = None) -> np.ndarray:
    if C is None:
        C = copy.deepcopy(R)
    assert R.ndim == 1 and C.ndim == 1, "Rows and cols must be vectors"
    rows = R.size
    cols = C.size
    out = np.zeros((rows, cols))
    for row in range(rows):
        for col in range(cols):
            out[row, col] = X[R[row], C[col]]
    return out


def isotropic_constitutive_matrix(E: float, v: float) -> np.ndarray:
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


def orthotropic_constitutive_matrix(E: np.ndarray, v: np.ndarray, G: np.ndarray):
    pass


def tet_volume(a, b, c, d):
    x1, y1, z1 = a
    x2, y2, z2 = b
    x3, y3, z3 = c
    x4, y4, z4 = d

    x = np.array([[1, x1, y1, z1], [1, x2, y2, z2], [1, x3, y3, z3], [1, x4, y4, z4]])

    return np.linalg.det(x) / 6


def assemble_shape_fn_matrix(a, b, c, d):
    V = tet_volume(a, b, c, d)

    def _beta(beta: float, gamma: float, delta: float) -> np.ndarray:
        return np.array(
            [[beta, 0, 0], [0, gamma, 0], [0, 0, delta], [gamma, beta, 0], [0, delta, gamma], [delta, 0, beta]]
        )

    def _shape_fn(p0, p1, p2, p3, p4, p5) -> float:
        return np.linalg.det(np.array([[1, p0, p1], [1, p2, p3], [1, p4, p5]]))

    x1, y1, z1 = a
    x2, y2, z2 = b
    x3, y3, z3 = c
    x4, y4, z4 = d

    beta_1 = -1 * _shape_fn(y2, z2, y3, z3, y4, z4)
    beta_2 = _shape_fn(y1, z1, y3, z3, y4, z4)
    beta_3 = -1 * _shape_fn(y1, z1, y2, z2, y4, z4)
    beta_4 = _shape_fn(y1, z1, y2, z2, y3, z3)

    gamma_1 = _shape_fn(x2, z2, x3, z3, x4, z4)
    gamma_2 = -1 * _shape_fn(x1, z1, x3, z3, x4, z4)
    gamma_3 = _shape_fn(x1, z1, x2, z2, x4, z4)
    gamma_4 = -1 * _shape_fn(x1, z1, x2, z2, x3, z3)

    delta_1 = -1 * _shape_fn(x2, y2, x3, y3, x4, y4)
    delta_2 = _shape_fn(x1, y1, x3, y3, x4, y4)
    delta_3 = -1 * _shape_fn(x1, y1, x2, y2, x4, y4)
    delta_4 = _shape_fn(x1, y1, x2, y2, x3, y3)

    b1 = _beta(beta_1, gamma_1, delta_1)
    b2 = _beta(beta_2, gamma_2, delta_2)
    b3 = _beta(beta_3, gamma_3, delta_3)
    b4 = _beta(beta_4, gamma_4, delta_4)

    # Matrix of partial derivatives making up the shape function matrix
    shape_fn_matrix = np.hstack((b1, b2, b3, b4))
    if V > 0:
        shape_fn_matrix /= V * 6
    else:
        shape_fn_matrix /= 6
    return shape_fn_matrix.astype(np.float32)


def assemble_element_stiffness_matrix(
    tet_indices: np.ndarray, tets: np.ndarray, B: np.ndarray, D: np.ndarray
) -> ElementStiffness:
    i0, i1, i2, i3 = tet_indices
    p1, p2, p3, p4 = tets[i0], tets[i1], tets[i2], tets[i3]
    V = tet_volume(p1, p2, p3, p4)
    e_stiffness = np.matmul(np.matmul(V * B.T, D), B)
    return ElementStiffness(e_stiffness, tet_indices)


def assemble_global_stiffness_matrix(elements: List[ElementStiffness], rows: int) -> csr_matrix:
    triplets = []
    for element in elements:
        k, tetrahedral = element
        i, j, m, n = tetrahedral

        triplets.append((3 * i, 3 * i, k[0, 0]))
        triplets.append((3 * i, 3 * i + 1, k[0, 1]))
        triplets.append((3 * i, 3 * i + 2, k[0, 2]))
        triplets.append((3 * i, 3 * j, k[0, 3]))
        triplets.append((3 * i, 3 * j + 1, k[0, 4]))
        triplets.append((3 * i, 3 * j + 2, k[0, 5]))
        triplets.append((3 * i, 3 * m, k[0, 6]))
        triplets.append((3 * i, 3 * m + 1, k[0, 7]))
        triplets.append((3 * i, 3 * m + 2, k[0, 8]))
        triplets.append((3 * i, 3 * n, k[0, 9]))
        triplets.append((3 * i, 3 * n + 1, k[0, 10]))
        triplets.append((3 * i, 3 * n + 2, k[0, 11]))

        triplets.append((3 * i + 1, 3 * i, k[1, 0]))
        triplets.append((3 * i + 1, 3 * i + 1, k[1, 1]))
        triplets.append((3 * i + 1, 3 * i + 2, k[1, 2]))
        triplets.append((3 * i + 1, 3 * j, k[1, 3]))
        triplets.append((3 * i + 1, 3 * j + 1, k[1, 4]))
        triplets.append((3 * i + 1, 3 * j + 2, k[1, 5]))
        triplets.append((3 * i + 1, 3 * m, k[1, 6]))
        triplets.append((3 * i + 1, 3 * m + 1, k[1, 7]))
        triplets.append((3 * i + 1, 3 * m + 2, k[1, 8]))
        triplets.append((3 * i + 1, 3 * n, k[1, 9]))
        triplets.append((3 * i + 1, 3 * n + 1, k[1, 10]))
        triplets.append((3 * i + 1, 3 * n + 2, k[1, 11]))

        triplets.append((3 * i + 2, 3 * i, k[2, 0]))
        triplets.append((3 * i + 2, 3 * i + 1, k[2, 1]))
        triplets.append((3 * i + 2, 3 * i + 2, k[2, 2]))
        triplets.append((3 * i + 2, 3 * j, k[2, 3]))
        triplets.append((3 * i + 2, 3 * j + 1, k[2, 4]))
        triplets.append((3 * i + 2, 3 * j + 2, k[2, 5]))
        triplets.append((3 * i + 2, 3 * m, k[2, 6]))
        triplets.append((3 * i + 2, 3 * m + 1, k[2, 7]))
        triplets.append((3 * i + 2, 3 * m + 2, k[2, 8]))
        triplets.append((3 * i + 2, 3 * n, k[2, 9]))
        triplets.append((3 * i + 2, 3 * n + 1, k[2, 10]))
        triplets.append((3 * i + 2, 3 * n + 2, k[2, 11]))

        # j
        triplets.append((3 * j, 3 * i, k[3, 0]))
        triplets.append((3 * j, 3 * i + 1, k[3, 1]))
        triplets.append((3 * j, 3 * i + 2, k[3, 2]))
        triplets.append((3 * j, 3 * j, k[3, 3]))
        triplets.append((3 * j, 3 * j + 1, k[3, 4]))
        triplets.append((3 * j, 3 * j + 2, k[3, 5]))
        triplets.append((3 * j, 3 * m, k[3, 6]))
        triplets.append((3 * j, 3 * m + 1, k[3, 7]))
        triplets.append((3 * j, 3 * m + 2, k[3, 8]))
        triplets.append((3 * j, 3 * n, k[3, 9]))
        triplets.append((3 * j, 3 * n + 1, k[3, 10]))
        triplets.append((3 * j, 3 * n + 2, k[3, 11]))

        triplets.append((3 * j + 1, 3 * i, k[4, 0]))
        triplets.append((3 * j + 1, 3 * i + 1, k[4, 1]))
        triplets.append((3 * j + 1, 3 * i + 2, k[4, 2]))
        triplets.append((3 * j + 1, 3 * j, k[4, 3]))
        triplets.append((3 * j + 1, 3 * j + 1, k[4, 4]))
        triplets.append((3 * j + 1, 3 * j + 2, k[4, 5]))
        triplets.append((3 * j + 1, 3 * m, k[4, 6]))
        triplets.append((3 * j + 1, 3 * m + 1, k[4, 7]))
        triplets.append((3 * j + 1, 3 * m + 2, k[4, 8]))
        triplets.append((3 * j + 1, 3 * n, k[4, 9]))
        triplets.append((3 * j + 1, 3 * n + 1, k[4, 10]))
        triplets.append((3 * j + 1, 3 * n + 2, k[4, 11]))

        triplets.append((3 * j + 2, 3 * i, k[5, 0]))
        triplets.append((3 * j + 2, 3 * i + 1, k[5, 1]))
        triplets.append((3 * j + 2, 3 * i + 2, k[5, 2]))
        triplets.append((3 * j + 2, 3 * j, k[5, 3]))
        triplets.append((3 * j + 2, 3 * j + 1, k[5, 4]))
        triplets.append((3 * j + 2, 3 * j + 2, k[5, 5]))
        triplets.append((3 * j + 2, 3 * m, k[5, 6]))
        triplets.append((3 * j + 2, 3 * m + 1, k[5, 7]))
        triplets.append((3 * j + 2, 3 * m + 2, k[5, 8]))
        triplets.append((3 * j + 2, 3 * n, k[5, 9]))
        triplets.append((3 * j + 2, 3 * n + 1, k[5, 10]))
        triplets.append((3 * j + 2, 3 * n + 2, k[5, 11]))

        # m
        triplets.append((3 * m, 3 * i, k[6, 0]))
        triplets.append((3 * m, 3 * i + 1, k[6, 1]))
        triplets.append((3 * m, 3 * i + 2, k[6, 2]))
        triplets.append((3 * m, 3 * j, k[6, 3]))
        triplets.append((3 * m, 3 * j + 1, k[6, 4]))
        triplets.append((3 * m, 3 * j + 2, k[6, 5]))
        triplets.append((3 * m, 3 * m, k[6, 6]))
        triplets.append((3 * m, 3 * m + 1, k[6, 7]))
        triplets.append((3 * m, 3 * m + 2, k[6, 8]))
        triplets.append((3 * m, 3 * n, k[6, 9]))
        triplets.append((3 * m, 3 * n + 1, k[6, 10]))
        triplets.append((3 * m, 3 * n + 2, k[6, 11]))

        triplets.append((3 * m + 1, 3 * i, k[7, 0]))
        triplets.append((3 * m + 1, 3 * i + 1, k[7, 1]))
        triplets.append((3 * m + 1, 3 * i + 2, k[7, 2]))
        triplets.append((3 * m + 1, 3 * j, k[7, 3]))
        triplets.append((3 * m + 1, 3 * j + 1, k[7, 4]))
        triplets.append((3 * m + 1, 3 * j + 2, k[7, 5]))
        triplets.append((3 * m + 1, 3 * m, k[7, 6]))
        triplets.append((3 * m + 1, 3 * m + 1, k[7, 7]))
        triplets.append((3 * m + 1, 3 * m + 2, k[7, 8]))
        triplets.append((3 * m + 1, 3 * n, k[7, 9]))
        triplets.append((3 * m + 1, 3 * n + 1, k[7, 10]))
        triplets.append((3 * m + 1, 3 * n + 2, k[7, 11]))

        triplets.append((3 * m + 2, 3 * i, k[8, 0]))
        triplets.append((3 * m + 2, 3 * i + 1, k[8, 1]))
        triplets.append((3 * m + 2, 3 * i + 2, k[8, 2]))
        triplets.append((3 * m + 2, 3 * j, k[8, 3]))
        triplets.append((3 * m + 2, 3 * j + 1, k[8, 4]))
        triplets.append((3 * m + 2, 3 * j + 2, k[8, 5]))
        triplets.append((3 * m + 2, 3 * m, k[8, 6]))
        triplets.append((3 * m + 2, 3 * m + 1, k[8, 7]))
        triplets.append((3 * m + 2, 3 * m + 2, k[8, 8]))
        triplets.append((3 * m + 2, 3 * n, k[8, 9]))
        triplets.append((3 * m + 2, 3 * n + 1, k[8, 10]))
        triplets.append((3 * m + 2, 3 * n + 2, k[8, 11]))

        # n
        triplets.append((3 * n, 3 * i, k[9, 0]))
        triplets.append((3 * n, 3 * i + 1, k[9, 1]))
        triplets.append((3 * n, 3 * i + 2, k[9, 2]))
        triplets.append((3 * n, 3 * j, k[9, 3]))
        triplets.append((3 * n, 3 * j + 1, k[9, 4]))
        triplets.append((3 * n, 3 * j + 2, k[9, 5]))
        triplets.append((3 * n, 3 * m, k[9, 6]))
        triplets.append((3 * n, 3 * m + 1, k[9, 7]))
        triplets.append((3 * n, 3 * m + 2, k[9, 8]))
        triplets.append((3 * n, 3 * n, k[9, 9]))
        triplets.append((3 * n, 3 * n + 1, k[9, 10]))
        triplets.append((3 * n, 3 * n + 2, k[9, 11]))

        triplets.append((3 * n + 1, 3 * i, k[10, 0]))
        triplets.append((3 * n + 1, 3 * i + 1, k[10, 1]))
        triplets.append((3 * n + 1, 3 * i + 2, k[10, 2]))
        triplets.append((3 * n + 1, 3 * j, k[10, 3]))
        triplets.append((3 * n + 1, 3 * j + 1, k[10, 4]))
        triplets.append((3 * n + 1, 3 * j + 2, k[10, 5]))
        triplets.append((3 * n + 1, 3 * m, k[10, 6]))
        triplets.append((3 * n + 1, 3 * m + 1, k[10, 7]))
        triplets.append((3 * n + 1, 3 * m + 2, k[10, 8]))
        triplets.append((3 * n + 1, 3 * n, k[10, 9]))
        triplets.append((3 * n + 1, 3 * n + 1, k[10, 10]))
        triplets.append((3 * n + 1, 3 * n + 2, k[10, 11]))

        triplets.append((3 * n + 2, 3 * i, k[11, 0]))
        triplets.append((3 * n + 2, 3 * i + 1, k[11, 1]))
        triplets.append((3 * n + 2, 3 * i + 2, k[11, 2]))
        triplets.append((3 * n + 2, 3 * j, k[11, 3]))
        triplets.append((3 * n + 2, 3 * j + 1, k[11, 4]))
        triplets.append((3 * n + 2, 3 * j + 2, k[11, 5]))
        triplets.append((3 * n + 2, 3 * m, k[11, 6]))
        triplets.append((3 * n + 2, 3 * m + 1, k[11, 7]))
        triplets.append((3 * n + 2, 3 * m + 2, k[11, 8]))
        triplets.append((3 * n + 2, 3 * n, k[11, 9]))
        triplets.append((3 * n + 2, 3 * n + 1, k[11, 10]))
        triplets.append((3 * n + 2, 3 * n + 2, k[11, 11]))

    i, j, v = zip(*triplets)
    del triplets

    return csr_matrix((v, (i, j)), shape=(rows, rows)).astype(np.float32)


def assemble_boundary_forces(K: csr_matrix, boundary_conditions: Dict[int, np.ndarray]) -> np.ndarray:
    F_e = np.zeros(len(boundary_conditions) * 3)
    I_e = np.zeros(len(boundary_conditions) * 3)

    i = 0
    for node, force in boundary_conditions.items():
        n = node * 3
        F_e[i : i + 3] = force
        I_e[i : i + 3] = np.arange(n, n + 3)
        i += 3

    # K_e
    return index_slice(K, I_e), F_e


def compute_U(
    K: csr_matrix, K_e: np.ndarray, F_e: np.ndarray, boundary_conditions: Dict[int, np.ndarray]
) -> np.ndarray:
    U_e = np.linalg.solve(K_e, F_e)
    U = np.zeros(K.shape[0])
    i = 0
    for node in boundary_conditions.keys():
        segment = node * 3
        U[segment : segment + 3] = U_e[i : i + 3]
        i += 3

    return U

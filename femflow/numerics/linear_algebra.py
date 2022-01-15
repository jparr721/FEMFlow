from collections import namedtuple
from typing import Any, List, Tuple, Union

import numpy as np
from scipy.linalg import expm
from scipy.sparse.csr import csr_matrix
from scipy.spatial.transform import Rotation as R


def is_square_matrix(mat: np.ndarray):
    return len(mat.shape) == 2 and mat.shape[0] == mat.shape[1]


def normalized(a: np.ndarray):
    return a / np.linalg.norm(a)


def angle_between_vectors(a: np.ndarray, b: np.ndarray) -> float:
    return np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return np.linalg.norm(a - b)


def midpoint(a: np.ndarray, b: np.ndarray) -> float:
    return np.median((a, b), axis=1)


def angle_axis(angle: float, axis: np.ndarray):
    return expm(np.cross(np.eye(3), normalized(axis) * angle))


def rotation_as_quat(rot: R) -> np.ndarray:
    """Scipy rotations are not in the proper quaternion order

    Args:
        rot (R): The scipy rotation in the form: x, y, z, w

    Returns:
        np.ndarray: The w, x, y, z ordered quaternion
    """
    x, y, z, w = rot.as_quat()
    return np.array([w, x, y, z])


def sparse(
    i: Union[np.ndarray, List[int]],
    j: Union[np.ndarray, List[int]],
    v: Union[np.ndarray, List[Any]],
    m: int,
    n: int,
) -> csr_matrix:
    """Computes a sparse matrix from an input set of indices in 2D and their values as a bijection followed by the shape

    Args:
        i (Union[np.ndarray, List[int]]): Input index x
        j (Union[np.ndarray, List[int]]): Input indices y
        v (Union[np.ndarray, List[Any]]): Input indices y
        m (int): Shape m
        n (int): Shape n

    Returns:
        csr_matrix: The csr matrix output
    """
    return csr_matrix((v, (i, j)), shape=(m, n))


def fast_diagonal_inverse(mat: csr_matrix):
    """Computes the fast diagonal inverse of a diagonal matrix.

    Args:
        mat (np.ndarray): The input matrix which is dense or sparse
    """
    for i, _ in enumerate(mat):
        mat[i, i] = 1 / mat[i, i]


def polar_decomp(m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = m[0, 0] + m[1, 1]
    y = m[1, 0] - m[0, 1]
    scale = 1.0 / np.sqrt(x * x + y * y)
    c = x * scale
    s = y * scale
    r = np.array([[c, -s], [s, c]], dtype=np.float64)
    s = np.matmul(r.T, m)
    return r, s


SVD = namedtuple("SVD", ["u", "sigma", "v"])


def svd_3d(m: np.ndarray) -> SVD:
    pass


def _svd_2d(m: np.ndarray) -> SVD:
    """Stable SVD for 2d matrices

    Based on http://math.ucla.edu/~cffjiang/research/svd/svd.pdf
    Algorithm 4 and libtaichi

    Args:
        m (np.ndarray): m The 2d matrix

    Returns:
        SVD: The svd result
    """
    R, S = polar_decomp(m)
    c = 0.0
    s = 0.0
    s1 = 0.0
    s2 = 0.0

    if abs(S[0, 1]) < 1e-5:
        c, s = 1, 0
        s1, s2 = S[0, 0], S[1, 1]
    else:
        tao = 0.5 * (S[0, 0] - S[1, 1])
        w = np.sqrt(tao ** 2 + S[0, 1] ** 2)
        t = 0.0
        if tao > 0:
            t = S[0, 1] / (tao + w)
        else:
            t = S[0, 1] / (tao - w)
        c = 1 / np.sqrt(t ** 2 + 1)
        s = -t * c
        s1 = c ** 2 * S[0, 0] - 2 * c * s * S[0, 1] + s ** 2 * S[1, 1]
        s2 = s ** 2 * S[0, 0] + 2 * c * s * S[0, 1] + c ** 2 * S[1, 1]
    V = np.zeros((2, 2))
    if s1 < s2:
        tmp = s1
        s1 = s2
        s2 = tmp
        V = np.array([[-s, c], [-c, -s]], dtype=np.float64)
    else:
        V = np.array([[c, s], [-s, c]], dtype=np.float64)

    U = np.matmul(R, V)
    sig = np.array([[s1, 0.0], [0.0, s2]], dtype=np.float64)

    return SVD(U, sig, V)


def svd(m: np.ndarray) -> SVD:
    return _svd_2d(m)

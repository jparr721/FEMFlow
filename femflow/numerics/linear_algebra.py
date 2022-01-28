from typing import Any, List, Tuple, Union

import numba as nb
import numpy as np
from scipy.linalg import expm
from scipy.sparse.csr import csr_matrix
from scipy.spatial.transform import Rotation as R


def matrix_to_vector(mat: np.ndarray) -> np.ndarray:
    if mat.ndim >= 3:
        raise ValueError("Must be at most 2D")

    if mat.ndim == 2:
        return mat.reshape(-1)
    else:
        return mat


def vector_to_matrix(vec: np.ndarray, cols: int) -> np.ndarray:
    if vec.ndim == 2:
        return vec
    else:
        return vec.reshape((vec.shape[0] // cols, cols))


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


@nb.jit
def polar_decomp_2d(m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Perform polar decomposition (A=UP) for 2x2 matrix.

    Mathematical concept refers to https://en.wikipedia.org/wiki/Polar_decomposition.

    Args:
        m (np.ndarray): input 2x2 matrix `m`.

    Returns:
        Decomposed 2x2 matrices `U` and `P`.
    """
    x = m[0, 0] + m[1, 1]
    y = m[1, 0] - m[0, 1]
    denom = np.sqrt(x * x + y * y) + 1e-10
    scale = 1.0 / denom
    c = x * scale
    s = y * scale
    r = np.array([[c, -s], [s, c]], dtype=np.float64)
    s = r.T @ m
    return r, s


@nb.njit
def polar_decomp_3d(m: np.ndarray):
    """Perform polar decomposition (A=UP) for 3x3 matrix.

    Mathematical concept refers to https://en.wikipedia.org/wiki/Polar_decomposition.

    Args:
        m (np.ndarray): input 3x3 matrix `m`.

    Returns:
        Decomposed 3x3 matrices `U` and `P`.
    """
    w, s, vh = np.linalg.svd(m, full_matrices=False)
    u = w.dot(vh)
    # a = up
    p = (vh.T.conj() * s).dot(vh)
    return u, p


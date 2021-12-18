from typing import Any, List, Tuple, Union

import numpy as np
import scipy as sp
from scipy.sparse.csr import csr_matrix
from scipy.spatial.transform import Rotation as R


def is_square_matrix(mat: np.ndarray):
    return len(mat.shape) == 2 and mat.shape[0] == mat.shape[1]


def normalized(a: np.ndarray):
    return a / np.linalg.norm(a)


def angle_between_vectors(a: np.ndarray, b: np.ndarray) -> float:
    return np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))


def distance(a: np.array, b: np.array) -> float:
    return np.linalg.norm(a - b)


def angle_axis(angle, axis):
    return sp.linalg.expm(np.cross(np.eye(3), normalized(axis) * angle))


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
    i: Union[np.ndarray, List[int]], j: Union[np.ndarray, List[int]], v: Union[np.ndarray, List[Any]], m: int, n: int
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


def fast_diagonal_inverse(mat: np.ndarray):
    """Computes the fast diagonal inverse of a diagonal matrix.

    Args:
        mat (np.ndarray): The input matrix which is dense or sparse
    """
    for i, _ in enumerate(mat):
        mat[i, i] = 1 / mat[i, i]

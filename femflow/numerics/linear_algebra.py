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
    r = np.array([[c, -s], [s, c]])
    s = r.T * m
    return r, s


SVD = namedtuple("SVD", ["u", "sigma", "v"])


def svd_2d(m: np.ndarray) -> SVD:
    """Stable SVD for 2d matrices

    Based on http://math.ucla.edu/~cffjiang/research/svd/svd.pdf
    Algorithm 4

    Args:
        m (np.ndarray): m The 2d matrix

    Returns:
        SVD: The svd result
    """
    sig = np.zeros((2, 2))
    v = np.zeros((2, 2))
    u, s = polar_decomp(m)

    c = 0
    s_ = 0

    if abs(s[0, 1]) < 1e-5:
        sig = s.copy()
        c = 1
        s_ = 0
    else:
        tao = 0.5 * (s[0, 0] - s[1, 1])
        w = np.sqrt(tao * tao + s[0, 1] * s[0, 1])
        t = s[0, 1] / (tao + w) if tao > 0 else s[0, 1] / (tao - w)
        c = 1.0 / np.sqrt(t * t + 1)
        s_ = -t * c
        sig[0, 0] = pow(c, 2) * s[0, 0] - 2 * c * s_ * s[0, 1] + pow(s_, 2) * s[1, 1]
        sig[1, 1] = pow(s_, 2) * s[0, 0] + 2 * c * s_ * s[0, 1] + pow(c, 2) * s[1, 1]

    if s[0, 0] < sig[1, 1]:
        sig[0, 0], sig[1, 1] = sig[1, 1], sig[0, 0]
        v[0, 0] = -s_
        v[0, 1] = -c
        v[1, 0] = c
        v[1, 1] = -s_
    else:
        v[0, 0] = c
        v[0, 1] = -s_
        v[1, 0] = s_
        v[1, 1] = c
    v = v.T

    u = np.matmul(u, v)

    return SVD(u, sig, v)


def svd_3d(m: np.ndarray) -> SVD:
    """An implementation of Eftychios Sifakis' 3D Matrix SVD Algorithm for SIMD.
    It's fast as _fuck_

    http://pages.cs.wisc.edu/~sifakis/project_pages/svd.html
    Computing the Singular Value Decomposition of 3x3 matrices with minimal
    branching and elementary floating point operations
    A. McAdams, A. Selle, R. Tamstorf, J. Teran and E. Sifakis

    //#####################################################################
    // Copyright (c) 2010-2011, Eftychios Sifakis.
    //
    // Redistribution and use in source and binary forms, with or without
    // modification, are permitted provided that the following conditions are met:
    //   * Redistributions of source code must retain the above copyright notice,
    //   this list of conditions and the following disclaimer.
    //   * Redistributions in binary form must reproduce the above copyright notice,
    //   this list of conditions and the following disclaimer in the documentation
    //   and/or
    //     other materials provided with the distribution.
    //
    // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    // ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
    // LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    // CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    // SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    // INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    // CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    // ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    // POSSIBILITY OF SUCH DAMAGE.
    //#####################################################################

    Args:
        m (np.ndarray): m

    Returns:
        SVD:
    """
    pass


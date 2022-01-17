from collections import namedtuple
from dataclasses import dataclass
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


def svd_3d(A: np.ndarray, iters=8) -> SVD:
    """Perform singular value decomposition (A=USV^T) for 3x3 matrix.
    Mathematical concept refers to
    https://en.wikipedia.org/wiki/Singular_value_decomposition.
    Args:
        A (ti.Matrix(3, 3)): input 3x3 matrix `A`.
        iters (int): iteration number to control algorithm precision.
    Returns:
        Decomposed 3x3 matrices `U`, 'S' and `V`.
    """

    class storage:
        def __init__(self):
            self.f = 0.0
            self.ui = 0

    def rsqrt(f: float):
        return 1 / np.sqrt(f)

    # sqrt(8.) + 3.
    four_gamma_squared = 5.82842712474619

    # .5 * sqrt(2. - sqrt(2.))
    sine_pi_over_eight = 0.3826834323650897

    # .5 * sqrt(2. + sqrt(2.))
    cosine_pi_over_eight = 0.9238795325112867

    Sfour_gamma_squared = storage()
    Ssine_pi_over_eight = storage()
    Scosine_pi_over_eight = storage()
    Sone_half = storage()
    Sone = storage()
    Stiny_number = storage()
    Ssmall_number = storage()
    Sa11 = storage()
    Sa21 = storage()
    Sa31 = storage()
    Sa12 = storage()
    Sa22 = storage()
    Sa32 = storage()
    Sa13 = storage()
    Sa23 = storage()
    Sa33 = storage()

    Sv11 = storage()
    Sv21 = storage()
    Sv31 = storage()
    Sv12 = storage()
    Sv22 = storage()
    Sv32 = storage()
    Sv13 = storage()
    Sv23 = storage()
    Sv33 = storage()
    Su11 = storage()
    Su21 = storage()
    Su31 = storage()
    Su12 = storage()
    Su22 = storage()
    Su32 = storage()
    Su13 = storage()
    Su23 = storage()
    Su33 = storage()
    Sc = storage()
    Ss = storage()
    Sch = storage()
    Ssh = storage()
    Stmp1 = storage()
    Stmp2 = storage()
    Stmp3 = storage()
    Stmp4 = storage()
    Stmp5 = storage()
    Sqvs = storage()
    Sqvvx = storage()
    Sqvvy = storage()
    Sqvvz = storage()

    Ss11 = storage()
    Ss21 = storage()
    Ss31 = storage()
    Ss22 = storage()
    Ss32 = storage()
    Ss33 = storage()

    Sfour_gamma_squared.f = four_gamma_squared
    Ssine_pi_over_eight.f = sine_pi_over_eight
    Scosine_pi_over_eight.f = cosine_pi_over_eight
    Sone_half.f = 0.5
    Sone.f = 1.0
    Stiny_number.f = 1.0e-20
    Ssmall_number.f = 1.0e-12

    Sa11.f = A[1, 1]
    Sa21.f = A[2, 1]
    Sa31.f = A[3, 1]
    Sa12.f = A[1, 2]
    Sa22.f = A[2, 2]
    Sa32.f = A[3, 2]
    Sa13.f = A[1, 3]
    Sa23.f = A[2, 3]
    Sa33.f = A[3, 3]

    Sqvs.f = 1.0
    Sqvvx.f = 0.0
    Sqvvy.f = 0.0
    Sqvvz.f = 0.0

    Ss11.f = Sa11.f * Sa11.f
    Stmp1.f = Sa21.f * Sa21.f
    Ss11.f = Stmp1.f + Ss11.f
    Stmp1.f = Sa31.f * Sa31.f
    Ss11.f = Stmp1.f + Ss11.f

    Ss21.f = Sa12.f * Sa11.f
    Stmp1.f = Sa22.f * Sa21.f
    Ss21.f = Stmp1.f + Ss21.f
    Stmp1.f = Sa32.f * Sa31.f
    Ss21.f = Stmp1.f + Ss21.f

    Ss31.f = Sa13.f * Sa11.f
    Stmp1.f = Sa23.f * Sa21.f
    Ss31.f = Stmp1.f + Ss31.f
    Stmp1.f = Sa33.f * Sa31.f
    Ss31.f = Stmp1.f + Ss31.f

    Ss22.f = Sa12.f * Sa12.f
    Stmp1.f = Sa22.f * Sa22.f
    Ss22.f = Stmp1.f + Ss22.f
    Stmp1.f = Sa32.f * Sa32.f
    Ss22.f = Stmp1.f + Ss22.f

    Ss32.f = Sa13.f * Sa12.f
    Stmp1.f = Sa23.f * Sa22.f
    Ss32.f = Stmp1.f + Ss32.f
    Stmp1.f = Sa33.f * Sa32.f
    Ss32.f = Stmp1.f + Ss32.f

    Ss33.f = Sa13.f * Sa13.f
    Stmp1.f = Sa23.f * Sa23.f
    Ss33.f = Stmp1.f + Ss33.f
    Stmp1.f = Sa33.f * Sa33.f
    Ss33.f = Stmp1.f + Ss33.f

    for _ in range(iters):
        Ssh.f = Ss21.f * Sone_half.f
        Stmp5.f = Ss11.f - Ss22.f

        Stmp2.f = Ssh.f * Ssh.f
        Stmp1.ui = 0xFFFFFFFF if (Stmp2.f >= Stiny_number.f) else 0
        Ssh.ui = Stmp1.ui & Ssh.ui
        Sch.ui = Stmp1.ui & Stmp5.ui
        Stmp2.ui = ~Stmp1.ui & Sone.ui
        Sch.ui = Sch.ui | Stmp2.ui

        Stmp1.f = Ssh.f * Ssh.f
        Stmp2.f = Sch.f * Sch.f
        Stmp3.f = Stmp1.f + Stmp2.f
        Stmp4.f = rsqrt(Stmp3.f)
        Ssh.f = Stmp4.f * Ssh.f
        Sch.f = Stmp4.f * Sch.f

        Stmp1.f = Sfour_gamma_squared.f * Stmp1.f
        Stmp1.ui = 0xFFFFFFFF if Stmp2.f <= Stmp1.f else 0

        Stmp2.ui = Ssine_pi_over_eight.ui & Stmp1.ui
        Ssh.ui = ~Stmp1.ui & Ssh.ui
        Ssh.ui = Ssh.ui | Stmp2.ui
        Stmp2.ui = Scosine_pi_over_eight.ui & Stmp1.ui
        Sch.ui = ~Stmp1.ui & Sch.ui
        Sch.ui = Sch.ui | Stmp2.ui

        Stmp1.f = Ssh.f * Ssh.f
        Stmp2.f = Sch.f * Sch.f
        Sc.f = Stmp2.f - Stmp1.f
        Ss.f = Sch.f * Ssh.f
        Ss.f = Ss.f + Ss.f

        Stmp3.f = Stmp1.f + Stmp2.f
        Ss33.f = Ss33.f * Stmp3.f
        Ss31.f = Ss31.f * Stmp3.f
        Ss32.f = Ss32.f * Stmp3.f
        Ss33.f = Ss33.f * Stmp3.f

        Stmp1.f = Ss.f * Ss31.f
        Stmp2.f = Ss.f * Ss32.f
        Ss31.f = Sc.f * Ss31.f
        Ss32.f = Sc.f * Ss32.f
        Ss31.f = Stmp2.f + Ss31.f
        Ss32.f = Ss32.f - Stmp1.f

        Stmp2.f = Ss.f * Ss.f
        Stmp1.f = Ss22.f * Stmp2.f
        Stmp3.f = Ss11.f * Stmp2.f
        Stmp4.f = Sc.f * Sc.f
        Ss11.f = Ss11.f * Stmp4.f
        Ss22.f = Ss22.f * Stmp4.f
        Ss11.f = Ss11.f + Stmp1.f
        Ss22.f = Ss22.f + Stmp3.f
        Stmp4.f = Stmp4.f - Stmp2.f
        Stmp2.f = Ss21.f + Ss21.f
        Ss21.f = Ss21.f * Stmp4.f
        Stmp4.f = Sc.f * Ss.f
        Stmp2.f = Stmp2.f * Stmp4.f
        Stmp5.f = Stmp5.f * Stmp4.f
        Ss11.f = Ss11.f + Stmp2.f
        Ss21.f = Ss21.f - Stmp5.f
        Ss22.f = Ss22.f - Stmp2.f

        Stmp1.f = Ssh.f * Sqvvx.f
        Stmp2.f = Ssh.f * Sqvvy.f
        Stmp3.f = Ssh.f * Sqvvz.f
        Ssh.f = Ssh.f * Sqvs.f

        Sqvs.f = Sch.f * Sqvs.f
        Sqvvx.f = Sch.f * Sqvvx.f
        Sqvvy.f = Sch.f * Sqvvy.f
        Sqvvz.f = Sch.f * Sqvvz.f

        Sqvvz.f = Sqvvz.f + Ssh.f
        Sqvs.f = Sqvs.f - Stmp3.f
        Sqvvx.f = Sqvvx.f + Stmp2.f
        Sqvvy.f = Sqvvy.f - Stmp1.f
        Ssh.f = Ss32.f * Sone_half.f
        Stmp5.f = Ss22.f - Ss33.f

        Stmp2.f = Ssh.f * Ssh.f
        Stmp1.ui = 0xFFFFFFFF if Stmp2.f >= Stiny_number.f else 0
        Ssh.ui = Stmp1.ui & Ssh.ui
        Sch.ui = Stmp1.ui & Stmp5.ui
        Stmp2.ui = ~Stmp1.ui & Sone.ui
        Sch.ui = Sch.ui | Stmp2.ui

        Stmp1.f = Ssh.f * Ssh.f
        Stmp2.f = Sch.f * Sch.f
        Stmp3.f = Stmp1.f + Stmp2.f
        Stmp4.f = rsqrt(Stmp3.f)
        Ssh.f = Stmp4.f * Ssh.f
        Sch.f = Stmp4.f * Sch.f

        Stmp1.f = Sfour_gamma_squared.f * Stmp1.f
        Stmp1.ui = 0xFFFFFFFF if Stmp2.f <= Stmp1.f else 0

        Stmp2.ui = Ssine_pi_over_eight.ui & Stmp1.ui
        Ssh.ui = ~Stmp1.ui & Ssh.ui
        Ssh.ui = Ssh.ui | Stmp2.ui
        Stmp2.ui = Scosine_pi_over_eight.ui & Stmp1.ui
        Sch.ui = ~Stmp1.ui & Sch.ui
        Sch.ui = Sch.ui | Stmp2.ui

        Stmp1.f = Ssh.f * Ssh.f
        Stmp2.f = Sch.f * Sch.f
        Sc.f = Stmp2.f - Stmp1.f
        Ss.f = Sch.f * Ssh.f
        Ss.f = Ss.f + Ss.f

        Stmp3.f = Stmp1.f + Stmp2.f
        Ss11.f = Ss11.f * Stmp3.f
        Ss21.f = Ss21.f * Stmp3.f
        Ss31.f = Ss31.f * Stmp3.f
        Ss11.f = Ss11.f * Stmp3.f

        Stmp1.f = Ss.f * Ss21.f
        Stmp2.f = Ss.f * Ss31.f
        Ss21.f = Sc.f * Ss21.f
        Ss31.f = Sc.f * Ss31.f
        Ss21.f = Stmp2.f + Ss21.f
        Ss31.f = Ss31.f - Stmp1.f

        Stmp2.f = Ss.f * Ss.f
        Stmp1.f = Ss33.f * Stmp2.f
        Stmp3.f = Ss22.f * Stmp2.f
        Stmp4.f = Sc.f * Sc.f
        Ss22.f = Ss22.f * Stmp4.f
        Ss33.f = Ss33.f * Stmp4.f
        Ss22.f = Ss22.f + Stmp1.f
        Ss33.f = Ss33.f + Stmp3.f
        Stmp4.f = Stmp4.f - Stmp2.f
        Stmp2.f = Ss32.f + Ss32.f
        Ss32.f = Ss32.f * Stmp4.f
        Stmp4.f = Sc.f * Ss.f
        Stmp2.f = Stmp2.f * Stmp4.f
        Stmp5.f = Stmp5.f * Stmp4.f
        Ss22.f = Ss22.f + Stmp2.f
        Ss32.f = Ss32.f - Stmp5.f
        Ss33.f = Ss33.f - Stmp2.f

        Stmp1.f = Ssh.f * Sqvvx.f
        Stmp2.f = Ssh.f * Sqvvy.f
        Stmp3.f = Ssh.f * Sqvvz.f
        Ssh.f = Ssh.f * Sqvs.f

        Sqvs.f = Sch.f * Sqvs.f
        Sqvvx.f = Sch.f * Sqvvx.f
        Sqvvy.f = Sch.f * Sqvvy.f
        Sqvvz.f = Sch.f * Sqvvz.f

        Sqvvx.f = Sqvvx.f + Ssh.f
        Sqvs.f = Sqvs.f - Stmp1.f
        Sqvvy.f = Sqvvy.f + Stmp3.f
        Sqvvz.f = Sqvvz.f - Stmp2.f
        Ssh.f = Ss31.f * Sone_half.f
        Stmp5.f = Ss33.f - Ss11.f

        Stmp2.f = Ssh.f * Ssh.f
        Stmp1.ui = 0xFFFFFFFF if Stmp2.f >= Stiny_number.f else 0
        Ssh.ui = Stmp1.ui & Ssh.ui
        Sch.ui = Stmp1.ui & Stmp5.ui
        Stmp2.ui = ~Stmp1.ui & Sone.ui
        Sch.ui = Sch.ui | Stmp2.ui

        Stmp1.f = Ssh.f * Ssh.f
        Stmp2.f = Sch.f * Sch.f
        Stmp3.f = Stmp1.f + Stmp2.f
        Stmp4.f = rsqrt(Stmp3.f)
        Ssh.f = Stmp4.f * Ssh.f
        Sch.f = Stmp4.f * Sch.f

        Stmp1.f = Sfour_gamma_squared.f * Stmp1.f
        Stmp1.ui = 0xFFFFFFFF if Stmp2.f <= Stmp1.f else 0

        Stmp2.ui = Ssine_pi_over_eight.ui & Stmp1.ui
        Ssh.ui = ~Stmp1.ui & Ssh.ui
        Ssh.ui = Ssh.ui | Stmp2.ui
        Stmp2.ui = Scosine_pi_over_eight.ui & Stmp1.ui
        Sch.ui = ~Stmp1.ui & Sch.ui
        Sch.ui = Sch.ui | Stmp2.ui

        Stmp1.f = Ssh.f * Ssh.f
        Stmp2.f = Sch.f * Sch.f
        Sc.f = Stmp2.f - Stmp1.f
        Ss.f = Sch.f * Ssh.f
        Ss.f = Ss.f + Ss.f

        Stmp3.f = Stmp1.f + Stmp2.f
        Ss22.f = Ss22.f * Stmp3.f
        Ss32.f = Ss32.f * Stmp3.f
        Ss21.f = Ss21.f * Stmp3.f
        Ss22.f = Ss22.f * Stmp3.f

        Stmp1.f = Ss.f * Ss32.f
        Stmp2.f = Ss.f * Ss21.f
        Ss32.f = Sc.f * Ss32.f
        Ss21.f = Sc.f * Ss21.f
        Ss32.f = Stmp2.f + Ss32.f
        Ss21.f = Ss21.f - Stmp1.f

        Stmp2.f = Ss.f * Ss.f
        Stmp1.f = Ss11.f * Stmp2.f
        Stmp3.f = Ss33.f * Stmp2.f
        Stmp4.f = Sc.f * Sc.f
        Ss33.f = Ss33.f * Stmp4.f
        Ss11.f = Ss11.f * Stmp4.f
        Ss33.f = Ss33.f + Stmp1.f
        Ss11.f = Ss11.f + Stmp3.f
        Stmp4.f = Stmp4.f - Stmp2.f
        Stmp2.f = Ss31.f + Ss31.f
        Ss31.f = Ss31.f * Stmp4.f
        Stmp4.f = Sc.f * Ss.f
        Stmp2.f = Stmp2.f * Stmp4.f
        Stmp5.f = Stmp5.f * Stmp4.f
        Ss33.f = Ss33.f + Stmp2.f
        Ss31.f = Ss31.f - Stmp5.f
        Ss11.f = Ss11.f - Stmp2.f

        Stmp1.f = Ssh.f * Sqvvx.f
        Stmp2.f = Ssh.f * Sqvvy.f
        Stmp3.f = Ssh.f * Sqvvz.f
        Ssh.f = Ssh.f * Sqvs.f

        Sqvs.f = Sch.f * Sqvs.f
        Sqvvx.f = Sch.f * Sqvvx.f
        Sqvvy.f = Sch.f * Sqvvy.f
        Sqvvz.f = Sch.f * Sqvvz.f

        Sqvvy.f = Sqvvy.f + Ssh.f
        Sqvs.f = Sqvs.f - Stmp2.f
        Sqvvz.f = Sqvvz.f + Stmp1.f
        Sqvvx.f = Sqvvx.f - Stmp3.f

    Stmp2.f = Sqvs.f * Sqvs.f
    Stmp1.f = Sqvvx.f * Sqvvx.f
    Stmp2.f = Stmp1.f + Stmp2.f
    Stmp1.f = Sqvvy.f * Sqvvy.f
    Stmp2.f = Stmp1.f + Stmp2.f
    Stmp1.f = Sqvvz.f * Sqvvz.f
    Stmp2.f = Stmp1.f + Stmp2.f

    Stmp1.f = rsqrt(Stmp2.f)
    Stmp4.f = Stmp1.f * Sone_half.f
    Stmp3.f = Stmp1.f * Stmp4.f
    Stmp3.f = Stmp1.f * Stmp3.f
    Stmp3.f = Stmp2.f * Stmp3.f
    Stmp1.f = Stmp1.f + Stmp4.f
    Stmp1.f = Stmp1.f - Stmp3.f

    Sqvs.f = Sqvs.f * Stmp1.f
    Sqvvx.f = Sqvvx.f * Stmp1.f
    Sqvvy.f = Sqvvy.f * Stmp1.f
    Sqvvz.f = Sqvvz.f * Stmp1.f

    Stmp1.f = Sqvvx.f * Sqvvx.f
    Stmp2.f = Sqvvy.f * Sqvvy.f
    Stmp3.f = Sqvvz.f * Sqvvz.f
    Sv11.f = Sqvs.f * Sqvs.f
    Sv22.f = Sv11.f - Stmp1.f
    Sv33.f = Sv22.f - Stmp2.f
    Sv33.f = Sv33.f + Stmp3.f
    Sv22.f = Sv22.f + Stmp2.f
    Sv22.f = Sv22.f - Stmp3.f
    Sv11.f = Sv11.f + Stmp1.f
    Sv11.f = Sv11.f - Stmp2.f
    Sv11.f = Sv11.f - Stmp3.f
    Stmp1.f = Sqvvx.f + Sqvvx.f
    Stmp2.f = Sqvvy.f + Sqvvy.f
    Stmp3.f = Sqvvz.f + Sqvvz.f
    Sv32.f = Sqvs.f * Stmp1.f
    Sv13.f = Sqvs.f * Stmp2.f
    Sv21.f = Sqvs.f * Stmp3.f
    Stmp1.f = Sqvvy.f * Stmp1.f
    Stmp2.f = Sqvvz.f * Stmp2.f
    Stmp3.f = Sqvvx.f * Stmp3.f
    Sv12.f = Stmp1.f - Sv21.f
    Sv23.f = Stmp2.f - Sv32.f
    Sv31.f = Stmp3.f - Sv13.f
    Sv21.f = Stmp1.f + Sv21.f
    Sv32.f = Stmp2.f + Sv32.f
    Sv13.f = Stmp3.f + Sv13.f
    Stmp2.f = Sa12.f
    Stmp3.f = Sa13.f
    Sa12.f = Sv12.f * Sa11.f
    Sa13.f = Sv13.f * Sa11.f
    Sa11.f = Sv11.f * Sa11.f
    Stmp1.f = Sv21.f * Stmp2.f
    Sa11.f = Sa11.f + Stmp1.f
    Stmp1.f = Sv31.f * Stmp3.f
    Sa11.f = Sa11.f + Stmp1.f
    Stmp1.f = Sv22.f * Stmp2.f
    Sa12.f = Sa12.f + Stmp1.f
    Stmp1.f = Sv32.f * Stmp3.f
    Sa12.f = Sa12.f + Stmp1.f
    Stmp1.f = Sv23.f * Stmp2.f
    Sa13.f = Sa13.f + Stmp1.f
    Stmp1.f = Sv33.f * Stmp3.f
    Sa13.f = Sa13.f + Stmp1.f

    Stmp2.f = Sa22.f
    Stmp3.f = Sa23.f
    Sa22.f = Sv12.f * Sa21.f
    Sa23.f = Sv13.f * Sa21.f
    Sa21.f = Sv11.f * Sa21.f
    Stmp1.f = Sv21.f * Stmp2.f
    Sa21.f = Sa21.f + Stmp1.f
    Stmp1.f = Sv31.f * Stmp3.f
    Sa21.f = Sa21.f + Stmp1.f
    Stmp1.f = Sv22.f * Stmp2.f
    Sa22.f = Sa22.f + Stmp1.f
    Stmp1.f = Sv32.f * Stmp3.f
    Sa22.f = Sa22.f + Stmp1.f
    Stmp1.f = Sv23.f * Stmp2.f
    Sa23.f = Sa23.f + Stmp1.f
    Stmp1.f = Sv33.f * Stmp3.f
    Sa23.f = Sa23.f + Stmp1.f

    Stmp2.f = Sa32.f
    Stmp3.f = Sa33.f
    Sa32.f = Sv12.f * Sa31.f
    Sa33.f = Sv13.f * Sa31.f
    Sa31.f = Sv11.f * Sa31.f
    Stmp1.f = Sv21.f * Stmp2.f
    Sa31.f = Sa31.f + Stmp1.f
    Stmp1.f = Sv31.f * Stmp3.f
    Sa31.f = Sa31.f + Stmp1.f
    Stmp1.f = Sv22.f * Stmp2.f
    Sa32.f = Sa32.f + Stmp1.f
    Stmp1.f = Sv32.f * Stmp3.f
    Sa32.f = Sa32.f + Stmp1.f
    Stmp1.f = Sv23.f * Stmp2.f
    Sa33.f = Sa33.f + Stmp1.f
    Stmp1.f = Sv33.f * Stmp3.f
    Sa33.f = Sa33.f + Stmp1.f

    Stmp1.f = Sa11.f * Sa11.f
    Stmp4.f = Sa21.f * Sa21.f
    Stmp1.f = Stmp1.f + Stmp4.f
    Stmp4.f = Sa31.f * Sa31.f
    Stmp1.f = Stmp1.f + Stmp4.f

    Stmp2.f = Sa12.f * Sa12.f
    Stmp4.f = Sa22.f * Sa22.f
    Stmp2.f = Stmp2.f + Stmp4.f
    Stmp4.f = Sa32.f * Sa32.f
    Stmp2.f = Stmp2.f + Stmp4.f

    Stmp3.f = Sa13.f * Sa13.f
    Stmp4.f = Sa23.f * Sa23.f
    Stmp3.f = Stmp3.f + Stmp4.f
    Stmp4.f = Sa33.f * Sa33.f
    Stmp3.f = Stmp3.f + Stmp4.f

    Stmp4.ui = 0xffffffff if Stmp1.f < Stmp2.f else 0

    Stmp5.ui = Sa11.ui ^ Sa12.ui
    Stmp5.ui = Stmp5.ui & Stmp4.ui
    Sa11.ui = Sa11.ui ^ Stmp5.ui
    Sa12.ui = Sa12.ui ^ Stmp5.ui

    Stmp5.ui = Sa21.ui ^ Sa22.ui
    Stmp5.ui = Stmp5.ui & Stmp4.ui
    Sa21.ui = Sa21.ui ^ Stmp5.ui
    Sa22.ui = Sa22.ui ^ Stmp5.ui

    Stmp5.ui = Sa31.ui ^ Sa32.ui
    Stmp5.ui = Stmp5.ui & Stmp4.ui
    Sa31.ui = Sa31.ui ^ Stmp5.ui
    Sa32.ui = Sa32.ui ^ Stmp5.ui

    Stmp5.ui = Sv11.ui ^ Sv12.ui
    Stmp5.ui = Stmp5.ui & Stmp4.ui
    Sv11.ui = Sv11.ui ^ Stmp5.ui
    Sv12.ui = Sv12.ui ^ Stmp5.ui

    Stmp5.ui = Sv21.ui ^ Sv22.ui
    Stmp5.ui = Stmp5.ui & Stmp4.ui
    Sv21.ui = Sv21.ui ^ Stmp5.ui
    Sv22.ui = Sv22.ui ^ Stmp5.ui

    Stmp5.ui = Sv31.ui ^ Sv32.ui
    Stmp5.ui = Stmp5.ui & Stmp4.ui
    Sv31.ui = Sv31.ui ^ Stmp5.ui
    Sv32.ui = Sv32.ui ^ Stmp5.ui

    Stmp5.ui = Stmp1.ui ^ Stmp2.ui
    Stmp5.ui = Stmp5.ui & Stmp4.ui
    Stmp1.ui = Stmp1.ui ^ Stmp5.ui
    Stmp2.ui = Stmp2.ui ^ Stmp5.ui

    Stmp5.f = -2.0
    Stmp5.ui = Stmp5.ui & Stmp4.ui
    Stmp4.f = 1.0
    Stmp4.f = Stmp4.f + Stmp5.f

    Sa12.f = Sa12.f * Stmp4.f
    Sa22.f = Sa22.f * Stmp4.f
    Sa32.f = Sa32.f * Stmp4.f

    Sv12.f = Sv12.f * Stmp4.f
    Sv22.f = Sv22.f * Stmp4.f
    Sv32.f = Sv32.f * Stmp4.f

    Stmp4.ui = 0xffffffff if Stmp1.f < Stmp3.f else 0

    Stmp5.ui = Sa11.ui ^ Sa13.ui
    Stmp5.ui = Stmp5.ui & Stmp4.ui
    Sa11.ui = Sa11.ui ^ Stmp5.ui
    Sa13.ui = Sa13.ui ^ Stmp5.ui

    Stmp5.ui = Sa21.ui ^ Sa23.ui
    Stmp5.ui = Stmp5.ui & Stmp4.ui
    Sa21.ui = Sa21.ui ^ Stmp5.ui
    Sa23.ui = Sa23.ui ^ Stmp5.ui

    Stmp5.ui = Sa31.ui ^ Sa33.ui
    Stmp5.ui = Stmp5.ui & Stmp4.ui
    Sa31.ui = Sa31.ui ^ Stmp5.ui
    Sa33.ui = Sa33.ui ^ Stmp5.ui

    Stmp5.ui = Sv11.ui ^ Sv13.ui
    Stmp5.ui = Stmp5.ui & Stmp4.ui
    Sv11.ui = Sv11.ui ^ Stmp5.ui
    Sv13.ui = Sv13.ui ^ Stmp5.ui

    Stmp5.ui = Sv21.ui ^ Sv23.ui
    Stmp5.ui = Stmp5.ui & Stmp4.ui
    Sv21.ui = Sv21.ui ^ Stmp5.ui
    Sv23.ui = Sv23.ui ^ Stmp5.ui

    Stmp5.ui = Sv31.ui ^ Sv33.ui
    Stmp5.ui = Stmp5.ui & Stmp4.ui
    Sv31.ui = Sv31.ui ^ Stmp5.ui
    Sv33.ui = Sv33.ui ^ Stmp5.ui

    Stmp5.ui = Stmp1.ui ^ Stmp3.ui
    Stmp5.ui = Stmp5.ui & Stmp4.ui
    Stmp1.ui = Stmp1.ui ^ Stmp5.ui
    Stmp3.ui = Stmp3.ui ^ Stmp5.ui

    Stmp5.f = -2.0
    Stmp5.ui = Stmp5.ui & Stmp4.ui
    Stmp4.f = 1.0
    Stmp4.f = Stmp4.f + Stmp5.f

    Sa11.f = Sa11.f * Stmp4.f
    Sa21.f = Sa21.f * Stmp4.f
    Sa31.f = Sa31.f * Stmp4.f

    Sv11.f = Sv11.f * Stmp4.f
    Sv21.f = Sv21.f * Stmp4.f
    Sv31.f = Sv31.f * Stmp4.f

    Stmp4.ui = 0xffffffff if Stmp2.f < Stmp3.f else 0

    Stmp5.ui = Sa12.ui ^ Sa13.ui
    Stmp5.ui = Stmp5.ui & Stmp4.ui
    Sa12.ui = Sa12.ui ^ Stmp5.ui
    Sa13.ui = Sa13.ui ^ Stmp5.ui

    Stmp5.ui = Sa22.ui ^ Sa23.ui
    Stmp5.ui = Stmp5.ui & Stmp4.ui
    Sa22.ui = Sa22.ui ^ Stmp5.ui
    Sa23.ui = Sa23.ui ^ Stmp5.ui

    Stmp5.ui = Sa32.ui ^ Sa33.ui
    Stmp5.ui = Stmp5.ui & Stmp4.ui
    Sa32.ui = Sa32.ui ^ Stmp5.ui
    Sa33.ui = Sa33.ui ^ Stmp5.ui

    Stmp5.ui = Sv12.ui ^ Sv13.ui
    Stmp5.ui = Stmp5.ui & Stmp4.ui
    Sv12.ui = Sv12.ui ^ Stmp5.ui
    Sv13.ui = Sv13.ui ^ Stmp5.ui

    Stmp5.ui = Sv22.ui ^ Sv23.ui
    Stmp5.ui = Stmp5.ui & Stmp4.ui
    Sv22.ui = Sv22.ui ^ Stmp5.ui
    Sv23.ui = Sv23.ui ^ Stmp5.ui

    Stmp5.ui = Sv32.ui ^ Sv33.ui
    Stmp5.ui = Stmp5.ui & Stmp4.ui
    Sv32.ui = Sv32.ui ^ Stmp5.ui
    Sv33.ui = Sv33.ui ^ Stmp5.ui

    Stmp5.ui = Stmp2.ui ^ Stmp3.ui
    Stmp5.ui = Stmp5.ui & Stmp4.ui
    Stmp2.ui = Stmp2.ui ^ Stmp5.ui
    Stmp3.ui = Stmp3.ui ^ Stmp5.ui

    Stmp5.f = -2.0
    Stmp5.ui = Stmp5.ui & Stmp4.ui
    Stmp4.f = 1.0
    Stmp4.f = Stmp4.f + Stmp5.f

    Sa13.f = Sa13.f * Stmp4.f
    Sa23.f = Sa23.f * Stmp4.f
    Sa33.f = Sa33.f * Stmp4.f

    Sv13.f = Sv13.f * Stmp4.f
    Sv23.f = Sv23.f * Stmp4.f
    Sv33.f = Sv33.f * Stmp4.f
    Su11.f = 1.0
    Su21.f = 0.0
    Su31.f = 0.0
    Su12.f = 0.0
    Su22.f = 1.0
    Su32.f = 0.0
    Su13.f = 0.0
    Su23.f = 0.0
    Su33.f = 1.0
    Ssh.f = Sa21.f * Sa21.f

    Ssh.ui = 0xffffffff if Ssh.f >= Ssmall_number.f else 0

    Ssh.ui = Ssh.ui & Sa21.ui;

    Stmp5.f = 0.0
    Sch.f = Stmp5.f - Sa11.f
    Sch.f = max(Sch.f, Sa11.f)
    Sch.f = max(Sch.f, Ssmall_number.f)
    Stmp5.ui = 0xffffffff if Sa11.f >= Stmp5.f else 0

    Stmp1.f = Sch.f * Sch.f
    Stmp2.f = Ssh.f * Ssh.f
    Stmp2.f = Stmp1.f + Stmp2.f
    Stmp1.f = rsqrt(Stmp2.f)

    Stmp4.f = Stmp1.f * Sone_half.f
    Stmp3.f = Stmp1.f * Stmp4.f
    Stmp3.f = Stmp1.f * Stmp3.f
    Stmp3.f = Stmp2.f * Stmp3.f
    Stmp1.f = Stmp1.f + Stmp4.f
    Stmp1.f = Stmp1.f - Stmp3.f
    Stmp1.f = Stmp1.f * Stmp2.f

    Sch.f = Sch.f + Stmp1.f

    Stmp1.ui = ~Stmp5.ui & Ssh.ui
    Stmp2.ui = ~Stmp5.ui & Sch.ui
    Sch.ui = Stmp5.ui & Sch.ui
    Ssh.ui = Stmp5.ui & Ssh.ui
    Sch.ui = Sch.ui | Stmp1.ui
    Ssh.ui = Ssh.ui | Stmp2.ui

    Stmp1.f = Sch.f * Sch.f
    Stmp2.f = Ssh.f * Ssh.f
    Stmp2.f = Stmp1.f + Stmp2.f
    Stmp1.f = rsqrt(Stmp2.f)

    Stmp4.f = Stmp1.f * Sone_half.f
    Stmp3.f = Stmp1.f * Stmp4.f
    Stmp3.f = Stmp1.f * Stmp3.f
    Stmp3.f = Stmp2.f * Stmp3.f
    Stmp1.f = Stmp1.f + Stmp4.f
    Stmp1.f = Stmp1.f - Stmp3.f

    Sch.f = Sch.f * Stmp1.f
    Ssh.f = Ssh.f * Stmp1.f

    Sc.f = Sch.f * Sch.f
    Ss.f = Ssh.f * Ssh.f
    Sc.f = Sc.f - Ss.f
    Ss.f = Ssh.f * Sch.f
    Ss.f = Ss.f + Ss.f

    Stmp1.f = Ss.f * Sa11.f
    Stmp2.f = Ss.f * Sa21.f
    Sa11.f = Sc.f * Sa11.f
    Sa21.f = Sc.f * Sa21.f
    Sa11.f = Sa11.f + Stmp2.f
    Sa21.f = Sa21.f - Stmp1.f

    Stmp1.f = Ss.f * Sa12.f
    Stmp2.f = Ss.f * Sa22.f
    Sa12.f = Sc.f * Sa12.f
    Sa22.f = Sc.f * Sa22.f
    Sa12.f = Sa12.f + Stmp2.f
    Sa22.f = Sa22.f - Stmp1.f

    Stmp1.f = Ss.f * Sa13.f
    Stmp2.f = Ss.f * Sa23.f
    Sa13.f = Sc.f * Sa13.f
    Sa23.f = Sc.f * Sa23.f
    Sa13.f = Sa13.f + Stmp2.f
    Sa23.f = Sa23.f - Stmp1.f

    Stmp1.f = Ss.f * Su11.f
    Stmp2.f = Ss.f * Su12.f
    Su11.f = Sc.f * Su11.f
    Su12.f = Sc.f * Su12.f
    Su11.f = Su11.f + Stmp2.f
    Su12.f = Su12.f - Stmp1.f

    Stmp1.f = Ss.f * Su21.f
    Stmp2.f = Ss.f * Su22.f
    Su21.f = Sc.f * Su21.f
    Su22.f = Sc.f * Su22.f
    Su21.f = Su21.f + Stmp2.f
    Su22.f = Su22.f - Stmp1.f

    Stmp1.f = Ss.f * Su31.f
    Stmp2.f = Ss.f * Su32.f
    Su31.f = Sc.f * Su31.f
    Su32.f = Sc.f * Su32.f
    Su31.f = Su31.f + Stmp2.f
    Su32.f = Su32.f - Stmp1.f
    Ssh.f = Sa31.f * Sa31.f

    Ssh.ui = 0xffffffff if Ssh.f >= Ssmall_number.f else 0

    Ssh.ui = Ssh.ui & Sa31.ui

    Stmp5.f = 0.0
    Sch.f = Stmp5.f - Sa11.f
    Sch.f = max(Sch.f, Sa11.f)
    Sch.f = max(Sch.f, Ssmall_number.f)
    Stmp5.ui = 0xffffffff if Sa11.f >= Stmp5.f else 0

    Stmp1.f = Sch.f * Sch.f
    Stmp2.f = Ssh.f * Ssh.f
    Stmp2.f = Stmp1.f + Stmp2.f
    Stmp1.f = rsqrt(Stmp2.f)

    Stmp4.f = Stmp1.f * Sone_half.f
    Stmp3.f = Stmp1.f * Stmp4.f
    Stmp3.f = Stmp1.f * Stmp3.f
    Stmp3.f = Stmp2.f * Stmp3.f
    Stmp1.f = Stmp1.f + Stmp4.f
    Stmp1.f = Stmp1.f - Stmp3.f
    Stmp1.f = Stmp1.f * Stmp2.f

    Sch.f = Sch.f + Stmp1.f

    Stmp1.ui = ~Stmp5.ui & Ssh.ui
    Stmp2.ui = ~Stmp5.ui & Sch.ui
    Sch.ui = Stmp5.ui & Sch.ui
    Ssh.ui = Stmp5.ui & Ssh.ui
    Sch.ui = Sch.ui | Stmp1.ui
    Ssh.ui = Ssh.ui | Stmp2.ui

    Stmp1.f = Sch.f * Sch.f
    Stmp2.f = Ssh.f * Ssh.f
    Stmp2.f = Stmp1.f + Stmp2.f
    Stmp1.f = rsqrt(Stmp2.f)

    Stmp4.f = Stmp1.f * Sone_half.f
    Stmp3.f = Stmp1.f * Stmp4.f
    Stmp3.f = Stmp1.f * Stmp3.f
    Stmp3.f = Stmp2.f * Stmp3.f
    Stmp1.f = Stmp1.f + Stmp4.f
    Stmp1.f = Stmp1.f - Stmp3.f

    Sch.f = Sch.f * Stmp1.f
    Ssh.f = Ssh.f * Stmp1.f

    Sc.f = Sch.f * Sch.f
    Ss.f = Ssh.f * Ssh.f
    Sc.f = Sc.f - Ss.f
    Ss.f = Ssh.f * Sch.f
    Ss.f = Ss.f + Ss.f

    Stmp1.f = Ss.f * Sa11.f
    Stmp2.f = Ss.f * Sa31.f
    Sa11.f = Sc.f * Sa11.f
    Sa31.f = Sc.f * Sa31.f
    Sa11.f = Sa11.f + Stmp2.f
    Sa31.f = Sa31.f - Stmp1.f

    Stmp1.f = Ss.f * Sa12.f
    Stmp2.f = Ss.f * Sa32.f
    Sa12.f = Sc.f * Sa12.f
    Sa32.f = Sc.f * Sa32.f
    Sa12.f = Sa12.f + Stmp2.f
    Sa32.f = Sa32.f - Stmp1.f

    Stmp1.f = Ss.f * Sa13.f
    Stmp2.f = Ss.f * Sa33.f
    Sa13.f = Sc.f * Sa13.f
    Sa33.f = Sc.f * Sa33.f
    Sa13.f = Sa13.f + Stmp2.f
    Sa33.f = Sa33.f - Stmp1.f

    Stmp1.f = Ss.f * Su11.f
    Stmp2.f = Ss.f * Su13.f
    Su11.f = Sc.f * Su11.f
    Su13.f = Sc.f * Su13.f
    Su11.f = Su11.f + Stmp2.f
    Su13.f = Su13.f - Stmp1.f

    Stmp1.f = Ss.f * Su21.f
    Stmp2.f = Ss.f * Su23.f
    Su21.f = Sc.f * Su21.f
    Su23.f = Sc.f * Su23.f
    Su21.f = Su21.f + Stmp2.f
    Su23.f = Su23.f - Stmp1.f

    Stmp1.f = Ss.f * Su31.f
    Stmp2.f = Ss.f * Su33.f
    Su31.f = Sc.f * Su31.f
    Su33.f = Sc.f * Su33.f
    Su31.f = Su31.f + Stmp2.f
    Su33.f = Su33.f - Stmp1.f
    Ssh.f = Sa32.f * Sa32.f

    Ssh.ui = 0xffffffff if Ssh.f >= Ssmall_number.f else 0

    Ssh.ui = Ssh.ui & Sa32.ui

    Stmp5.f = 0.0
    Sch.f = Stmp5.f - Sa22.f
    Sch.f = max(Sch.f, Sa22.f)
    Sch.f = max(Sch.f, Ssmall_number.f)
    Stmp5.ui = 0xffffffff if Sa22.f >= Stmp5.f else 0

    Stmp1.f = Sch.f * Sch.f
    Stmp2.f = Ssh.f * Ssh.f
    Stmp2.f = Stmp1.f + Stmp2.f
    Stmp1.f = rsqrt(Stmp2.f)

    Stmp4.f = Stmp1.f * Sone_half.f
    Stmp3.f = Stmp1.f * Stmp4.f
    Stmp3.f = Stmp1.f * Stmp3.f
    Stmp3.f = Stmp2.f * Stmp3.f
    Stmp1.f = Stmp1.f + Stmp4.f
    Stmp1.f = Stmp1.f - Stmp3.f
    Stmp1.f = Stmp1.f * Stmp2.f

    Sch.f = Sch.f + Stmp1.f

    Stmp1.ui = ~Stmp5.ui & Ssh.ui
    Stmp2.ui = ~Stmp5.ui & Sch.ui
    Sch.ui = Stmp5.ui & Sch.ui
    Ssh.ui = Stmp5.ui & Ssh.ui
    Sch.ui = Sch.ui | Stmp1.ui
    Ssh.ui = Ssh.ui | Stmp2.ui

    Stmp1.f = Sch.f * Sch.f
    Stmp2.f = Ssh.f * Ssh.f
    Stmp2.f = Stmp1.f + Stmp2.f
    Stmp1.f = rsqrt(Stmp2.f)

    Stmp4.f = Stmp1.f * Sone_half.f
    Stmp3.f = Stmp1.f * Stmp4.f
    Stmp3.f = Stmp1.f * Stmp3.f
    Stmp3.f = Stmp2.f * Stmp3.f
    Stmp1.f = Stmp1.f + Stmp4.f
    Stmp1.f = Stmp1.f - Stmp3.f

    Sch.f = Sch.f * Stmp1.f
    Ssh.f = Ssh.f * Stmp1.f

    Sc.f = Sch.f * Sch.f
    Ss.f = Ssh.f * Ssh.f
    Sc.f = Sc.f - Ss.f
    Ss.f = Ssh.f * Sch.f
    Ss.f = Ss.f + Ss.f

    Stmp1.f = Ss.f * Sa21.f
    Stmp2.f = Ss.f * Sa31.f
    Sa21.f = Sc.f * Sa21.f
    Sa31.f = Sc.f * Sa31.f
    Sa21.f = Sa21.f + Stmp2.f
    Sa31.f = Sa31.f - Stmp1.f

    Stmp1.f = Ss.f * Sa22.f
    Stmp2.f = Ss.f * Sa32.f
    Sa22.f = Sc.f * Sa22.f
    Sa32.f = Sc.f * Sa32.f
    Sa22.f = Sa22.f + Stmp2.f
    Sa32.f = Sa32.f - Stmp1.f

    Stmp1.f = Ss.f * Sa23.f
    Stmp2.f = Ss.f * Sa33.f
    Sa23.f = Sc.f * Sa23.f
    Sa33.f = Sc.f * Sa33.f
    Sa23.f = Sa23.f + Stmp2.f
    Sa33.f = Sa33.f - Stmp1.f

    Stmp1.f = Ss.f * Su12.f
    Stmp2.f = Ss.f * Su13.f
    Su12.f = Sc.f * Su12.f
    Su13.f = Sc.f * Su13.f
    Su12.f = Su12.f + Stmp2.f
    Su13.f = Su13.f - Stmp1.f

    Stmp1.f = Ss.f * Su22.f
    Stmp2.f = Ss.f * Su23.f
    Su22.f = Sc.f * Su22.f
    Su23.f = Sc.f * Su23.f
    Su22.f = Su22.f + Stmp2.f
    Su23.f = Su23.f - Stmp1.f

    Stmp1.f = Ss.f * Su32.f
    Stmp2.f = Ss.f * Su33.f
    Su32.f = Sc.f * Su32.f
    Su33.f = Sc.f * Su33.f
    Su32.f = Su32.f + Stmp2.f
    Su33.f = Su33.f - Stmp1.f
    # end

    u = np.zeros((3, 3))
    sig = np.zeros((3, 3))
    v = np.zeros((3, 3))

    u[1,1] = Su11.f;
    u[2,1] = Su21.f;
    u[3,1] = Su31.f;
    u[1,2] = Su12.f;
    u[2,2] = Su22.f;
    u[3,2] = Su32.f;
    u[1,3] = Su13.f;
    u[2,3] = Su23.f;
    u[3,3] = Su33.f;

    v[1,1] = Sv11.f;
    v[2,1] = Sv21.f;
    v[3,1] = Sv31.f;
    v[1,2] = Sv12.f;
    v[2,2] = Sv22.f;
    v[3,2] = Sv32.f;
    v[1,3] = Sv13.f;
    v[2,3] = Sv23.f;
    v[3,3] = Sv33.f;

    sig[1, 1] = Sa11.f
    sig[2, 2] = Sa22.f
    sig[3, 3] = Sa33.f

    return SVD(u, sig, v)

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

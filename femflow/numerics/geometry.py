import copy
from collections import namedtuple

import igl
import numba as nb
import numpy as np
import wildmeshing as wm
from scipy.sparse import csr_matrix

from .linear_algebra import normalized


def per_face_normals(v: np.ndarray, f: np.ndarray, z: np.ndarray = np.zeros(3)) -> np.ndarray:
    """Computes the per face normals with default for degenerate cases

    Args:
        v (np.ndarray): The vertex positions (n x 3)
        f (np.ndarray): The faces (n x 3)
        z (np.ndarray, optional): The default normal for degenerate faces. Defaults to np.zeros(3).

    Returns:
        np.ndarray: The normal matrix (n x 3)
    """
    assert v.ndim == 2, "V must be a matrix"
    assert f.ndim == 2, "F must be a matrix"
    assert z.shape[0] == 3 and z.ndim == 1, "Z must be a vector 3"
    n = np.zeros((f.shape[0], 3))

    for i, row in enumerate(f):
        x, y, z = row
        p1 = v[y] - v[x]
        p2 = v[z] - v[x]
        normal = np.cross(p1, p2)

        # Degenerate normal
        if np.linalg.norm(normal) == 0:
            n[i, :] = z
        else:
            n[i, :] = normalized(normal)

    return n.astype(np.float32)


TetMesh = namedtuple("TetMesh", ["vertices", "tetrahedra", "faces"])


def tetrahedralize_surface_mesh(v: np.ndarray, f: np.ndarray, stop_quality=1000) -> TetMesh:
    """Computes a tetrahedral mesh from a surface mesh, recomputes faces and returns them. The types are automatically
    converted the the appropriate value.

    Args:
        v (np.ndarray): The vertices, (n x 3)
        f (np.ndarray): The faces, (n x 3)
        stop_quality (int, optional): The stop quality of the tet wild alg. Defaults to 1000.

    Returns:
        TetMesh: Tetmesh of the surface.
    """
    if v.ndim != 2:
        raise ValueError("Vertices must be a matrix")
    if f.ndim != 2:
        raise ValueError("Faces must be a matrix")

    tetrahedralizer = wm.Tetrahedralizer(stop_quality=stop_quality)
    tetrahedralizer.set_mesh(v, f)
    tetrahedralizer.tetrahedralize()
    v, t = tetrahedralizer.get_tet_mesh()
    f = igl.boundary_facets(t)
    v = v.astype(np.float32)
    t = t.astype(np.uint32)
    f = f.astype(np.uint32)
    return TetMesh(v, t, f)


@nb.njit
def tet_volume(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    x1, y1, z1 = a
    x2, y2, z2 = b
    x3, y3, z3 = c
    x4, y4, z4 = d

    x = np.array([[1, x1, y1, z1], [1, x2, y2, z2], [1, x3, y3, z3], [1, x4, y4, z4]])

    return np.float32(np.linalg.det(x) / 6)


def index_sparse_matrix_by_indices(X: csr_matrix, R: np.ndarray, C: np.ndarray = None) -> np.ndarray:
    if C is None:
        C = copy.deepcopy(R)
    assert R.ndim == 1 and C.ndim == 1, "Rows and cols must be vectors"
    rows = R.size
    cols = C.size
    RR = []
    CC = []
    for row in range(rows):
        for col in range(cols):
            RR.append(R[row])
            CC.append(C[col])
    return X[RR, CC].reshape(rows, cols)


@nb.njit
def grid(res: np.ndarray) -> np.ndarray:
    vertices = np.empty((np.prod(res), len(res)))
    sub = np.zeros(res.shape)

    for i in range(vertices.shape[0]):
        for c in range(len(res) - 1):
            if sub[c] >= res[c]:
                sub[c] = 0
                sub[c + 1] += 1
        for c in range(len(res)):
            vertices[i, c] = sub[c] / (res[c] - 1)
        sub[0] += 1
    return vertices

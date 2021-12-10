from collections import namedtuple

import igl
import numpy as np
import wildmeshing as wm

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
    tetrahedralizer = wm.Tetrahedralizer(stop_quality=stop_quality)
    tetrahedralizer.set_mesh(v, f)
    tetrahedralizer.tetrahedralize()
    v, t = tetrahedralizer.get_tet_mesh()
    f = igl.boundary_facets(t)
    return TetMesh(v, t, f)

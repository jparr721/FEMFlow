from typing import Tuple, Union

import igl
import numpy as np

from femflow.numerics.geometry import per_face_normals


def load_mesh_file(
    filename: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not filename.lower().endswith(".mesh"):
        raise ValueError("Input filename is not a .mesh file")

    v, t, f = igl.read_mesh(filename)

    if v is None:
        raise ValueError("Mesh file contains no vertices")
    if f is None:
        raise ValueError("Mesh file contains no faces")
    if t is None:
        raise ValueError("Mesh file contains no tetrahedra")

    t = t.astype(np.int32)
    f = f.astype(np.int32)
    n = per_face_normals(v, f)

    return v, t, n, f


def load_obj_file(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not filename.lower().endswith(".obj"):
        raise ValueError("Input filename is not a .obj file")

    v, tc, n, f, _, _ = igl.read_obj(filename)

    if v is None:
        raise ValueError("Mesh file contains no vertices")
    if f is None:
        raise ValueError("Mesh file contains no faces")
    if n is None:
        n = per_face_normals(v, f)

    f = f.astype(np.int32)

    return v, tc, n, f

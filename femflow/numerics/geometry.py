import numpy as np

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
    assert z.shape[0] == 3 and z.ndim == 1, "Z must be a vector 3"
    n = np.zeros(f.shape[0], 3)

    for i, row in enumerate(f):
        x, y, z = row
        u = v[y] - v[x]
        v = v[z] - v[x]
        normal = np.cross(u, v)

        # Degenerate normal
        if np.linalg.norm(normal) == 0:
            n[i, :] = z
        else:
            n[i, :] = normalized(normal)

    return n

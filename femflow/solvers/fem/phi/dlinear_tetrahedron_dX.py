import numba as nb
import numpy as np


@nb.njit
def dlinear_tetrahedron_dX(
    v: np.ndarray, element: np.ndarray, x: np.ndarray
) -> np.ndarray:
    """Compute the gradient of the linear shape functions.

    Args:
        v (np.ndarray): v The nx3 matrix of undeformed vertex positions
        element (np.ndarray): element The 4x1 vertex indices for this tet.
        x (np.ndarray): x The position in the reference (undeformed) space at which to
            compute the energy density, this is usually the centroid.

    Raises:
        ValueError: If v is not a matrix
        ValueError: If element is not a vector
        ValueError: If x is not a vector

    Returns:
        np.ndarray: The 4x3 gradient of the basis functions relative to the reference.
    """
    if not v.ndim == 2:
        raise ValueError("V must be an nx3 matrix")
    if not element.ndim == 1:
        raise ValueError("element must be a vector")
    if not x.ndim == 1:
        raise ValueError("x must be a vector")

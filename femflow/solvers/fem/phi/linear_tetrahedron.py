import numba as nb
import numpy as np


@nb.njit
def linear_tetrahedron(v: np.ndarray, element: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Compute the linear shape functions for a single tetrahedron. This is basically
    just an indexing pattern for the generalized coordinates of the tetrahedral.

    We can compute the system of equations from a single tetrahedral quite trivially:

    (X - X0)   (dX1 dX2 dX3)   (phi1(X))
    (Y - Y0) = (dY1 dY2 dY3) * (phi2(X))
    (Z - Z0)   (dZ1 dZ2 dZ3)   (phi3(x))
               |     T     |

    Where the shape functions are directly computed by:
    T^-1(X - X0)
    Which we can solve via a linear solver.

    Finally, we construct phi via the linear combination of its parts:
    phi0(X) = 1 - phi1(X) - phi2(X) - phi3(X)

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
        np.ndarray: The 4x1 values of the basis functions
    """
    if not v.ndim == 2:
        raise ValueError("V must be an nx3 matrix")
    if not element.ndim == 1:
        raise ValueError("element must be a vector")
    if not x.ndim == 1:
        raise ValueError("x must be a vector")

    # Aquire our values in v
    x0, x1, x2, x3 = v[element]

    # Compute T as the difference between x0 to the other points
    T = np.zeros((3, 3))
    T[:, 0] = x1 - x0
    T[:, 1] = x2 - x0
    T[:, 2] = x3 - x0

    # Solve for the shape functions (right side of system)
    phi = np.zeros(4)
    phi[1:] = np.linalg.solve(T, (x - x0))

    # Finally, get the linear combination of phi0
    phi[0] = 1 - phi[1] - phi[2] - phi[3]
    return phi


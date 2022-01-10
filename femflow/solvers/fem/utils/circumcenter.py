import numba as nb
import numpy as np


def circumcenter(tet: np.ndarray) -> np.ndarray:
    """Computes the circumcenter for a single tetrahedral element.

    Args:
        tet (np.ndarray): tet The tet values (from the vertex positions)

    Raises:
        ValueError: If tet is not 4x3.

    Returns:
        np.ndarray: The circumcenter point in 3D euclidean space.
    """
    if tet.shape != (4, 3):
        raise ValueError("Tetrahedral must be a 4x3 matrix")

    A = np.zeros((3, 3))
    b = np.zeros(3)
    n0 = pow(np.linalg.norm(tet[0]), 2)

    for i in range(3):
        A[i] = tet[i + 1] - tet[0]
        b[i] = pow(np.linalg.norm(tet[i + 1]), 2) - n0

    return 0.5 * np.linalg.solve(A, b)


if __name__ == "__main__":
    t = np.array([[0, 0, 0], [0, 1, 0], [0, 1, 1], [1, 1, 1]])

    print(circumcenter(t))

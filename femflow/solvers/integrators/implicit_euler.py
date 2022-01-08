import numpy as np
from scipy.sparse import csr_matrix


def energy():
    pass


def force(f: np.ndarray, q2: np.ndarray, qdot2: np.ndarray):
    pass


def stiffness(K: np.ndarray, q2: np.ndarray, qdot2: np.ndarray):
    pass


def hessian(d_h: np.ndarray, x: np.ndarray):
    pass


def newtons_method(x0: np.ndarray, max_steps: int, epsilon: float = 2.1e-16) -> float:
    """Computes newtons method from an initial point

    Args:
        x0 (np.ndarray): x0 The initial point for newtons search
        max_steps (int): max_steps Max iterations for newtons method

    Returns:
        float: The update value for x0 (x1)
    """
    for i in range(max_steps):
        d_g = d_implicit_euler_cost(x0)
        if d_g < epsilon:
            return d_g


def d_implicit_euler_cost(dx: np.ndarray, x: np.ndarray) -> np.ndarray:
    pass


def implicit_euler(q: np.ndarray, qdot: np.ndarray, dt: float, mass: csr_matrix):
    pass

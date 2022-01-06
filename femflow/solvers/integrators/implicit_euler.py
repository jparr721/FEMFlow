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


def newtons_method(x0: np.ndarray, max_steps: int) -> float:
    return 0.0


def d_implicit_euler_cost(dx: np.ndarray, x: np.ndarray):
    pass


def implicit_euler(q: np.ndarray, qdot: np.ndarray, dt: float, mass: csr_matrix):
    pass

import numpy as np
from typing_extensions import Protocol

from femflow.numerics.geometry import grid


def primitive(k: float, t: float, pos: np.ndarray) -> float:
    two_pi = (2.0 * np.pi) / k
    x, y, z = pos

    return np.cos(two_pi * x) + np.cos(two_pi * y) + np.cos(two_pi * z) - t


def gyroid(k: float, t: float, pos: np.ndarray) -> float:
    two_pi = (2.0 * np.pi) / k
    x, y, z = pos

    return (
        np.sin(two_pi * x) * np.cos(two_pi * y)
        + np.sin(two_pi * y) * np.cos(two_pi * z)
        + np.sin(two_pi * z) * np.cos(two_pi * x)
        - t
    )


def diamond(k: float, t: float, pos: np.ndarray) -> float:
    two_pi = (2.0 * np.pi) / k
    x, y, z = pos

    return (
        np.sin(two_pi * x) * np.sin(two_pi * y) * np.sin(two_pi * z)
        + np.sin(two_pi * x) * np.cos(two_pi * y) * np.cos(two_pi * z)
        + np.cos(two_pi * x) * np.sin(two_pi * y) * np.cos(two_pi * z)
        + np.cos(two_pi * x) * np.cos(two_pi * y) * np.sin(two_pi * z)
        - t
    )


class ImplicitFn(Protocol):
    def __call__(self, k: float, t: float, pos: np.ndarray) -> float:
        ...


def generate_implicit_points(
    implicit_fn: ImplicitFn, k: float, t: float, res: int
) -> np.ndarray:
    g = grid(np.array((res, res, res)))
    inside = np.array([implicit_fn(k, t, row) for row in g])
    return g[inside > t]


def generate_cube_points(start: np.ndarray, res: int = 10) -> np.ndarray:
    x = np.linspace(*start, num=res)
    y = np.linspace(*start, num=res)
    z = np.linspace(*start, num=res)

    all_pts = []
    for layer in z:
        for row in y:
            for col in x:
                all_pts.append([layer, row, col])
    return np.array(all_pts, dtype=np.float64)


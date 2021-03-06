from typing import Tuple, Union

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
    implicit_fn: Union[ImplicitFn, str], k: float, t: float, res: int
) -> np.ndarray:
    if isinstance(implicit_fn, str):
        if implicit_fn == "gyroid":
            implicit_fn = gyroid
        elif implicit_fn == "diamond":
            implicit_fn = diamond
        elif implicit_fn == "primitive":
            implicit_fn = primitive
        else:
            raise ValueError("Invalid implicit function specified")

    g = grid(np.array((res, res, res)))
    inside = np.array([implicit_fn(k, t, row) for row in g])
    return g[inside > t]


def generate_cube_points(
    xb: Tuple[int, int], yb: Tuple[int, int], zb: Tuple[int, int], res: int = 10
) -> np.ndarray:
    x = np.linspace(*xb, num=res)
    y = np.linspace(*yb, num=res)
    z = np.linspace(*zb, num=res)

    all_pts = []
    for layer in z:
        for row in y:
            for col in x:
                all_pts.append([layer, row, col])
    return np.array(all_pts, dtype=np.float64)


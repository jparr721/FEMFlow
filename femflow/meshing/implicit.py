import numba as nb
import numpy as np

from femflow.numerics.geometry import grid


@nb.njit(parallel=True)
def gyroid(amplitude: float, resolution: int) -> np.ndarray:
    def _fn(amplitude: float, pos: np.ndarray) -> float:
        two_pi = (2.0 * np.pi) / amplitude
        x, y, z = pos
        return (
            np.sin(two_pi * x) * np.cos(two_pi * y)
            + np.sin(two_pi * y) * np.cos(two_pi * z)
            + np.sin(two_pi * z) * np.cos(two_pi * x)
        )

    gv = grid(np.array((resolution, resolution, resolution)))
    gf = np.zeros(gv.shape[0])
    for i in nb.prange(gv.shape[0]):
        row = gv[i]
        gf[i] = _fn(amplitude, row)

    return np.clip(gf, 0, 1)


@nb.njit(parallel=True)
def gyroid_2d(amplitude: float, resolution: int) -> np.ndarray:
    def _fn(amplitude: float, pos: np.ndarray) -> float:
        two_pi = (2.0 * np.pi) / amplitude
        x, y, z = pos
        return (
            np.sin(two_pi * x) * np.cos(two_pi * y)
            + np.sin(two_pi * y) * np.cos(two_pi * z)
            + np.sin(two_pi * z) * np.cos(two_pi * x)
        )

    gv = grid(np.array((resolution, resolution)))
    gf = np.zeros(gv.shape[0])
    for i in nb.prange(gv.shape[0]):
        row = gv[i]
        gf[i] = _fn(amplitude, row)

    return np.clip(gf, 0, 1)


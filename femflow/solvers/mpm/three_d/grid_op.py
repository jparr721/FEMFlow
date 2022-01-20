import numba as nb
import numpy as np


@nb.njit
def grid_op(
    grid_resolution: int,
    dt: float,
    gravity: float,
    grid_velocity: np.ndarray,
    grid_mass: np.ndarray,
):
    for i in range(grid_resolution + 1):
        for j in range(grid_resolution + 1):
            for k in range(grid_resolution + 1):
                if grid_mass[i, j, k] > 0:
                    grid_velocity[i, j, k] /= grid_mass[i, j, k]
                    grid_velocity[i, j, k][1] += dt * gravity
                    boundary = 0.05
                    x = i / grid_resolution
                    y = j / grid_resolution
                    z = k / grid_resolution
                    if (
                        x < boundary
                        or x > 1 - boundary
                        or y > 1 - boundary
                        or z > 1 - boundary
                    ):
                        grid_velocity[i, j, k] = 0
                    if y < boundary:
                        grid_velocity[i, j, k][1] = max(0.0, grid_velocity[i, j, k][1])
                    if z < boundary:
                        grid_velocity[i, j, k][2] = max(0.0, grid_velocity[i, j, k][2])

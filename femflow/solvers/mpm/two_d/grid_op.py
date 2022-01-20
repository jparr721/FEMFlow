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
    boundary = 0.05
    for i in range(grid_resolution + 1):
        for j in range(grid_resolution + 1):
            if grid_mass[i, j] > 0:
                grid_velocity[i, j] /= grid_mass[i, j]
                grid_velocity[i, j][1] += dt * gravity
                x = i / grid_resolution
                y = j / grid_resolution
                if x < boundary or x > 1 - boundary or y > 1 - boundary:
                    grid_velocity[i, j] = 0.0
                if y < boundary:
                    grid_velocity[i, j][1] = max(0.0, grid_velocity[i, j][1])

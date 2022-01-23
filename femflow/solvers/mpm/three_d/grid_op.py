import numba as nb
import numpy as np


@nb.njit
def grid_op(
    grid_resolution: int,
    dx: float,
    dt: float,
    gravity: float,
    grid_velocity: np.ndarray,
    grid_mass: np.ndarray,
):
    """Grid normalization and gravity application

    Args:
        grid_resolution (int): grid_resolution
        dt (float): dt
        gravity (float): gravity
        grid_velocity (np.ndarray): grid_velocity
        grid_mass (np.ndarray): grid_mass
    """
    v_allowed = dx * 0.9 / dt
    boundary = 3
    # boundary = 0.05
    for i in range(grid_resolution + 1):
        for j in range(grid_resolution + 1):
            for k in range(grid_resolution + 1):
                if grid_mass[i, j, k][0] > 0:
                    grid_velocity[i, j, k] /= grid_mass[i, j, k][0]
                    grid_velocity[i, j, k][1] += dt * gravity

                    grid_velocity[i, j, k] = np.clip(
                        grid_velocity[i, j, k], -v_allowed, v_allowed
                    )

                    # for d in range(3):
                    #     I = [i, j, k]
                    #     if I[d] < boundary and grid_velocity[i, j, k][d] < 0:
                    #         grid_velocity[i, j, k][d] = 0  # Boundary conditions
                    #     if (
                    #         I[d] >= grid_resolution - boundary
                    #         and grid_velocity[i, j, k][d] > 0
                    #     ):
                    #         grid_velocity[i, j, k][d] = 0
                    # x = i / grid_resolution
                    # y = j / grid_resolution
                    # z = k / grid_resolution
                I = [i, j, k]
                for d in range(3):
                    if I[d] < boundary and grid_velocity[i, j, k][d] < 0:
                        grid_velocity[i, j, k][d] = 0
                    if (
                        I[d] >= grid_resolution - boundary
                        and grid_velocity[i, j, k][d] > 0
                    ):
                        grid_velocity[i, j, k][d] = 0
                    # if (
                    #     x < boundary
                    #     or x > 1 - boundary
                    #     or y < boundary
                    #     or y > 1 - boundary
                    #     or z < boundary
                    #     or z > 1 - boundary
                    # ):
                    #     grid_velocity[i, j, k] = 0
                    # if y < boundary:
                    #     grid_velocity[i, j, k][1] = max(0.0, grid_velocity[i, j, k][1])
                    # if z < boundary:
                    #     grid_velocity[i, j, k][2] = max(0.0, grid_velocity[i, j, k][2])


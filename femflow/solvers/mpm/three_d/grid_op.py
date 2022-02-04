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
    """Grid normalization and gravity application, this also handles the collision
    scenario which, right now, is "STICKY", meaning the velocity is set to zero during
    collision scenarios.

    Args:
        grid_resolution (int): grid_resolution
        dt (float): dt
        gravity (float): gravity
        grid_velocity (np.ndarray): grid_velocity
        grid_mass (np.ndarray): grid_mass
    """
    v_allowed = dx * 0.9 / dt
    boundary = 1
    for i in range(grid_resolution + 1):
        for j in range(grid_resolution + 1):
            for k in range(grid_resolution + 1):
                if grid_mass[i, j, k][0] > 0:
                    grid_velocity[i, j, k] /= grid_mass[i, j, k][0]
                    grid_velocity[i, j, k][1] += dt * gravity

                    grid_velocity[i, j, k] = np.clip(
                        grid_velocity[i, j, k], -v_allowed, v_allowed
                    )

                I = [i, j, k]
                for d in range(3):
                    if I[d] < boundary and grid_velocity[i, j, k][d] < 0:
                        grid_velocity[i, j, k][d] = 0
                    if (
                        I[d] >= grid_resolution - boundary
                        and grid_velocity[i, j, k][d] > 0
                    ):
                        grid_velocity[i, j, k][d] = 0
    # points = []
    # normals = []
    # for d in [0, 1, 2]:
    #     point = [0, 0, 0]
    #     normal = [0, 0, 0]
    #     if d == 2:
    #         boundary /= 4
    #         boundary *= 2  # Thickness
    #     point[d] = boundary
    #     normal[d] = -1

    #     points.append(point)
    #     normals.append(normal)

    #     point[d] = grid_resolution - boundary
    #     normal[d] = 1

    #     points.append(point)
    #     normals.append(normal)

    # points = np.array(points)
    # normals = np.array(normals)
    # check_collision_points(points, normals, grid_resolution, dx, grid_velocity)


@nb.njit
def check_collision_points(
    points: np.ndarray,
    normals: np.ndarray,
    grid_resolution: int,
    dx: float,
    grid_velocity: np.ndarray,
):
    for point, normal in zip(points, normals):
        denom = np.sqrt(np.sum(np.square(normal)))
        normal = normal + (1.0 / denom)
        for i in range(grid_resolution + 1):
            for j in range(grid_resolution + 1):
                for k in range(grid_resolution + 1):
                    I = np.array([i, j, k])
                    offset = I * dx - point
                    if np.dot(offset, normal) < 0:
                        grid_velocity[i, j, k] = 0.0

import numpy as np

from .parameters import MPMParameters


def grid_velocity(params: MPMParameters, grid: np.ndarray):
    for i in range(params.grid_resolution):
        for j in range(params.grid_resolution):
            g = grid[i, j]

            if g[2] > 0:
                # Normalize the grid item by the mass
                g /= g[2]

                # Apply gravity
                g += params.dt * np.array((0, params.gravity, 0))

                boundary = 0.05

                # Node coordinates
                x = i / params.grid_resolution
                y = j / params.grid_resolution

                if x < boundary or x > 1 - boundary or y > 1 - boundary:
                    # Stop the particle at this grid cell
                    g = np.zeros(3)

                # Don't let the objects go under
                if y < boundary:
                    g[1] = max(0, g[1])

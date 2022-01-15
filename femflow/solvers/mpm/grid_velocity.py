import numpy as np
from loguru import logger

from .parameters import MPMParameters


def grid_velocity(params: MPMParameters, grid: np.ndarray):
    if params.debug:
        logger.info(f"Grid Stats Before: {grid.max()} {len(grid.nonzero())} {grid.min()}")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j][2] > 0:
                # Normalize the grid item by the mass
                grid[i, j] /= grid[i, j][2]

                # Apply gravity
                grid[i, j] += params.dt * np.array((0, params.gravity, 0))

                boundary = 0.05

                # Node coordinates
                x = i / params.grid_resolution
                y = j / params.grid_resolution

                if x < boundary or x > 1 - boundary or y > 1 - boundary:
                    # Stop the particle at this grid cell
                    grid[i, j] = np.zeros(3)

                # Don't let the objects go under
                if y < boundary:
                    grid[i, j][1] = max(0, grid[i, j][1])
    if params.debug:
        logger.info(f"Grid Stats After: {grid.max()} {len(grid.nonzero())} {grid.min()}")

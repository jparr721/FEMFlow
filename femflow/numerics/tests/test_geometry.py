import numpy as np

from ..geometry import grid


def test_compute_grid():
    c_grid = grid(np.array((3, 3, 3)))

    compare_grid = np.array(
        [
            [0, 0, 0],
            [0.5, 0, 0],
            [1, 0, 0],
            [0, 0.5, 0],
            [0.5, 0.5, 0],
            [1, 0.5, 0],
            [0, 1, 0],
            [0.5, 1, 0],
            [1, 1, 0],
            [0, 0, 0.5],
            [0.5, 0, 0.5],
            [1, 0, 0.5],
            [0, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [1, 0.5, 0.5],
            [0, 1, 0.5],
            [0.5, 1, 0.5],
            [1, 1, 0.5],
            [0, 0, 1],
            [0.5, 0, 1],
            [1, 0, 1],
            [0, 0.5, 1],
            [0.5, 0.5, 1],
            [1, 0.5, 1],
            [0, 1, 1],
            [0.5, 1, 1],
            [1, 1, 1],
        ]
    )

    assert np.array_equal(c_grid, compare_grid)

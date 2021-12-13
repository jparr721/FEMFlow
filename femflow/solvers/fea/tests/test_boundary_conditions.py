import numpy as np

from ..boundary_conditions import *

v = np.array(
    [
        [1.0, -1.0, -1.0],
        [1.0, -1.0, 1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [1.0, 1.0, 1.1],
        [-1.0, 1.0, 1.0],
        [-1.0, 1.0, -1.0],
    ]
)


def test_compute_top_bottom_plate_nodes():
    force_nodes, interior_nodes, fixed_nodes = top_bottom_plate_dirilect_conditions(v)


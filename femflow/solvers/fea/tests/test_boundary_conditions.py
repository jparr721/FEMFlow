# import numpy as np

# from ..boundary_conditions import *

# v = np.array(
#     [
#         [1.0, -1.0, -1.0],
#         [1.0, -1.0, 1.0],
#         [-1.0, -1.0, 1.0],
#         [-1.0, -1.0, -1.0],
#         [-0.5, -0.5, -0.5],
#         [1.0, 1.0, -1.0],
#         [1.0, 1.0, 1.1],
#         [-1.0, 1.0, 1.0],
#         [-1.0, 1.0, -1.0],
#     ]
# )


# def test_compute_top_bottom_plate_nodes():
#     force_nodes, interior_nodes, fixed_nodes = top_bottom_plate_dirilect_conditions(v)
#     force_nodes_comp = np.array([4, 5, 6, 7])
#     interior_nodes_comp = np.array([])
#     fixed_nodes_comp = np.array([0, 1, 2, 3])

#     assert (force_nodes == force_nodes_comp).all()
#     assert (interior_nodes == interior_nodes_comp).all()
#     assert (fixed_nodes == fixed_nodes_comp).all()


# def test_compute_top_bottom_plate_nodes_with_interior():
#     force_nodes, interior_nodes, fixed_nodes = top_bottom_plate_dirilect_conditions(v)
#     force_nodes_comp = np.array([5, 6, 7, 8])
#     interior_nodes_comp = np.array([4])
#     fixed_nodes_comp = np.array([0, 1, 2, 3])

#     assert (force_nodes == force_nodes_comp).all()
#     assert (interior_nodes == interior_nodes_comp).all()
#     assert (fixed_nodes == fixed_nodes_comp).all()

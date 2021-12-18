from typing import List, Tuple

import numba as nb
import numpy as np

BoundaryConditions = dict[int, np.ndarray]


def __find(axis: int, v: np.ndarray, boundary: float, epsilon: float) -> List[int]:
    return [i for i, node in enumerate(v) if np.isclose(node[axis], boundary, atol=epsilon)]


def find_max_surface_nodes(axes: List[int], v: np.ndarray, epsilon: float = 0.0) -> np.ndarray:
    axis_maxes = v.max(axis=0)
    return np.concatenate([__find(axis, v, axis_maxes[axis], epsilon) for axis in axes])


def find_min_surface_nodes(axes: List[int], v: np.ndarray, epsilon: float = 0.0) -> np.ndarray:
    axis_mins = v.min(axis=0)
    return np.concatenate([__find(axis, v, axis_mins[axis], epsilon) for axis in axes])


def top_bottom_plate_dirilect_conditions(v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Finds the boundary conditions for the top-bottom plates of a uniaxial compression

    Args:
        v (np.ndarray): The n x 3 vertex mesh

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: The force nodes, interior nodes, and fixed nodes
    """
    # Always the y-axis
    axis = [1]

    # 1% of the min/max for wiggle room (for really _really_ close nodes)
    epsilon = 0.01
    force_nodes = find_max_surface_nodes(axis, v, epsilon)
    force_nodes.sort()
    fixed_nodes = find_min_surface_nodes(axis, v, epsilon)
    fixed_nodes.sort()
    interior_nodes = np.array([i for i in range(len(v)) if i not in force_nodes and i not in fixed_nodes])
    interior_nodes.sort()
    return force_nodes, interior_nodes, fixed_nodes


@nb.njit
def basic_dirilecht_boundary_conditions(
    force: np.ndarray, force_nodes: np.ndarray, active_nodes: np.ndarray
) -> BoundaryConditions:
    """Generates the basic dirilect boundary conditions for a given mesh

    Args:
        force (np.ndarray): The force to apply at each node, (3 x 1)
        force_nodes (np.ndarray): The nodes to apply force to (indexes directly into v)
        active_nodes (np.ndarray): The nodes with non-fixed degrees of freedom

    Returns:
        BoundaryConditions: The boundary conditions for this mesh, can be indexed directly by v.
    """
    zero = np.zeros(3)

    boundary_conditions = dict()
    for node in force_nodes:
        boundary_conditions[node] = force
    for node in active_nodes:
        boundary_conditions[node] = zero
    return boundary_conditions

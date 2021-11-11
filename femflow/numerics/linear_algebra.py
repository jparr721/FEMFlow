from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R


def normalized(a: np.array):
    return a / np.linalg.norm(a)


def _rotate(u, v, center, neighbor, theta):
    cross = normalized(np.cross(u, v))
    ht = theta / 2
    q = np.array([cross[0] * np.sin(ht), cross[1] * np.sin(ht), cross[2] * np.sin(ht), np.cos(ht)])
    rot_matrix = R.from_quat(q).as_matrix()

    origin = neighbor - center
    normalized_origin = normalized(origin)
    neighbor_center_distance = np.linalg.norm(origin)

    rotated_neighbor = np.matmul(rot_matrix, normalized_origin)

    # Since we translated to the origin, we need to translate back to our original position
    rotated_neighbor *= neighbor_center_distance
    rotated_neighbor += center

    return rotated_neighbor


def three_node_planar_rotation(theta, center: np.array, left: np.array, right: np.array) -> Tuple[np.array, np.array]:
    u = normalized(left - center)
    v = normalized(right - center)
    return (_rotate(u, v, center, left, theta), _rotate(v, u, center, right, theta))


def angle_between_vectors(a: np.array, b: np.array) -> float:
    return np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))


def joint_point_point_angle_rad(root: np.array, n1: np.array, n2: np.array) -> float:
    n1d = normalized(n1 - root)
    n2d = normalized(n2 - root)
    return angle_between_vectors(n1d, n2d)


def distance(a: np.array, b: np.array) -> float:
    return np.linalg.norm(a - b)


from typing import Tuple

import numpy as np
import scipy as sp
from scipy.spatial.transform import Rotation as R


def normalized(a: np.array):
    return a / np.linalg.norm(a)


def three_node_planar_rotation(theta, center: np.array, left: np.array, right: np.array) -> Tuple[np.array, np.array]:
    u = normalized(left - center)
    v = normalized(right - center)

    def rotate(u, v, center, neighbor, theta):
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

    return (rotate(u, v, center, left, theta), rotate(v, u, center, right, theta))


def angle_between_vectors(a: np.array, b: np.array) -> float:
    return np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))


def joint_point_point_angle_rad(root: np.array, n1: np.array, n2: np.array) -> float:
    n1d = normalized(n1 - root)
    n2d = normalized(n2 - root)
    return angle_between_vectors(n1d, n2d)


def distance(a: np.array, b: np.array) -> float:
    return np.linalg.norm(a - b)


def angle_axis(angle, axis):
    return sp.linalg.expm(np.cross(np.eye(3), normalized(axis) * angle))


def quaternion_multiply(a, b) -> np.ndarray:
    a_w, a_x, a_y, a_z = a
    b_w, b_x, b_y, b_z = b
    return np.array(
        [
            a_w * b_w - a_x * b_x - a_y * b_y - a_z * b_z,
            a_w * b_x + a_x * b_w + a_y * b_z - a_z * b_y,
            a_w * b_y + a_y * b_w + a_z * b_x - a_x * b_z,
            a_w * b_z + a_z * b_w + a_x * b_y - a_y * b_x,
        ],
        dtype=np.float64,
    )


def rotation_as_quat(rot: R) -> np.ndarray:
    """Scipy rotations are not in the proper quaternion order

    Args:
        rot (R): The scipy rotation in the form: x, y, z, w

    Returns:
        np.ndarray: The w, x, y, z ordered quaternion
    """
    x, y, z, w = rot.as_quat()
    return np.array([w, x, y, z])

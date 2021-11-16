import numpy as np
import pyrr
from loguru import logger
from numerics.linear_algebra import angle_axis, normalized, quaternion_multiply, rotation_as_quat
from scipy.spatial.transform import Rotation as R

# class Camera(object):
#     def __init__(self, width, height):
#         self.width = width
#         self.height = height

#         self.base_zoom = 1.0
#         self.zoom = 1.0
#         self.view_angle = 45.0
#         self.near = 1.0
#         self.far = 100.0
#         self.base_translation = np.zeros(3)
#         self.translation = np.zeros(3)
#         self.eye = np.array([0, 0, 5])
#         self.center = np.zeros(3)
#         self.up = np.array([0, 1, 0])

#         self.view_matrix = np.eye(4, dtype=np.float32)
#         self.projection_matrix = np.eye(4, dtype=np.float32)
#         self.normal_matrix = np.eye(4, dtype=np.float32)

#         self.trackball_angle = R.identity()

#     def update(self):
#         self.view_matrix = self.look_at(self.eye, self.center, self.up)
#         t = self.base_translation + self.translation
#         q_scale = self.trackball_angle.as_matrix() * (self.zoom * self.base_zoom)
#         t *= q_scale.diagonal()
#         q_mat = np.eye(4)
#         q_mat[:3, :3] = q_scale
#         q_mat[:3, 3] = t
#         self.view_matrix *= q_mat
#         self.normal_matrix = np.linalg.inv(self.view_matrix).T

#         fh = np.tan(self.view_angle / 360.0 * np.pi) * self.near
#         fw = fh * self.width / self.height
#         self.projection_matrix = self.frustum(-fw, fw, -fh, fh, self.near, self.far)

#     @staticmethod
#     def frustum(left, right, bottom, top, near_val, far_val):
#         P = np.zeros((4, 4), order="F")
#         P[0, 0] = (2.0 * near_val) / (right - left)
#         P[1, 1] = (2.0 * near_val) / (top - bottom)
#         P[0, 2] = (right + left) / (right - left)
#         P[1, 2] = (top + bottom) / (top - bottom)
#         P[2, 2] = -(far_val + near_val) / (far_val - near_val)
#         P[3, 2] = -1.0
#         P[2, 3] = -(2.0 * far_val * near_val) / (far_val - near_val)

#         return P

#     @staticmethod
#     def look_at(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
#         f = normalized(center - eye)
#         s = normalized(np.cross(f, up))
#         u = np.cross(s, f)

#         ret = np.eye(4)

#         ret[0, 0] = s[0]
#         ret[0, 1] = s[1]
#         ret[0, 2] = s[2]
#         ret[1, 0] = u[0]
#         ret[1, 1] = u[1]
#         ret[1, 2] = u[2]
#         ret[2, 0] = -f[0]
#         ret[2, 1] = -f[1]
#         ret[2, 2] = -f[2]
#         ret[0, 3] = -np.dot(s, eye)
#         ret[1, 3] = -np.dot(u, eye)
#         ret[2, 3] = np.dot(f, eye)

#         return ret

#     @staticmethod
#     def two_axis_valudator_fixed_up(
#         w: int, h: int, speed: float, down_quat: np.ndarray, down_x: int, down_y: int, mouse_x: int, mouse_y: int
#     ) -> np.ndarray:
#         axis = np.array([0, 1, 0])

#         aa = angle_axis(np.pi * (mouse_x - down_x) / w * speed / 2.0, normalized(axis))
#         x_axis_rot = rotation_as_quat(R.from_matrix(aa))

#         quat = normalized(quaternion_multiply(down_quat, x_axis_rot))

#         axis = np.array([1, 0, 0])
#         aa = angle_axis(np.pi * (mouse_y - down_y) / h * speed / 2.0, normalized(axis))
#         y_axis_rot = rotation_as_quat(R.from_matrix(aa))

#         quat = quaternion_multiply(y_axis_rot, quat)
#         quat = normalized(quat)

#         return quat


class Camera(object):
    def __init__(
        self,
        *,
        near_plane=0.1,
        far_plane=1000.0,
        rotation_sensitivity=0.01,
        pan_sensitivity=0.01,
        zoom_sensitivity=0.1,
        min_fov=10.0,
        max_fov=120.0,
        fov=45.0,
        min_radius=0.1,
        max_radius=1000.0,
        aspect_ratio=(4.0 / 3.0),
    ):
        """Basic camera with pan, zoom, and rotate behavior, controllable by any input system

        Args:
            near_plane (float, optional): Near plane of camera view. Defaults to 0.1.
            far_plane (float, optional): Far plane of camera view. Defaults to 100.0.
            rotation_sensitivity (float, optional): Rotation sensitivity. Defaults to 0.01.
            pan_sensitivity (float, optional): Pan sensitivity. Defaults to 0.01.
            zoom_sensitivity (float, optional): Zoom sensitivity. Defaults to 0.01.
            min_fov (float, optional): Min camera FOV. Defaults to 10.0.
            max_fov (float, optional): Max camera FOV. Defaults to 120.0.
            fov (float, optional): FOV Vvlue. Defaults to 65.0.
            min_radius (float, optional): Min camera radius. Defaults to 0.1.
            max_radius (int, optional): Max camera radius. Defaults to 1000.
            aspect_ratio (tuple, optional): Camera aspect ratio. Defaults to (4.0 / 3.0).
        """

        self.zooming = False
        self.panning = False
        self.rotating = False

        self._view_matrix = np.eye(4)
        self._projection_matrix = np.eye(4)

        self.eye = np.array([0, 0, -5])
        self.center = np.zeros(3)
        self.up = np.array([0, 1, 0])

        self.near_plane = near_plane
        self.far_plane = far_plane

        self.rotation_sensitivity = rotation_sensitivity
        self.pan_sensitivity = pan_sensitivity
        self.zoom_sensitivity = zoom_sensitivity

        self.min_fov = min_fov
        self.max_fov = max_fov
        self.fov = fov

        self.min_radius = min_radius
        self.max_radius = max_radius

        self.aspect_ratio = aspect_ratio
        self.displacement = np.zeros(3)

        self.r = 1.0
        self.theta = np.pi / 2
        self.phi = np.pi / 2

    @property
    def view_matrix(self):
        self._compile()
        return self._view_matrix

    @property
    def projection_matrix(self):
        self._compile()
        return self._projection_matrix

    @property
    def left_direction(self):
        self._compile()
        dir = normalized(self.center - self.eye)
        return normalized(np.cross(dir, self.up))

    @property
    def down_direction(self):
        self._compile()
        return -self.up

    def resize(self, width: int, height: int):
        self.width = width
        self.height = height
        self.set_perspective()
        self._compile()

    def zoom(self, dr: float):
        if self.r + dr > self.max_radius:
            self.r = self.max_radius
            self._compile()
            return

        if self.r + dr < self.min_radius:
            self.r = self.min_radius
            self._compile()
            return

        self.r += dr
        self._compile()

    def pan(self, du: float, dv: float):
        u_dir = self.left_direction
        v_dir = self.down_direction

        u_disp = (du * self.pan_sensitivity) * u_dir
        v_disp = (dv * self.pan_sensitivity) * v_dir
        pan_disp = u_disp + v_disp

        self.displacement += pan_disp
        self._compile()

    def rotate(self, du: float, dv: float):
        self.theta -= du * self.rotation_sensitivity
        self.phi += dv * self.rotation_sensitivity
        self._compile()

    def set_perspective(self):
        if self.fov > self.max_fov:
            self.fov = self.max_fov
        elif self.fov < self.min_fov:
            self.fov = self.min_fov

        self._projection_matrix = pyrr.matrix44.create_perspective_projection_matrix(
            self.fov, self.aspect_ratio, self.near_plane, self.far_plane
        )
        print(self._projection_matrix)

    @staticmethod
    def look_at(eye: np.ndarray, at: np.ndarray, up: np.ndarray):
        return pyrr.matrix44.create_look_at(eye, at, up)

    @staticmethod
    def _spherical_to_cartesian(r: float, theta: float, phi: float):
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        vec = np.zeros(3)
        vec[0] = r * (cos_theta * sin_phi)
        vec[1] = r * cos_phi
        vec[2] = r * (sin_theta * sin_phi)
        return vec

    @staticmethod
    def _sperical_to_cartesian_dPhi(r: float, theta: float, phi: float):
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        vec = np.zeros(3)
        vec[0] = r * (cos_phi * cos_theta)
        vec[1] = -r * sin_phi
        vec[2] = r * (cos_phi * sin_theta)
        return vec

    def _compile(self):
        self.center = np.zeros(3)
        self.eye = self._spherical_to_cartesian(self.r, self.theta, self.phi)
        self.up = normalized(self._sperical_to_cartesian_dPhi(self.r, self.theta, self.phi))

        # --------------------------------------------------------------------------------
        # Invert the up direction (since the spherical coordinates have phi
        # increasing downwards. Therefore we would like to have the (vector)
        # direction of the derivative inversed.
        # --------------------------------------------------------------------------------
        self.up *= -1.0
        self.center += self.displacement
        self.eye += self.displacement

        self._view_matrix = self.look_at(self.eye, self.center, self.up)

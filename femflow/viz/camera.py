import numpy as np
from numerics.linear_algebra import normalized


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
        fov=65.0,
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

        self.eye = np.array([0, 0, 0])
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

    def reset(self):
        self.eye = np.array([0, 0, 0])
        self.center = np.zeros(3)
        self.up = np.array([0, 1, 0])
        self.r = 1.0
        self.theta = np.pi / 2
        self.phi = np.pi / 2
        self.displacement = np.zeros(3)

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

    def snap_to_mesh(self, max_z: float, midx: float, midy: float):
        self.r = 4 * max_z
        self.displacement = np.array([0, 0, -2 * max_z])

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
        """Set the projection matrix from the persepctive values. Stolen from pyrr.
        """
        if self.fov > self.max_fov:
            self.fov = self.max_fov
        elif self.fov < self.min_fov:
            self.fov = self.min_fov

        ymax = self.near_plane * np.tan(self.fov * np.pi / 360.0)
        xmax = ymax * self.aspect_ratio

        left = -xmax
        right = xmax
        bottom = -ymax
        top = ymax

        A = (right + left) / (right - left)
        B = (top + bottom) / (top - bottom)
        C = -(self.far_plane + self.near_plane) / (self.far_plane - self.near_plane)
        D = -2.0 * self.far_plane * self.near_plane / (self.far_plane - self.near_plane)
        E = 2.0 * self.near_plane / (right - left)
        F = 2.0 * self.near_plane / (top - bottom)

        self._projection_matrix = np.array(
            ((E, 0.0, 0.0, 0.0), (0.0, F, 0.0, 0.0), (A, B, C, -1.0), (0.0, 0.0, D, 0.0))
        )

    @staticmethod
    def look_at(eye: np.ndarray, at: np.ndarray, up: np.ndarray) -> np.ndarray:
        """Create look at matrix, stolen from pyrr so I can modify it.

        Args:
            eye (np.ndarray): The eye position
            at (np.ndarray): What we're looking at
            up (np.ndarray): The up direction

        Returns:
            np.ndarray: The look at matrix
        """
        eye = np.asarray(eye)
        target = np.asarray(at)
        up = np.asarray(up)

        forward = normalized(target - eye)
        side = normalized(np.cross(forward, up))
        up = normalized(np.cross(side, forward))

        return np.array(
            (
                (side[0], up[0], -forward[0], 0.0),
                (side[1], up[1], -forward[1], 0.0),
                (side[2], up[2], -forward[2], 0.0),
                (-np.dot(side, eye), -np.dot(up, eye), np.dot(forward, eye), 1.0),
            ),
        )

    @staticmethod
    def spherical_to_cartesian(r: float, theta: float, phi: float) -> np.ndarray:
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
    def sperical_to_cartesian_dPhi(r: float, theta: float, phi: float) -> np.ndarray:
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
        self.eye = self.spherical_to_cartesian(self.r, self.theta, self.phi)
        self.up = normalized(self.sperical_to_cartesian_dPhi(self.r, self.theta, self.phi))

        # --------------------------------------------------------------------------------
        # Invert the up direction (since the spherical coordinates have phi
        # increasing downwards. Therefore we would like to have the (vector)
        # direction of the derivative inversed.
        # --------------------------------------------------------------------------------
        self.up *= -1.0
        self.center += self.displacement
        self.eye += self.displacement

        self._view_matrix = self.look_at(self.eye, self.center, self.up)

import numpy as np
from lib.math.linear_algebra import normalized


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
        aspect_ratio=(4.0 / 3.0)
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

        self.eye = np.array([0, 0, 5])
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
    def view_direction(self):
        self._compile()
        return normalized(self.center - self.eye)

    @property
    def right_direction(self):
        self._compile()
        dir = normalized(self.center - self.eye)
        return normalized(np.cross(self.up, dir))

    @property
    def left_direction(self):
        self._compile()
        dir = normalized(self.center - self.eye)
        return normalized(np.cross(dir, self.up))

    @property
    def up_direction(self):
        self._compile()
        return self.up

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

        aspect_ratio = self.width / self.height
        y_max = self.near_plane * np.tan(self.fov * np.pi) / 360
        x_max = y_max * aspect_ratio
        self._set_frustum(-x_max, x_max, -y_max, y_max)

    def _set_frustum(self, left: float, right: float, bottom: float, top: float):
        t1 = 2.0 * self.near_plane
        t2 = right - left
        t3 = top - bottom
        t4 = self.far_plane - self.near_plane

        self._projection_matrix = np.array(
            [
                [t1 / t2, 0, 0, 0],
                [0, t1 / t3, 0, 0],
                [(right + left) / t2, (top + bottom) / t3, -(self.far_plane + self.near_plane) / t4, -1.0],
                [0, 0, (-t1 * self.far_plane) / t4, 0],
            ]
        )

    @staticmethod
    def look_at(eye: np.ndarray, at: np.ndarray, up: np.ndarray):
        matrix = np.eye(4)
        f = normalized(at - eye)
        s = normalized(np.cross(f, up))
        u = np.cross(s, f)

        matrix[0, 0] = s[0]
        matrix[0, 1] = s[1]
        matrix[0, 2] = s[2]
        matrix[0, 3] = -np.dot(s, eye)

        matrix[1, 0] = u[0]
        matrix[1, 1] = u[1]
        matrix[1, 2] = u[2]
        matrix[1, 3] = -np.dot(u, eye)

        matrix[2, 0] = -f[0]
        matrix[2, 1] = -f[1]
        matrix[2, 2] = -f[2]
        matrix[2, 3] = np.dot(f, eye)

        return matrix

    @staticmethod
    def _spherical_to_cartesian(r: float, theta: float, phi: float):
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        return np.array([r * (cos_theta * sin_phi), r * cos_phi, r * (sin_theta * sin_phi)])

    @staticmethod
    def _sperical_to_cartesian_dPhi(r: float, theta: float, phi: float):
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        return np.array([r * (cos_phi * cos_theta), -r * sin_phi, r * (cos_phi * sin_theta)])

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

from typing import Tuple

import igl
import numpy as np
import wildmeshing as wm
from loguru import logger
from numerics.geometry import per_face_normals
from PIL import Image


class Mesh(object):
    def __init__(
        self,
        data: np.ndarray,
        *,
        faces=None,
        tetrahedra=None,
        colors=None,
        normals=None,
        textures=None,
        tetrahedralize=False,
    ):
        self.vertices = None
        self.faces = faces
        self.tetrahedra = tetrahedra
        self.colors = colors
        self.normals = normals
        self.textures = textures
        self.tetrahedralize = tetrahedralize

        # TODO(@jparr721) - Make this constructor less stupid
        if type(data) == str:
            self._init_from_file(data)
        elif faces is not None and tetrahedra is None:
            self._init_from_surface_mesh(data, faces)
        elif tetrahedra is not None and faces is None:
            self._init_from_volume_mesh(data, tetrahedra)
        else:
            self.vertices = self._deflate(data)
            self.faces = self._deflate(faces)
            self.tetrahedra = self._deflate(tetrahedra)
        if self.colors is None:
            self.colors = np.tile(np.array([0.96, 0.88, 0.44]), len(self.vertices / 3)).astype(np.float32)
        if self.textures is None:
            self.textures = np.tile(np.array([0.96, 0.88, 0.74]), (len(self.vertices / 3), 1)).astype(np.float32)
            self.textures_u = 3
            self.textures_v = 3
            # self.textures_u, self.textures_v =
        elif type(self.textures) == str:
            logger.info(f"Loading texture from file: {self.textures}")
            i = Image.open(self.textures)
            self.textures = np.array(i.getdata(), dtype=np.uint16)
            self.textures_u, self.textures_v = i.width, i.height

        self.rest_positions = self.vertices

    def update(self, displacements: np.array):
        self.vertices = self.rest_positions + displacements

    def axis_max(self, axis: int) -> float:
        mv = -1
        for i in range(axis, len(self.vertices), 3):
            mv = max(mv, self.vertices[i])
        return mv

    def axis_min(self, axis: int) -> float:
        mv = 1e10
        for i in range(axis, len(self.vertices), 3):
            mv = min(mv, self.vertices[i])
        return mv

    def unroll_to_igl_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._inflate(self.vertices, 3), self._inflate(self.faces, 3).astype(np.int32)

    def _compute_normals(self, v: np.ndarray, f: np.ndarray) -> np.ndarray:
        self.normals = -self._deflate(per_face_normals(v, f)).astype(np.float32)

    def _init_from_file(self, filename: str):
        logger.info(f"Loading mesh from file: {filename}")
        V, F = igl.read_triangle_mesh(filename)
        self._init_from_surface_mesh(V, F)

    def _init_from_surface_mesh(self, V: np.ndarray, F: np.ndarray):
        if len(F.shape) > 1 and F.shape[1] == 4:
            F = igl.boundary_facets(F)

        if self.tetrahedralize:
            # TODO(@jparr721) - Make sure this works.
            VT, T = self._tetrahedralize(V, F)
            self.tetrahedra = self._deflate(T)
            self.vertices = self._deflate(VT)
            F = igl.boundary_facets(T)
            self.faces = self._deflate(F)
        else:
            self.vertices = self._deflate(V)
            self.faces = self._deflate(F)

        self._compute_normals(V, F)
        self.faces = self.faces.astype(np.uint32)
        self.vertices = self.vertices.astype(np.float32)

    def _init_from_volume_mesh(self, V: np.ndarray, T: np.ndarray):
        self.vertices = self._deflate(V)
        F = igl.boundary_facets(T)
        self.faces = self._deflate(F)
        self._compute_normals(V, F)
        self.tetrahedra = T

    def _deflate(self, matrix: np.ndarray) -> np.ndarray:
        """Flatten a matrix ino a 1d vector

        Args:
            matrix (np.ndarray): Input matrix

        Returns:
            np.ndarray: Flattened vector
        """
        assert matrix.ndim <= 2, "Too many matrix dimensions!"
        if matrix.ndim == 1:
            return matrix
        return matrix.reshape(-1)

    def _inflate(self, vector: np.ndarray, cols: int) -> np.ndarray:
        """Inflates a vector into an n by m matrix

        Args:
            vector (np.ndarray): The input vector
            cols (int): The matrix cols

        Returns:
            np.ndarray: The output matrix
        """
        assert vector.ndim == 1, "Input must be a vector"
        return vector.reshape(vector.shape[0] // cols, cols)

    def _tetrahedralize(self, V: np.ndarray, F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        tetrahedralizer = wm.Tetrahedralizer(stop_quality=1000)
        tetrahedralizer.set_mesh(V, F)
        tetrahedralizer.tetrahedralize()
        return tetrahedralizer.get_tet_mesh()

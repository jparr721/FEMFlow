from typing import List, Tuple, Union

import igl
import numpy as np
import wildmeshing as wm
from loguru import logger


class Mesh(object):
    def __init__(self, data: np.ndarray, *, faces=None, tetrahedra=None, colors=None, tetrahedralize=False):
        self.vertices = None
        self.faces = faces
        self.tetrahedra = tetrahedra
        self.colors = colors
        self.tetrahedralize = tetrahedralize

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
            self.colors = np.tile(np.random.rand(3), len(self.vertices / 3)).astype(np.float32)
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

    def _init_from_file(self, filename: str):
        logger.info(f"Loading mesh from file: {filename}")
        V, F = igl.read_triangle_mesh(filename)
        self._init_from_surface_mesh(V, F)

    def _init_from_surface_mesh(self, V: np.ndarray, F: np.ndarray):
        if len(F.shape) > 1 and F.shape[1] == 4:
            F = igl.boundary_facets(F)

        if self.tetrahedralize:
            VT, T = self._tetrahedralize(V, F)
            self.tetrahedra = self._deflate(T)
            self.vertices = self._deflate(VT)
            self.faces = self._deflate(igl.boundary_facets(VT))
        else:
            self.vertices = self._deflate(V)
            self.faces = self._deflate(F)

        self.faces = self.faces.astype(np.uint32)
        self.vertices = self.vertices.astype(np.float32)

    def _init_from_volume_mesh(self, V: np.ndarray, T: np.ndarray):
        self.vertices = self._deflate(V)
        self.faces = self._deflate(igl.boundary_facets(T))
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

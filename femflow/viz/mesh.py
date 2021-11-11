from typing import List, Union

import igl
import numpy as np
import wildmeshing as wm
from loguru import logger
from OpenGL.GL import *


class Mesh(object):
    def __init__(
        self,
        data: Union[str, List[float], List[List[float]], np.array],
        *,
        surface=None,
        volumes=None,
        tetrahedralize=False,
    ):
        self.DEFAULT_MESH_COLOR = np.array([1, 0, 0, 1], dtype=np.float32)

        self.faces = None
        self.vertices = None
        self.tetrahedra = None
        self.tetrahedralize = tetrahedralize

        if type(data) == str:
            self._init_from_file(data)
        elif surface is not None and volumes is None:
            self._init_from_surface_mesh(data, surface)
        elif volumes is not None and surface is None:
            self._init_from_volume_mesh(data, volumes)

        self.colors = np.tile(self.DEFAULT_MESH_COLOR, len(self.vertices / 3))
        self.rest_positions = self.vertices

    def update(self, displacements: np.array):
        self.vertices = self.rest_positions + displacements

    def _init_from_file(self, filename: str):
        logger.info(f"Loading mesh from file: {filename}")
        V, F = igl.read_triangle_mesh(filename)

        if len(F.shape) > 1 and F.shape[1] == 4:
            F = igl.boundary_facets(F)

        if self.tetrahedralize:
            VT, T = self._tetrahedralize(V, F)
            self.tetrahedra = self._flatten(T)
            self.vertices = self._flatten(VT)
            self.faces = self._flatten(igl.boundary_facets(VT))
        else:
            self.vertices = self._flatten(V)
            self.faces = self._flatten(F)

    def _init_from_surface_mesh(
        self, V: Union[List[float], List[List[float]], np.array], F: Union[List[float], List[List[float]], np.array]
    ):
        if len(F.shape) > 1 and F.shape[1] == 4:
            F = igl.boundary_facets(F)

        if self.tetrahedralize:
            VT, T = self._tetrahedralize(V, F)
            self.tetrahedra = self._flatten(T)
            self.vertices = self._flatten(VT)
            self.faces = self._flatten(igl.boundary_facets(VT))
        else:
            self.vertices = self._flatten(V)
            self.faces = self._flatten(F)

    def _init_from_volume_mesh(
        self, V: Union[List[float], List[List[float]], np.array], T: Union[List[float], List[List[float]], np.array]
    ):
        self.vertices = self._flatten(V)
        self.faces = self._flatten(igl.boundary_facets(T))
        self.tetrahedra = T

    def _flatten(self, matrix):
        if len(matrix.shape) > 1 and matrix.shape[1] == 1:
            return matrix
        return matrix.reshape(-1)

    def _tetrahedralize(self, V: np.array, F: np.array):
        tetrahedralizer = wm.Tetrahedralizer(stop_quality=1000)
        tetrahedralizer.set_mesh(V, F)
        tetrahedralizer.tetrahedralize()
        return tetrahedralizer.get_tet_mesh()

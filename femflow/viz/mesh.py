from typing import List, Tuple

import igl
import numpy as np
import wildmeshing as wm
from loguru import logger
from meshing.loader import load_mesh_file, load_obj_file
from PIL import Image


class Texture(object):
    def __init__(self, data: np.ndarray, u: int, v: int):
        self.data = data
        self.u = u
        self.v = v

    @property
    def size(self):
        return self.data.size

    @staticmethod
    def from_file(image_file: str):
        image = Image.open(image_file)
        data = np.array(image.transpose(Image.FLIP_TOP_BOTTOM).getdata(), dtype=np.uint8)
        u, v = image.width, image.height
        return Texture(data, u, v)


class Mesh(object):
    def __init__(
        self,
        vertices: np.ndarray = np.array([]),
        faces: np.ndarray = np.array([]),
        tetrahedra: np.ndarray = np.array([]),
        colors: np.ndarray = np.array([]),
        normals: np.ndarray = np.array([]),
        textures: Texture = Texture(np.array([]), 0, 0),
    ):
        self.vertices = self.as_vector(vertices).astype(np.float32)
        self.faces = self.as_vector(faces).astype(np.uint32)
        self.tetrahedra = self.as_vector(tetrahedra)
        self.normals = self.as_vector(normals)
        self.colors = self.as_vector(colors)
        self.textures = textures

        if self.textures.size == 0 and self.colors.size == 0:
            self.colors = np.tile(np.random.rand(3), len(self.vertices.data) // 3).astype(np.float32)

    @staticmethod
    def from_file(filename: str):
        if filename.lower().endswith(".obj"):
            v, n, f = load_obj_file(filename)
            return Mesh(vertices=v, normals=n, faces=f)
        if filename.lower().endswith(".mesh"):
            v, t, n, f = load_mesh_file(filename)
            return Mesh(vertices=v, tetrahedra=t, normals=n, faces=f)

    def tetrahedralize(self):
        raise NotImplementedError("Not yet implemented")

    def as_vector(self, array: np.ndarray):
        assert array.ndim < 3, "Must be at most 2D"
        if array.ndim == 2:
            return array.reshape(-1)
        else:
            return array

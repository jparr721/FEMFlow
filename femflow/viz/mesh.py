import copy

import numpy as np
from femflow.meshing.loader import load_mesh_file, load_obj_file
from femflow.numerics.geometry import per_face_normals, tetrahedralize_surface_mesh
from loguru import logger
from PIL import Image


class Texture(object):
    def __init__(self, data: np.ndarray, tc: np.ndarray, u: int, v: int):
        self.data = data
        self.tc = tc
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
        return Texture(data, np.array([]), u, v)


class Mesh(object):
    def __init__(
        self,
        vertices: np.ndarray = np.array([]),
        faces: np.ndarray = np.array([]),
        tetrahedra: np.ndarray = np.array([]),
        colors: np.ndarray = np.array([]),
        normals: np.ndarray = np.array([]),
        textures: Texture = Texture(np.array([]), np.array([]), 0, 0),
    ):
        self.vertices = self.as_vector(vertices).astype(np.float32)
        self.faces = self.as_vector(faces).astype(np.uint32)
        self.tetrahedra = self.as_vector(tetrahedra)
        self.normals = self.as_vector(normals)
        self.colors = self.as_vector(colors)
        self.textures = textures

        self.world_coordinates = copy.deepcopy(self.vertices)

        if self.textures.size == 0 and self.colors.size == 0:
            self.colors = np.tile(np.random.rand(3), len(self.vertices.data) // 3).astype(np.float32)
            logger.info(f"Random Color: {self.colors[:3]}")

    @staticmethod
    def from_file(filename: str):
        if filename.lower().endswith(".obj"):
            v, tc, n, f = load_obj_file(filename, include_uv=True)
            mesh = Mesh(vertices=v, normals=n, faces=f)
            mesh.textures.tc = tc
            return mesh
        if filename.lower().endswith(".mesh"):
            v, t, n, f = load_mesh_file(filename)
            return Mesh(vertices=v, tetrahedra=t, normals=n, faces=f)

    def transform(self, delta: np.ndarray):
        # TODO(@jparr721) FIX THIS STUPID SHIT
        # This is here in place of broadcasting until I sort out the dangling reference issue in the render pass
        for i, row in enumerate(np.add(self.world_coordinates, delta)):
            self.vertices[i] = row

    def tetrahedralize(self):
        if self.tetrahedra.size > 0:
            logger.warning("Mesh is already tetrahedralized")
            return
        v, t, f = tetrahedralize_surface_mesh(self.as_matrix(self.vertices, 3), self.as_matrix(self.faces, 3))
        self.vertices = self.as_vector(v)
        self.faces = self.as_vector(f)
        self.tetrahedra = self.as_vector(t)
        self.normals = self.as_vector(per_face_normals(v, f))
        # TODO(@jparr721) Change this later
        self.colors = np.tile(np.random.rand(3), len(self.vertices.data) // 3).astype(np.float32)
        self.world_coordinates = copy.deepcopy(self.vertices)
        logger.info(f"Random Color: {self.colors[:3]}")

    def as_vector(self, array: np.ndarray):
        assert array.ndim < 3, "Must be at most 2D"
        if array.ndim == 2:
            return array.reshape(-1)
        else:
            return array

    def as_matrix(self, array: np.ndarray, cols: int):
        if array.ndim == 2:
            return array
        else:
            return array.reshape((array.shape[0] // cols, cols))

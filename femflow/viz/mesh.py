import copy
import igl

from femflow.meshing.implicit import gyroid
from femflow.numerics.bintensor3 import bintensor3
import numpy as np
from femflow.meshing.loader import load_mesh_file, load_obj_file
from femflow.numerics.geometry import per_face_normals, tetrahedralize_surface_mesh
from loguru import logger
from PIL import Image
from scipy.sparse import csr_matrix

_MESH_TYPES = {"gyroid", "cuboid"}


class Texture(object):
    """Texture.
    """

    def __init__(self, data: np.ndarray, tc: np.ndarray, u: int, v: int):
        """A texture represents a texture for a mesh

        Args:
            data (np.ndarray): Numpy array containing the data of the texture
            tc (np.ndarray): The texture cordinates
            u (int): # U
            v (int): # V
        """
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
        self.faces = self.as_vector(faces).astype(np.int32)
        self.tetrahedra = self.as_vector(tetrahedra)
        self.normals = self.as_vector(normals)
        self.colors = self.as_vector(colors)
        self.textures = textures

        self.world_coordinates = copy.deepcopy(self.vertices)

        if self.textures.size == 0 and self.colors.size == 0:
            self._set_random_color()

    def save(self, filename: str) -> bool:
        """Saves this mesh to an obj file

        Args:
            filename (str): The name of the file to save to

        Returns:
            bool: True if saved successfully, false otherwise
        """
        return igl.write_obj(
            filename, self.as_matrix(self.vertices, 3), self.as_matrix(self.faces, 3)
        )

    @property
    def tetrahedralized(self) -> bool:
        """Lets you know if the mesh contains tetrahedral elements.

        Returns:
            bool: Whether or not the mesh is tetrahedralized
        """
        return self.tetrahedra is not None and self.tetrahedra.size > 0

    @staticmethod
    def from_file(filename: str) -> "Mesh":
        if filename.lower().endswith(".obj"):
            v, tc, n, f = load_obj_file(filename)
            mesh = Mesh(vertices=v, normals=n, faces=f)
            mesh.textures.tc = tc
            return mesh
        elif filename.lower().endswith(".mesh"):
            v, t, n, f = load_mesh_file(filename)
            return Mesh(vertices=v, tetrahedra=t, normals=n, faces=f)
        else:
            raise ValueError(f"Input file type from file {filename} is not supported")

    @staticmethod
    def from_type(mesh_type: str, **kwargs) -> "Mesh":
        """Creates a mesh from a type string.

        Args:
            mesh_type (str): The type of mesh
            kwargs: amplitude (float), resolution(float), dimension(int) if mesh_type is gyroid.

        Returns:
            Mesh: The mesh object
        """
        if mesh_type not in _MESH_TYPES:
            raise ValueError(
                f"Mesh type {mesh_type} is not in the accepted list of types"
            )

        if mesh_type == "gyroid":
            if "amplitude" not in kwargs:
                raise ValueError(
                    "Property 'amplitude' is required when parameterizing a gyroid"
                )
            if "resolution" not in kwargs:
                raise ValueError(
                    "Property 'resolution' is required when parameterizing a gyroid"
                )
            if "dimension" not in kwargs:
                raise ValueError(
                    "Property 'dimension' is required when parameterizing a gyroid"
                )
            amplitude = float(kwargs["amplitude"])
            resolution = float(kwargs["resolution"])
            dimension = int(kwargs["dimension"])
            scalar_field = gyroid(amplitude, dimension)
            scalar_field = bintensor3(scalar_field)
            scalar_field.padding(0)
            scalar_field.padding(1)
            scalar_field.padding(2)
            v, f = scalar_field.tomesh(resolution)
            return Mesh(vertices=v, faces=f)

        if mesh_type == "cuboid":
            raise NotImplementedError()

        return Mesh()

    def reload_from_mesh(self, mesh: "Mesh") -> None:
        self.vertices = mesh.vertices
        self.faces = mesh.faces
        self.tetrahedra = mesh.tetrahedra
        self.normals = mesh.normals
        self.colors = mesh.colors
        self.textures = mesh.textures

    def reload_from_surface(self, v: np.ndarray, f: np.ndarray) -> None:
        mesh = Mesh(v, f)
        self.vertices = mesh.vertices.copy()
        self.faces = mesh.faces.copy()
        self.tetrahedra = np.array([])
        self.normals = self.as_vector(per_face_normals(v, f))
        self._set_random_color()

    def reload_from_file(self, filename: str) -> None:
        mesh = Mesh.from_file(filename)
        self.vertices = mesh.vertices
        self.faces = mesh.faces
        self.tetrahedra = mesh.tetrahedra
        self.normals = mesh.normals
        self.colors = mesh.colors
        self.textures = mesh.textures

    def transform(self, delta: csr_matrix):
        # yay broadcasting!
        self.vertices[:] = np.add(self.world_coordinates, delta.toarray().reshape(-1))

    def tetrahedralize(self):
        if self.tetrahedra.size > 0:
            logger.warning("Mesh is already tetrahedralized")
            return
        v, t, f = tetrahedralize_surface_mesh(
            self.as_matrix(self.vertices, 3), self.as_matrix(self.faces, 3)
        )
        self.vertices = self.as_vector(v)
        self.faces = self.as_vector(f)
        self.tetrahedra = self.as_vector(t)
        self.normals = self.as_vector(per_face_normals(v, f))
        # TODO(@jparr721) Change this later
        self.colors = np.tile(np.random.rand(3), len(self.vertices.data) // 3).astype(
            np.float32
        )
        self.world_coordinates = copy.deepcopy(self.vertices)
        logger.info(f"Random Color: {self.colors[:3]}")

    def as_vector(self, array: np.ndarray) -> np.ndarray:
        assert array.ndim < 3, "Must be at most 2D"
        if array.ndim == 2:
            return array.reshape(-1)
        else:
            return array

    def as_matrix(self, array: np.ndarray, cols: int) -> np.ndarray:
        if array.ndim == 2:
            return array
        else:
            return array.reshape((array.shape[0] // cols, cols))

    def _set_random_color(self):
        self.colors = np.tile(np.random.rand(3), len(self.vertices.data) // 3).astype(
            np.float32
        )
        logger.info(f"Random Color: {self.colors[:3]}")

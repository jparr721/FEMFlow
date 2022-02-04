import copy

import igl
import numpy as np
from loguru import logger
from scipy.sparse import csr_matrix

from femflow.meshing.implicit import gyroid
from femflow.meshing.loader import load_mesh_file, load_obj_file
from femflow.numerics.bintensor3 import bintensor3
from femflow.numerics.geometry import per_face_normals, tetrahedralize_surface_mesh
from femflow.numerics.linear_algebra import matrix_to_vector, vector_to_matrix

_MESH_TYPES = {"gyroid", "cuboid"}


class Mesh(object):
    def __init__(
        self,
        vertices: np.ndarray = np.array([]),
        faces: np.ndarray = np.array([]),
        tetrahedra: np.ndarray = np.array([]),
        colors: np.ndarray = np.array([]),
        normals: np.ndarray = np.array([]),
    ):
        self.vertices = matrix_to_vector(vertices).astype(np.float32)
        self.faces = matrix_to_vector(faces).astype(np.int32)
        self.tetrahedra = matrix_to_vector(tetrahedra)
        self.normals = matrix_to_vector(normals)
        self.colors = matrix_to_vector(colors)

        self.world_coordinates = copy.deepcopy(self.vertices)

        if self.colors.size == 0:
            self._set_default_color()

    def __setitem__(self, name: str, value: np.ndarray):
        if not isinstance(value, np.ndarray):
            raise ValueError("Input type must be an ndarray")
        if value.ndim > 2:
            raise ValueError("Three-Dimensional arrays are not supported")
        if value.ndim == 2:
            value = matrix_to_vector(value)
        # Don't blast current referencs to this object
        self.__dict__[name] = value

        if name == "vertices":
            self._set_default_color()

    def __add__(self, mesh: "Mesh") -> "Mesh":
        vertices = np.array([])
        faces = np.array([])
        tetrahedra = np.array([])
        colors = np.array([])
        normals = np.array([])

        if self.vertices.size > 0:
            vertices = np.concatenate((self.vertices, mesh.vertices))
        if self.faces.size > 0:
            faces = np.concatenate((self.faces, mesh.faces))
        if self.tetrahedra.size > 0:
            tetrahedra = np.concatenate((self.tetrahedra, mesh.tetrahedra))
        if self.colors.size > 0:
            colors = np.concatenate((self.colors, mesh.colors))
        if self.normals.size > 0:
            normals = np.concatenate((self.normals, mesh.normals))
        return Mesh(vertices, faces, tetrahedra, colors, normals)

    def __iadd__(self, mesh: "Mesh") -> None:
        if self.vertices.size > 0:
            self.vertices = np.concatenate((self.vertices, mesh.vertices))
        if self.faces.size > 0:
            self.faces = np.concatenate((self.faces, mesh.faces))
        if self.tetrahedra.size > 0:
            self.tetrahedra = np.concatenate((self.tetrahedra, mesh.tetrahedra))
        if self.colors.size > 0:
            self.colors = np.concatenate((self.colors, mesh.colors))
        if self.normals.size > 0:
            self.normals = np.concatenate((self.normals, mesh.normals))

    def save(self, filename: str) -> bool:
        """Saves this mesh to an obj file

        Args:
            filename (str): The name of the file to save to

        Returns:
            bool: True if saved successfully, false otherwise
        """
        return igl.write_obj(
            filename, vector_to_matrix(self.vertices, 3), vector_to_matrix(self.faces, 3)
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
            v, _, n, f = load_obj_file(filename)
            mesh = Mesh(vertices=v, normals=n, faces=f)
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

    def translate_x(self, amount: float):
        self.vertices[np.arange(0, len(self.vertices), 3)] += amount

    def translate_y(self, amount: float):
        self.vertices[np.arange(1, len(self.vertices), 3)] += amount

    def translate_z(self, amount: float):
        self.vertices[np.arange(2, len(self.vertices), 3)] += amount

    def reset_positions(self) -> None:
        self.vertices[:] = self.world_coordinates

    def reload_from_mesh(self, mesh: "Mesh") -> None:
        self.vertices = mesh.vertices
        self.faces = mesh.faces
        self.tetrahedra = mesh.tetrahedra
        self.normals = mesh.normals
        self.colors = mesh.colors

    def reload_from_surface(self, v: np.ndarray, f: np.ndarray) -> None:
        mesh = Mesh(v, f)
        self.vertices = mesh.vertices.copy()
        self.faces = mesh.faces.copy()
        self.tetrahedra = np.array([])
        self.normals = matrix_to_vector(per_face_normals(v, f))
        self._set_default_color()

    def reload_from_file(self, filename: str) -> None:
        mesh = Mesh.from_file(filename)
        self.vertices = mesh.vertices
        self.faces = mesh.faces
        self.tetrahedra = mesh.tetrahedra
        self.normals = mesh.normals
        self.colors = mesh.colors

    def transform(self, delta: csr_matrix):
        # yay broadcasting!
        self.vertices[:] = np.add(self.world_coordinates, delta.toarray().reshape(-1))

    def replace(self, new_positions: np.ndarray):
        self.vertices[:] = new_positions

    def tetrahedralize(self):
        if self.tetrahedra.size > 0:
            logger.warning("Mesh is already tetrahedralized")
            return
        v, t, f = tetrahedralize_surface_mesh(
            vector_to_matrix(self.vertices, 3), vector_to_matrix(self.faces, 3)
        )
        self.vertices = matrix_to_vector(v)
        self.faces = matrix_to_vector(f)
        self.tetrahedra = matrix_to_vector(t)
        self.normals = matrix_to_vector(per_face_normals(v, f))
        self._set_default_color()
        self.world_coordinates = copy.deepcopy(self.vertices)

    def set_color(self, color: np.ndarray):
        self.colors = np.tile(color, len(self.vertices) // 3).astype(np.float32)

    def _set_default_color(self):
        color = np.array((255.0 / 255.0, 235.0 / 255.0, 80.0 / 255.0))
        self.set_color(color)

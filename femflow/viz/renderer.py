import os
from enum import IntEnum

import numpy as np
from loguru import logger
from OpenGL.GL import *
from OpenGL.GLU import *

from .camera import Camera
from .mesh import Mesh
from .shader_program import ShaderProgram

FRAG_SHADER_PATH = os.path.join(
    os.getcwd(), "femflow", "viz", "shaders", "core.frag.glsl"
)
VERTEX_SHADER_PATH = os.path.join(
    os.getcwd(), "femflow", "viz", "shaders", "core.vs.glsl"
)


def build_vertex_buffer(
    location: int,
    buffer: GLint,
    stride: int,
    data: np.ndarray,
    offset: ctypes.c_void_p = ctypes.c_void_p(0),
    refresh: bool = True,
):
    glBindBuffer(GL_ARRAY_BUFFER, buffer)
    if refresh:
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_DYNAMIC_DRAW)
    glVertexAttribPointer(
        location, stride, GL_FLOAT, GL_FALSE, stride * data.itemsize, offset
    )
    glEnableVertexAttribArray(location)


def build_index_buffer(buffer: GLint, data: np.ndarray):
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffer)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)


def make_shader_program(vert_shader_path: str, frag_shader_path: str) -> ShaderProgram:
    shader_program = ShaderProgram()
    shader_program.add_shader(GL_VERTEX_SHADER, vert_shader_path)
    shader_program.add_shader(GL_FRAGMENT_SHADER, frag_shader_path)
    shader_program.link()
    return shader_program


class RenderMode(IntEnum):
    MESH = 0
    LINES = 1
    MESH_AND_LINES = 2


class Renderer(object):
    def __init__(self, render_mode: RenderMode = RenderMode.MESH_AND_LINES):
        self.shader_program = make_shader_program(VERTEX_SHADER_PATH, FRAG_SHADER_PATH)
        self.shader_program.bind()
        self.view = self.shader_program.uniform_location("view")
        self.projection = self.shader_program.uniform_location("projection")
        self.light = self.shader_program.uniform_location("light")
        self.normal_matrix = self.shader_program.uniform_location("normal_matrix")

        # Array Objects
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        # Buffers
        self.position_vbo = glGenBuffers(1)
        self.color_vbo = glGenBuffers(1)
        self.normal_vbo = glGenBuffers(1)
        self.faces_ibo = glGenBuffers(1)

        self.shader_program.release()
        self.render_mode = render_mode

        self.mesh: Mesh = Mesh()
        self._wireframe_color = np.zeros(self.mesh.colors.shape)

        self.rebuild_buffers()
        glEnable(GL_DEPTH_TEST)

        self.phi = 0.0001
        self.r = 10
        self.light_x = np.cos(self.phi) * self.r
        self.light_y = np.sin(self.phi) * self.r

    def destroy(self):
        logger.info("Destorying renderer")
        self.shader_program.destroy()
        glDeleteBuffers(
            1, [self.position_vbo, self.color_vbo, self.normal_vbo, self.faces_ibo,],
        )
        glDeleteVertexArrays(1, [self.vao])

    def render(self, camera: Camera):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self.rebuild_buffers()
        self.shader_program.bind()
        self.shader_program.set_matrix_uniform(self.view, camera.view_matrix)

        if self.mesh is not None:
            if self.render_mode == RenderMode.MESH:
                self._render_mesh()
            elif self.render_mode == RenderMode.LINES:
                self._render_lines()
            else:
                self._render_mesh_and_lines()

        self.shader_program.release()

    def resize(self, width: int, height: int, camera: Camera):
        glViewport(0, 0, width, height)
        self.shader_program.bind()
        self.shader_program.set_matrix_uniform(self.projection, camera.projection_matrix)
        self.shader_program.set_matrix_uniform(self.view, camera.view_matrix)

        self.shader_program.set_vector_uniform(
            self.light, np.array([self.light_x, 5.0, self.light_y])
        )
        self.shader_program.set_matrix_uniform(
            self.normal_matrix, np.linalg.inv(camera.view_matrix).T
        )

        self.shader_program.release()

    def rebuild_buffers(self):
        build_vertex_buffer(0, self.position_vbo, 3, self.mesh.vertices)
        build_vertex_buffer(2, self.color_vbo, 3, self.mesh.colors)

        if self.mesh.colors.shape != self._wireframe_color.shape:
            self._wireframe_color = np.zeros(self.mesh.colors.shape)

        if self.mesh.normals.size > 0:
            build_vertex_buffer(1, self.normal_vbo, 3, self.mesh.normals)

        build_index_buffer(self.faces_ibo, self.mesh.faces)

    def _render_mesh(self):
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glDrawElements(GL_TRIANGLES, self.mesh.faces.size, GL_UNSIGNED_INT, None)

    def _render_lines(self):
        build_vertex_buffer(2, self.color_vbo, 3, self._wireframe_color)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        glDrawElements(GL_TRIANGLES, self.mesh.faces.size, GL_UNSIGNED_INT, None)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

    def _render_mesh_and_lines(self):
        glEnable(GL_POLYGON_OFFSET_LINE)
        glPolygonOffset(-1.0, -1.0)

        self._render_mesh()
        self._render_lines()

        glDisable(GL_POLYGON_OFFSET_LINE)

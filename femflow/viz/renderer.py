import os
from enum import Enum

import numpy as np
from loguru import logger
from OpenGL.GL import *
from OpenGL.GLU import *
from utils.graphics.textures import build_texture

from .camera import Camera
from .mesh import Mesh
from .shader_program import ShaderProgram

FRAG_SHADER_PATH = os.path.join(os.getcwd(), "femflow", "viz", "shaders", "core.frag.glsl")
VERTEX_SHADER_PATH = os.path.join(os.getcwd(), "femflow", "viz", "shaders", "core.vs.glsl")


def build_vertex_buffer(
    location: GLint,
    buffer: GLint,
    stride: int,
    data: np.ndarray,
    offset: ctypes.c_void_p = ctypes.c_void_p(0),
    refresh: bool = True,
):
    glBindBuffer(GL_ARRAY_BUFFER, buffer)
    if refresh:
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_DYNAMIC_DRAW)
    glVertexAttribPointer(location, stride, GL_FLOAT, GL_FALSE, stride * data.itemsize, offset)
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


class RenderMode(Enum):
    MESH = 0
    LINES = 1
    MESH_AND_LINES = 2


class Renderer(object):
    def __init__(self, mesh: Mesh = None, render_mode: RenderMode = RenderMode.MESH):
        self.shader_program = make_shader_program(VERTEX_SHADER_PATH, FRAG_SHADER_PATH)
        self.shader_program.bind()
        self.view = self.shader_program.uniform_location("view")
        self.projection = self.shader_program.uniform_location("projection")
        self.light = self.shader_program.uniform_location("light")

        # Array Objects
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        # Buffers
        self.position_vbo = glGenBuffers(1)
        self.color_vbo = glGenBuffers(1)
        self.normal_vbo = glGenBuffers(1)
        self.faces_ibo = glGenBuffers(1)

        self.texture_vbo = glGenTextures(1)

        self.shader_program.release()
        self.render_mode = render_mode

        self.mesh: Mesh = mesh
        self.rebuild_buffers()
        glEnable(GL_DEPTH_TEST)

    def destroy(self):
        logger.info("Destorying renderer")
        self.shader_program.destroy()
        glDeleteBuffers(1, [self.position_vbo, self.color_vbo, self.texture_vbo, self.normal_vbo, self.faces_ibo])
        glDeleteVertexArrays(1, [self.vao])

    def render(self, camera: Camera):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.apply_render_mode()
        self.rebuild_buffers()
        self.shader_program.bind()
        self.shader_program.set_matrix_uniform(self.view, camera.view_matrix)
        if self.mesh is not None:
            glEnable(GL_POLYGON_OFFSET_LINE)
            glPolygonOffset(-1.0, -1.0)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            glDrawElements(GL_TRIANGLES, self.mesh.faces.size, GL_UNSIGNED_INT, None)
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glDrawElements(GL_TRIANGLES, self.mesh.faces.size, GL_UNSIGNED_INT, None)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        self.shader_program.release()

    def append_mesh(self, mesh: Mesh):
        if self.mesh is not None:
            self.mesh += mesh
        else:
            self.mesh = mesh

        self.rebuild_buffers()

    def resize(self, width: int, height: int, camera: Camera):
        logger.debug(f"Resizing to width: {width}, height: {height}")
        # camera.snap_to_mesh(
        #     self.mesh.axis_max(2),
        #     (self.mesh.axis_max(0) - self.mesh.axis_min(1)) / 2,
        #     (self.mesh.axis_max(1) - self.mesh.axis_min(1)) / 2,
        # )
        glViewport(0, 0, width, height)
        self.shader_program.bind()
        self.shader_program.set_matrix_uniform(self.projection, camera.projection_matrix)
        self.shader_program.set_matrix_uniform(self.view, camera.view_matrix)
        self.shader_program.set_matrix_uniform(self.light, camera.projection_matrix)
        self.shader_program.release()

    def rebuild_buffers(self):
        # If we have no meshes, we have no data to add
        if self.mesh:
            build_vertex_buffer(0, self.position_vbo, 3, self.mesh.vertices)
            build_vertex_buffer(3, self.color_vbo, 3, self.mesh.colors)
            if self.mesh.normals.size > 0:
                build_vertex_buffer(2, self.normal_vbo, 3, self.mesh.normals)

            if self.mesh.textures.size > 0:
                build_texture(
                    self.texture_vbo,
                    self.mesh.textures.data,
                    self.mesh.textures.u,
                    self.mesh.textures.v,
                    GL_REPEAT,
                    GL_LINEAR,
                )

            build_index_buffer(self.faces_ibo, self.mesh.faces)

    def apply_render_mode(self):
        if self.render_mode == RenderMode.MESH:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        elif self.render_mode == RenderMode.LINES:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        else:
            # TODO(@jparr721) - Show mesh and lines.
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

import ctypes
import os

import numpy as np
from loguru import logger
from OpenGL.GL import *
from OpenGL.GLU import *

from .camera import Camera
from .gl_util import log_errors
from .mesh import Mesh
from .shader_program import ShaderProgram


class Renderer(object):
    def __init__(self, mesh: Mesh = None):
        self.FRAG_SHADER_PATH = os.path.join(os.getcwd(), "femflow", "core.frag.glsl")
        self.VERTEX_SHADER_PATH = os.path.join(os.getcwd(), "femflow", "core.vs.glsl")

        self.shader_program = ShaderProgram()
        self.shader_program.add_shader(GL_VERTEX_SHADER, self.VERTEX_SHADER_PATH)
        self.shader_program.add_shader(GL_FRAGMENT_SHADER, self.FRAG_SHADER_PATH)
        self.shader_program.link()
        self.shader_program.bind()

        self.projection = self.shader_program.uniform_location("projection")
        self.view = self.shader_program.uniform_location("view")
        self.mesh = mesh

        # Buffers
        self.position_vbo = -1
        self.color_vbo = -1
        self.faces_ibo = -1

        if mesh is not None:
            self._build_buffers()

        self.shader_program.release()
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        glEnable(GL_DEPTH_TEST)

        # TODO(@jparr721) Add dirty states for rendering.

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.info("Destroying buffer objects")
        self.shader_program.destroy()
        glDeleteBuffers(1, [self.position_vbo, self.color_vbo, self.faces_ibo])
        glDeleteVertexArrays(1, [self.vao])

    def set_mesh(self, mesh: Mesh):
        self.mesh = mesh
        self._build_buffers()

    def resize(self, width, height, camera: Camera):
        logger.debug(f"Resizing to width: {width}, height: {height}")
        camera.resize(width, height)
        glViewport(0, 0, width, height)
        self.shader_program.bind()
        self.shader_program.set_matrix_uniform(self.projection, camera.projection_matrix)
        self.shader_program.set_matrix_uniform(self.view, camera.view_matrix)
        self.shader_program.release()

    def render(self, camera: Camera):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self.reload_buffers()
        self.shader_program.bind()

        self.shader_program.set_matrix_uniform(self.view, camera.view_matrix)
        glDrawElements(GL_TRIANGLES, self.mesh.faces.size, GL_UNSIGNED_INT, None)
        self.shader_program.release()

        log_errors(self.render.__name__)

    def reload_buffers(self):
        assert self.mesh is not None, "No mesh found! Cannot initialize buffers!"

        self._bind_vbo("position", self.position_vbo, 3, self.mesh.vertices)
        self._bind_vbo("color", self.color_vbo, 3, self.mesh.colors)
        self._bind_ibo(self.mesh.faces)

        log_errors(self.reload_buffers.__name__)

    def _build_buffers(self):
        logger.info("Initializing buffer objects")
        assert self.mesh is not None, "No mesh found! Cannot initialize buffers!"

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        self.position_vbo = glGenBuffers(1)
        self.color_vbo = glGenBuffers(1)
        self.faces_ibo = glGenBuffers(1)

        self.reload_buffers()
        log_errors(self._build_buffers.__name__)

    def _bind_vbo(self, name: str, buffer: int, stride: int, data: np.ndarray, refresh: bool = True):
        handle = glGetAttribLocation(self.shader_program.id, name)
        glBindBuffer(GL_ARRAY_BUFFER, buffer)
        if refresh:
            glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_DYNAMIC_DRAW)
        glVertexAttribPointer(handle, stride, GL_FLOAT, GL_FALSE, stride * data.itemsize, ctypes.c_void_p(0))
        glEnableVertexAttribArray(handle)

    def _bind_ibo(self, data: np.ndarray):
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.faces_ibo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)

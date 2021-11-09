import atexit
import ctypes
import logging
import os

from OpenGL.GL import *

from gl_util import log_errors
from mesh import Mesh
from shader_program import ShaderProgram

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Renderer(object):
    def __init__(self, mesh: Mesh = None):
        self.FRAG_SHADER_PATH = os.path.join(os.getcwd(), "femflow", "core.frag.glsl")
        self.VERTEX_SHADER_PATH = os.path.join(os.getcwd(), "femflow", "core.vs.glsl")

        self.shader_program = ShaderProgram()
        self.shader_program.add_shader(GL_VERTEX_SHADER, self.VERTEX_SHADER_PATH)
        self.shader_program.add_shader(GL_FRAGMENT_SHADER, self.FRAG_SHADER_PATH)
        self.shader_program.link()
        self.shader_program.bind()

        self.mvp = self.shader_program.uniform_location("mvp")
        self.mesh = mesh

        if mesh is not None:
            self._build_buffers()

        self.shader_program.release()

        # TODO(@jparr721) Add dirty states for rendering.

        atexit.register(self.destroy_buffers)

    def set_mesh(self, mesh: Mesh):
        self.mesh = mesh
        self._build_buffers()

    def render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self.reload_buffers()
        self.shader_program.bind()
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, len(self.mesh.faces), GL_UNSIGNED_INT, None)
        glBindVertexArray(self.vao)  # self.vao might need to be 0
        self.shader_program.release()
        log_errors(self.render.__name__)

    def reload_buffers(self):
        assert self.mesh is not None, "No mesh found! Cannot initialize buffers!"
        self._bind_vbo("position", self.position_vbo, 3, self.mesh.vertices)
        self._bind_vbo("color", self.color_vbo, 4, self.mesh.colors)
        self._bind_ibo(self.mesh.faces, True)
        log_errors(self.reload_buffers.__name__)

    def destroy_buffers(self):
        logger.info("Destroying buffer objects")
        glDeleteBuffers(1, [self.position_vbo, self.color_vbo, self.faces_index_buffer])
        glDeleteVertexArrays(1, [self.vao])

    def _build_buffers(self):
        logger.info("Initializing buffer objects")
        assert self.mesh is not None, "No mesh found! Cannot initialize buffers!"

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        self.position_vbo = glGenBuffers(1)
        self.color_vbo = glGenBuffers(1)

        self._bind_vbo("position", self.position_vbo, 3, self.mesh.vertices)
        self._bind_vbo("color", self.color_vbo, 4, self.mesh.colors)
        self._bind_ibo(self.mesh.faces, True)
        log_errors(self._build_buffers.__name__)

    def _bind_vbo(self, name: str, buffer: int, stride: int, data, refresh: bool = True):
        handle = glGetAttribLocation(self.shader_program.id, name)
        glBindBuffer(GL_ARRAY_BUFFER, buffer)
        if refresh:
            list_type = GLfloat * len(data)
            glBufferData(GL_ARRAY_BUFFER, ctypes.sizeof(ctypes.c_float) * len(data), list_type(*data), GL_DYNAMIC_DRAW)

        glVertexAttribPointer(handle, stride, GL_FLOAT, GL_FALSE, stride * ctypes.sizeof(ctypes.c_float), None)
        glEnableVertexAttribArray(handle)

    def _bind_ibo(self, data, refresh: bool = False):
        if refresh:
            self.faces_index_buffer = glGenBuffers(1)
        list_type = GLfloat * len(data)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.faces_index_buffer)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, ctypes.sizeof(ctypes.c_int) * len(data), list_type(*data), GL_STATIC_DRAW)

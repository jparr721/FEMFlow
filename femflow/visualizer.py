import ctypes
import logging

import glfw
import imgui
from imgui.integrations.glfw import GlfwRenderer
from OpenGL.GL import *

logger = logging.getLogger(__name__)


class Visualizer(object):
    def __init__(self):
        self.WINDOW_TITLE = "FEMFlow Viewer"
        self.WINDOW_WIDTH = 1200
        self.WINDOW_HEIGHT = 800
        self.FRAG_SHADER_PATH = "core.frag.glsl"
        self.VERTEX_SHADER_PATH = "core.vs.glsl"

    def __exit__(self):
        glDeleteBuffers(1, [self.vbo_id])
        glDeleteVertexArrays(1, [self.vao_id])
        glfw.terminate()

    def launch(self):
        assert glfw.init(), "GLFW is not initialized!"

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        logger.debug("Drawing window")

        self.window = glfw.create_window(self.WINDOW_WIDTH, self.WINDOW_HEIGHT, self.WINDOW_TITLE, None, None)

        assert self.window, "GLFW failed to open the window"

        glfw.make_context_current(self.window)
        glClearColor(0, 0, 0, 1)

    def create_vao(self):
        self.vao_id = glGenVertexArrays(1)
        glBindVertexArray(self.vao_id)

    def create_vbo(self):
        assert self.vao_id, "No VAO found!"
        vertex_data = [-1, -1, 0, 1, -1, 0, 0, 1, 0]

        self.vbo_id = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_id)
        array_type = GLfloat * len(vertex_data)
        glBufferData(
            GL_ARRAY_BUFFER, len(vertex_data) * ctypes.sizeof(ctypes.c_float), array_type(*vertex_data), GL_STATIC_DRAW
        )

        glVertexAttribPointer(0, 3, GL_FLOAT, False, 0, None)
        glEnableVertexAttribArray(0)

    def create_shaders(self):
        self.vertex_shader_source = ""
        self.fragment_shader_source = ""

        with open(self.VERTEX_SHADER_PATH, "r") as f:
            self.vertex_shader_source = f.read()

        assert len(self.vertex_shader_source) > 0, "No vertex shader found"

        with open(self.FRAG_SHADER_PATH, "r") as f:
            self.fragment_shader_source = f.read()

        assert len(self.fragment_shader_source) > 0, "No fragment shader found"

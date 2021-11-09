import ctypes
from collections import defaultdict

from OpenGL.GL import *

from gl_util import log_errors
from shader import Shader


class ShaderProgram(object):
    def __init__(self):
        self.id = glCreateProgram()
        self.shaders = defaultdict(Shader)

    def __del__(self):
        for shader in self.shaders.values():
            glDeleteShader(shader.id)
        glDeleteProgram(self.id)

    def add_shader(self, shader_type: GLenum, path: str):
        shader = Shader(shader_type, path)
        assert shader.build(), "Shader failed to build!"
        self.shaders[shader_type] = shader
        glAttachShader(self.id, shader.id)
        log_errors(self.add_shader.__name__)

    def link(self):
        glLinkProgram(self.id)

    def bind(self):
        glUseProgram(self.id)

    def release(self):
        glUseProgram(0)

    def set_matrix_uniform_identity(self):
        glLoadIdentity()

    def uniform_location(self, name: str) -> int:
        return glGetUniformLocation(self.id, name)

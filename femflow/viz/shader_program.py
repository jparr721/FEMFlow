from typing import Dict, Union

from loguru import logger
from OpenGL.GL import *

from .shader import Shader


class ShaderProgram(object):
    def __init__(self):
        self.id = glCreateProgram()
        self.shaders: Dict[int, Shader] = dict()

    def destroy(self):
        logger.info("Destroying shaders")
        for shader in self.shaders.values():
            glDeleteShader(shader.id)
        glDeleteProgram(self.id)

    def add_shader(self, shader_type: int, path: str):
        logger.info(f"Adding shader: {shader_type} at path: {path}")
        shader = Shader(shader_type, path)
        assert shader.build(), "Shader failed to build!"
        self.shaders[shader_type] = shader
        glAttachShader(self.id, shader.id)

    def link(self):
        glLinkProgram(self.id)
        value = glGetProgramInfoLog(self.id)
        if value:
            logger.error(value)

    def bind(self):
        glUseProgram(self.id)

    def release(self):
        glUseProgram(0)

    def set_matrix_uniform_identity(self):
        glLoadIdentity()

    def set_matrix_uniform(self, location: Union[int, str], uniform):
        if isinstance(location, str):
            location = self.uniform_location(location)

        glUniformMatrix4fv(location, 1, GL_FALSE, uniform)

    def set_vector_uniform(self, location: Union[int, str], uniform):
        if isinstance(location, str):
            location = self.uniform_location(location)

        glUniform3f(location, *uniform)

    def uniform_location(self, name: str) -> int:
        return glGetUniformLocation(self.id, name)

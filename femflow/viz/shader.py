from loguru import logger
from OpenGL.GL import GL_COMPILE_STATUS, GLenum, glCreateShader, glGetShaderInfoLog, glGetShaderiv, glShaderSource
from OpenGL.raw.GL.VERSION.GL_2_0 import GL_INFO_LOG_LENGTH, glCompileShader

from .gl_util import log_errors


class Shader(object):
    def __init__(self, type: GLenum, path: str):
        self.id = -1
        self.type = type
        self.shader_path = path

    def __repr__(self):
        return "Id: " + repr(self.id) + " Type: " + repr(self.type) + " Path: " + repr(self.shader_path)

    def build(self) -> bool:
        shader_source = ""
        with open(self.shader_path, "r") as f:
            shader_source = f.read()

        assert len(shader_source) > 0, "Shader is empty"
        self.id = glCreateShader(self.type)

        assert self.id > 0, "Shader failed to create"

        glShaderSource(self.id, shader_source)
        glCompileShader(self.id)

        glGetShaderiv(self.id, GL_COMPILE_STATUS)
        log_len = glGetShaderiv(self.id, GL_INFO_LOG_LENGTH)
        if log_len:
            message = glGetShaderInfoLog(self.id)
            logger.error(message)
            return False

        log_errors(self.build.__name__)

        return True

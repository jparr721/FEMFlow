import logging

from OpenGL.GL import GL_COMPILE_STATUS, GLenum, glCreateShader, glGetShaderInfoLog, glGetShaderiv, glShaderSource

from gl_util import log_errors

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Shader(object):
    def __init__(self, type: GLenum, path: str):
        self.id = -1
        self.type = type
        self.shader_path = path

    def build(self) -> bool:
        shader_source = ""
        with open(self.shader_path, "r") as f:
            shader_source = f.read()

        assert len(shader_source) > 0, "Shader is empty"
        self.id = glCreateShader(self.type)
        glShaderSource(self.id, shader_source)

        not_linked = glGetShaderiv(self.id, GL_COMPILE_STATUS)
        if not_linked:
            message = glGetShaderInfoLog(self.id)
            logger.error(message)
            return False

        log_errors(self.build.__name__)

        return True

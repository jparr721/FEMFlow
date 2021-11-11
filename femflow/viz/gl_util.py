from loguru import logger
from OpenGL.GL import GL_NO_ERROR, glGetError
from OpenGL.GLU import gluErrorString


def log_errors(fn_name: str):
    while True:
        err = glGetError()
        if err == GL_NO_ERROR:
            break

        logger.error(fn_name, gluErrorString(err))

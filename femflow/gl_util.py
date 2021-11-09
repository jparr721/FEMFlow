from logging import error

from OpenGL.GL import GL_NO_ERROR, glGetError
from OpenGL.GLU import gluErrorString


def log_errors(fn_name: str):
    while True:
        err = glGetError()
        if err == GL_NO_ERROR:
            break

        print(f"Error in {fn_name}: ", gluErrorString(err))

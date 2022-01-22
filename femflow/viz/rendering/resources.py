import os
from enum import IntEnum

import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *

from ..shader_program import ShaderProgram

FRAG_SHADER_PATH = os.path.join(
    os.getcwd(), "femflow", "viz", "shaders", "core.frag.glsl"
)
VERTEX_SHADER_PATH = os.path.join(
    os.getcwd(), "femflow", "viz", "shaders", "core.vs.glsl"
)


class RenderMode(IntEnum):
    MESH = 0
    LINES = 1
    MESH_AND_LINES = 2


def build_vertex_buffer(
    location: int,
    buffer: GLint,
    stride: int,
    data: np.ndarray,
    offset: ctypes.c_void_p = ctypes.c_void_p(0),
    refresh: bool = True,
):
    glBindBuffer(GL_ARRAY_BUFFER, buffer)
    if refresh:
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_DYNAMIC_DRAW)
    glVertexAttribPointer(
        location, stride, GL_FLOAT, GL_FALSE, stride * data.itemsize, offset
    )
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

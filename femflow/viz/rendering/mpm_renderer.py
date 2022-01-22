import os
from enum import IntEnum

import numpy as np
from loguru import logger
from OpenGL.GL import *
from OpenGL.GLU import *

from ..camera import Camera
from ..mesh import Mesh
from ..mpm_mesh import MPMMesh
from .resources import *


class MPMRenderer(object):
    def __init__(self, render_mode: RenderMode = RenderMode.MESH):
        self.render_mode = render_mode

        self.shader_program = make_shader_program(VERTEX_SHADER_PATH, FRAG_SHADER_PATH)
        self.shader_program.bind()
        self.view = self.shader_program.uniform_location("view")
        self.projection = self.shader_program.uniform_location("projection")
        self.light = self.shader_program.uniform_location("light")
        self.normal_matrix = self.shader_program.uniform_location("normal_matrix")

        # Array Objects
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        # Buffers
        self.position_vbo = glGenBuffers(1)
        self.color_vbo = glGenBuffers(1)
        self.normal_vbo = glGenBuffers(1)
        self.faces_ibo = glGenBuffers(1)

        self.shader_program.release()
        self.mesh = MPMMesh()

import abc
from collections import defaultdict
from typing import Dict

from .resources import *


class Renderer(object):
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

        self.buffers: Dict[str, int] = defaultdict(int)
        self._bind_buffers()

    @abc.abstractmethod
    def _bind_buffers(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def _build_buffers(self):
        raise NotImplementedError()

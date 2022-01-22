import abc
from collections import defaultdict
from typing import Dict

from ..camera import Camera
from ..mesh import Mesh
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

        # Set up and assign buffers
        self.buffers: Dict[str, GLint] = defaultdict(GLint)
        self._bind_buffers()

        self.shader_program.release()

        self.mesh: Mesh = Mesh()

        self._reload_buffers()

        # Lighting
        self.phi = 0.0001
        self.r = 10
        self.light_pos = np.array(
            [np.cos(self.phi) * self.r, 5.0, np.sin(self.phi) * self.r]
        )

        glEnable(GL_DEPTH_TEST)

    @abc.abstractmethod
    def _bind_buffers(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def _reload_buffers(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def render(self, camera: Camera):
        raise NotImplementedError()

    def resize(self, width: int, height: int, camera: Camera):
        glViewport(0, 0, width, height)
        self.shader_program.bind()
        self.shader_program.set_matrix_uniform(self.projection, camera.projection_matrix)
        self.shader_program.set_matrix_uniform(self.view, camera.view_matrix)

        self.shader_program.set_vector_uniform(self.light, self.light_pos)
        self.shader_program.set_matrix_uniform(
            self.normal_matrix, np.linalg.inv(camera.view_matrix).T
        )

        self.shader_program.release()

    def destroy(self):
        self.shader_program.destroy()
        glDeleteBuffers(1, list(self.buffers.values()))
        glDeleteVertexArrays(1, [self.vao])


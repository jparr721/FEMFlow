import abc
from collections import defaultdict
from functools import cache
from typing import Dict

from femflow.numerics.linear_algebra import matrix_to_vector

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

    def _render_grid(self, grid_size: int = 100):
        vertices, colors = self._make_grid_data(grid_size)

        build_vertex_buffer(0, self.buffers["position"], 3, vertices)
        build_vertex_buffer(2, self.buffers["color"], 3, colors)
        glDrawArrays(GL_LINES, 0, vertices.size // 3)

    @cache
    def _make_grid_data(self, grid_size: int):
        grid_color = [0.5, 0.5, 0.5]

        spacing_scale = 1.0

        colors = []
        vertices = []
        for i in range(-grid_size, grid_size):
            vertices.append([i * spacing_scale, 0.0, -grid_size * spacing_scale])
            vertices.append([i * spacing_scale, 0.0, grid_size * spacing_scale])
            vertices.append([-grid_size * spacing_scale, 0.0, i * spacing_scale])
            vertices.append([grid_size * spacing_scale, 0.0, i * spacing_scale])

        colors = [grid_color] * len(vertices)

        return (
            matrix_to_vector(np.array(vertices)).astype(np.float32),
            matrix_to_vector(np.array(colors)).astype(np.float32),
        )

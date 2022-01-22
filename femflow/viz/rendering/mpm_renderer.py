from functools import cache

from OpenGL.GL import *
from OpenGL.GLU import *

from femflow.numerics.linear_algebra import matrix_to_vector

from ..camera import Camera
from .renderer import Renderer
from .resources import *


class MPMRenderer(Renderer):
    def __init__(self, render_mode: RenderMode = RenderMode.MESH):
        super().__init__(render_mode)

    def _bind_buffers(self):
        self.buffers["position"] = glGenBuffers(1)
        self.buffers["color"] = glGenBuffers(1)

    def _reload_buffers(self):
        build_vertex_buffer(0, self.buffers["position"], 3, self.mesh.vertices)
        build_vertex_buffer(2, self.buffers["color"], 3, self.mesh.colors)

    def render(self, camera: Camera):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        vertices, colors = self._render_grid()

        build_vertex_buffer(0, self.buffers["position"], 3, vertices)
        build_vertex_buffer(2, self.buffers["color"], 3, colors)
        glDrawArrays(GL_POINTS, 0, vertices.size)

        self._reload_buffers()
        self.shader_program.bind()
        self.shader_program.set_matrix_uniform(self.view, camera.view_matrix)
        glPointSize(5.0)
        glDrawArrays(GL_POINTS, 0, self.mesh.vertices.size)
        self.shader_program.release()

    @cache
    def _render_grid(self):
        grid_size = 100
        grid_color = [1.0, 1.0, 1.0]

        v_low = -int(grid_size * 0.5)
        v_high = int(grid_size * 0.5)
        u_low = -int(grid_size * 0.5)
        u_high = int(grid_size * 0.5)
        spacing_scale = 1.0

        colors = []
        vertices = []
        for i in range(u_low, u_high):
            vertices.append([i * spacing_scale, 0.0, v_low * spacing_scale])
            vertices.append([i * spacing_scale, 0.0, v_high * spacing_scale])

        for i in range(v_low, v_high):
            vertices.append([u_low * spacing_scale, 0.0, i * spacing_scale])
            vertices.append([u_high * spacing_scale, 0.0, i * spacing_scale])

        colors = [grid_color] * len(vertices)

        return matrix_to_vector(np.array(vertices)), matrix_to_vector(np.array(colors))

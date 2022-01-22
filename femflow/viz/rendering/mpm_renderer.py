from OpenGL.GL import *
from OpenGL.GLU import *

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

        self._reload_buffers()
        self.shader_program.bind()
        self.shader_program.set_matrix_uniform(self.view, camera.view_matrix)
        glDrawArrays(GL_POINTS, 0, self.mesh.vertices.size)
        self.shader_program.release()

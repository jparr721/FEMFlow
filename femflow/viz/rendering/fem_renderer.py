import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *

from ..camera import Camera
from ..mesh import Mesh
from .renderer import Renderer
from .resources import *


class FEMRenderer(Renderer):
    def __init__(self, render_mode: RenderMode = RenderMode.MESH_AND_LINES):
        self._wireframe_color = np.zeros(1)
        super().__init__(render_mode)

    def _bind_buffers(self):
        # Buffers
        self.buffers["position"] = glGenBuffers(1)
        self.buffers["color"] = glGenBuffers(1)
        self.buffers["normal"] = glGenBuffers(1)
        self.buffers["faces"] = glGenBuffers(1)

    def _reload_buffers(self):
        build_vertex_buffer(0, self.buffers["position"], 3, self.mesh.vertices)
        build_vertex_buffer(2, self.buffers["color"], 3, self.mesh.colors)

        if self.mesh.colors.shape != self._wireframe_color.shape:
            self._wireframe_color = np.zeros(self.mesh.colors.shape)

        if self.mesh.normals.size > 0:
            build_vertex_buffer(1, self.buffers["normal"], 3, self.mesh.normals)

        build_index_buffer(self.buffers["faces"], self.mesh.faces)

    def set_mesh(self, mesh: Mesh):
        self.mesh = mesh
        self._wireframe_color = np.zeros(self.mesh.colors.shape)

    def render(self, camera: Camera):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self._reload_buffers()
        self.shader_program.bind()
        self.shader_program.set_matrix_uniform(self.view, camera.view_matrix)

        if self.render_mode == RenderMode.MESH:
            self._render_mesh()
        elif self.render_mode == RenderMode.LINES:
            self._render_lines()
        else:
            self._render_mesh_and_lines()

        self.shader_program.release()

    def _render_mesh(self):
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glDrawElements(GL_TRIANGLES, self.mesh.faces.size, GL_UNSIGNED_INT, None)

    def _render_lines(self):
        build_vertex_buffer(2, self.buffers["color"], 3, self._wireframe_color)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        glDrawElements(GL_TRIANGLES, self.mesh.faces.size, GL_UNSIGNED_INT, None)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

    def _render_mesh_and_lines(self):
        glEnable(GL_POLYGON_OFFSET_LINE)
        glPolygonOffset(-1.0, -1.0)

        self._render_mesh()
        self._render_lines()

        glDisable(GL_POLYGON_OFFSET_LINE)

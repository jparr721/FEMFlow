from functools import cache

from OpenGL.GL import *
from OpenGL.GLU import *

from femflow.numerics.linear_algebra import vector_to_matrix

from ..camera import Camera
from .renderer import Renderer
from .resources import *


class MPMRenderer(Renderer):
    def __init__(self, render_mode: RenderMode = RenderMode.MESH):
        super().__init__(render_mode)

    def _bind_buffers(self):
        self.buffers["position"] = glGenBuffers(1)
        self.buffers["color"] = glGenBuffers(1)
        self.buffers["faces"] = glGenBuffers(1)

    def _reload_buffers(self):
        build_vertex_buffer(0, self.buffers["position"], 3, self.mesh.vertices)
        build_vertex_buffer(2, self.buffers["color"], 3, self.mesh.colors)

    def render(self, camera: Camera):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self.shader_program.bind()
        self.shader_program.set_matrix_uniform(self.view, camera.view_matrix)

        self._render_grid()

        if self.mesh.vertices.size >= 3:
            self._render_bounding_box()

            self._reload_buffers()
            self._render_points()

        self.shader_program.release()

    def _render_points(self):
        glPointSize(5.0)
        glDrawArrays(GL_POINTS, 0, self.mesh.vertices.size)

    def _render_bounding_box(self):
        vertices, faces, colors = self._make_bb_data()
        build_vertex_buffer(0, self.buffers["position"], 3, vertices)
        build_vertex_buffer(2, self.buffers["color"], 3, colors)
        build_index_buffer(self.buffers["faces"], faces)

        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        glDrawElements(GL_TRIANGLES, faces.size, GL_UNSIGNED_INT, None)

        # Leaving this as lines breaks imgui.
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

    @cache
    def _make_bb_data(self):
        v = vector_to_matrix(self.mesh.vertices, 3)
        minx, miny, minz = np.amin(v, axis=0)
        maxx, maxy, maxz = np.amax(v, axis=0) * 10

        minx -= maxx / 2
        maxx -= maxx / 2
        minz -= maxz / 2
        maxz -= maxz / 2
        miny = 0

        # self.mesh.translate_x(maxx / 2)
        # self.mesh.translate_z(maxz / 2)

        vertices = np.array(
            [
                minx,
                miny,
                minz,  # Back Bottom Left
                maxx,
                miny,
                minz,  # Back Bottom Right
                minx,
                maxy,
                minz,  # Back Top Left
                maxx,
                maxy,
                minz,  # Back Top Right
                minx,
                miny,
                maxz,  # Front Bottom Left
                maxx,
                miny,
                maxz,  # Front Bottom Right
                minx,
                maxy,
                maxz,  # Front Top Left
                maxx,
                maxy,
                maxz,  # Front Top Right
            ],
            dtype=np.float32,
        )

        faces = np.array(
            [
                0,
                1,
                3,
                1,
                3,
                2,
                1,
                7,
                3,
                1,
                5,
                7,
                4,
                7,
                5,
                4,
                6,
                7,
                6,
                2,
                7,
                7,
                2,
                3,
                0,
                2,
                6,
                4,
                0,
                6,
            ],
            dtype=np.uint32,
        )

        color = [1.0, 0.0, 0.0]
        colors = np.tile(color, len(vertices) // 3).astype(np.float32)

        return vertices, faces, colors


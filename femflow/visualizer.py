import logging

import glfw
import imgui
from imgui.integrations.glfw import GlfwRenderer
from OpenGL.GL import *

from mesh import Mesh
from renderer import Renderer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class Visualizer(object):
    def __init__(self):
        self.WINDOW_TITLE = "FEMFlow Viewer"
        self.WINDOW_WIDTH = 1200
        self.WINDOW_HEIGHT = 800
        self.background_color = [0, 0, 0, 0]

        assert glfw.init(), "GLFW is not initialized!"

        logger.info("GLFW Initialized.")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        logger.debug("Drawing window")

        self.window = glfw.create_window(self.WINDOW_WIDTH, self.WINDOW_HEIGHT, self.WINDOW_TITLE, None, None)

        assert self.window, "GLFW failed to open the window"

        glfw.make_context_current(self.window)
        glClearColor(*self.background_color)

        mesh = Mesh("femflow/cuboid.obj")
        self.renderer: Renderer = Renderer(mesh)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.info("Destroying glfw")
        glfw.terminate()

    def launch(self):
        while not glfw.window_should_close(self.window):
            glfw.swap_buffers(self.window)
            glfw.poll_events()
            self.renderer.render()


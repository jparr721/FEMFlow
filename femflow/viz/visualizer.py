import glfw
import imgui
import numpy as np
from imgui.integrations.glfw import GlfwRenderer
from loguru import logger
from OpenGL.GL import *

from .camera import Camera
from .input import Input
from .mesh import Mesh
from .renderer import Renderer


class Visualizer(object):
    def __init__(self):
        self.WINDOW_TITLE = "FEMFlow Viewer"
        self.WINDOW_WIDTH = 1200
        self.WINDOW_HEIGHT = 800
        self.background_color = [1, 1, 1, 0]

        self.camera = Camera()
        self.camera.resize(self.WINDOW_WIDTH, self.WINDOW_HEIGHT)

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

        glfw.set_mouse_button_callback(self.window, self.mouse_callback)
        glfw.set_scroll_callback(self.window, self.scroll_callback)
        glfw.set_cursor_pos_callback(self.window, self.mouse_move_callback)
        glfw.set_window_size_callback(self.window, self.window_size_callback)

        glfw.make_context_current(self.window)
        glEnable(GL_DEPTH_TEST)
        glClearColor(*self.background_color)

        self.input = Input()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.info("Destroying glfw")
        glfw.terminate()

    def scroll_callback(self, window, xoffset, yoffset):
        self.input.scroll_event(window, xoffset, yoffset, self.camera)

    def mouse_callback(self, window, button, action, mods):
        self.input.handle_mouse(window, button, action, mods, self.camera)

    def mouse_move_callback(self, window, xpos, ypos):
        self.input.handle_mouse_move(window, xpos, ypos, self.camera)

    def window_size_callback(self, window, width, height):
        self.camera.resize(width, height)

        if self.renderer is not None:
            self.renderer.resize(width, height, self.camera)

    def launch(self):
        mesh = Mesh("femflow/cube.ply")
        with Renderer(mesh) as self.renderer:
            while not glfw.window_should_close(self.window):
                glfw.swap_buffers(self.window)
                glfw.poll_events()
                self.renderer.render(self.camera)

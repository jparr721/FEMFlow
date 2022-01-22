from typing import List

import glfw
import imgui
from imgui.integrations.glfw import GlfwRenderer
from loguru import logger
from OpenGL.GL import *

from ..camera import Camera
from ..input import Input
from ..mesh import Mesh
from ..rendering.fem_renderer import FEMRenderer
from .visualizer_window import VisualizerWindow


class Window(object):
    def __init__(self, title: str = "FEMFlow GUI"):
        self.title = title

        self.background_color = [1.0, 1.0, 1.0, 0.0]

        self.camera = Camera()
        self.input = Input()

        assert glfw.init(), "GLFW is not initialized!"
        imgui.create_context()
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        monitor = glfw.get_primary_monitor()
        mode = glfw.get_video_mode(monitor)
        self.window_width = int(mode.size.width * 0.5)
        self.window_height = int(mode.size.height * 0.5)
        self.camera.resize(self.window_width, self.window_height)

        logger.info("Opening main window")

        self._window = glfw.create_window(
            self.window_width, self.window_height, self.title, None, None
        )

        assert self._window, "GLFW failed to open the window"

        glfw.make_context_current(self._window)
        self.imgui_impl = GlfwRenderer(self._window, attach_callbacks=True)

        glfw.set_mouse_button_callback(self._window, self.mouse_callback)
        glfw.set_scroll_callback(self._window, self.scroll_callback)
        glfw.set_cursor_pos_callback(self._window, self.mouse_move_callback)
        glfw.set_window_size_callback(self._window, self.window_size_callback)

        self.renderer = FEMRenderer()
        glClearColor(*self.background_color)

        self.windows: List[VisualizerWindow] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.info("Destroying imgui")
        self.imgui_impl.shutdown()

        logger.info("Destroying glfw")
        glfw.terminate()

    @property
    def any_window_focused(self):
        return False

    def scroll_callback(self, window, xoffset, yoffset):
        if not self.any_window_focused:
            self.input.scroll_event(window, xoffset, yoffset, self.camera)
        else:
            self.imgui_impl.scroll_callback(window, xoffset, yoffset)

    def mouse_callback(self, window, button, action, mods):
        if not self.any_window_focused:
            self.input.handle_mouse(window, button, action, mods, self.camera)
        else:
            self.imgui_impl.mouse_callback(window, button, action, mods)

    def mouse_move_callback(self, window, xpos, ypos):
        if not self.any_window_focused:
            self.input.handle_mouse_move(window, xpos, ypos, self.camera)

    def window_size_callback(self, _, width, height):
        self.camera.resize(width, height)
        self.imgui_impl.io.display_size = width, height
        self.window_width = width
        self.window_height = height
        self.renderer.resize(self.window_width, self.window_height, self.camera)

    def launch(self):
        self.camera.resize(self.window_width, self.window_height)
        self.renderer.resize(self.window_width, self.window_height, self.camera)
        while not glfw.window_should_close(self._window):
            glfw.poll_events()
            self.imgui_impl.process_inputs()

            imgui.new_frame()

            # Render all widows before anything else
            [window() for window in self.windows]
            self.renderer.render(self.camera)
            imgui.render()
            self.imgui_impl.render(imgui.get_draw_data())

            glfw.swap_buffers(self._window)
        self.renderer.destroy()

    def add_mesh(self, obj_file: str):
        if not obj_file.lower().endswith(".obj"):
            raise ValueError("Only OBJ files are supported")
        self.renderer.mesh = Mesh.from_file(obj_file)
        self.renderer.mesh.tetrahedralize()

    def add_window(self, window: VisualizerWindow):
        pass

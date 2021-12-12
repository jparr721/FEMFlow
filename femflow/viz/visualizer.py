import os

import glfw
import imgui
import numpy as np
from imgui.integrations.glfw import GlfwRenderer
from loguru import logger
from OpenGL.GL import *
from utils.filesystem import file_dialog

from .camera import Camera
from .input import Input
from .mesh import Mesh, Texture
from .renderer import Renderer

logger.add("femflow.log", mode="w+")


class Visualizer(object):
    def __init__(self):
        self.WINDOW_TITLE = "FEMFlow Viewer"
        self.window_width = 1200
        self.window_height = 800
        self.background_color = [1, 1, 1, 0]

        self.camera = Camera()
        self.camera.resize(self.window_width, self.window_height)

        self.log_window_focused = False
        self.log_window_width = self.window_width
        self.log_window_height = self.window_height * 0.2 if self.window_height >= 800 else 160
        self.log_window_pos = (0, self.window_height - self.log_window_height)

        self.menu_window_focused = False
        self.menu_window_width = self.window_width * 0.15 if self.window_width >= 800 else 130
        self.menu_window_height = self.window_height - self.log_window_height
        self.menu_window_pos = (0, 0)

        assert glfw.init(), "GLFW is not initialized!"

        logger.info("GLFW Initialized.")

        imgui.create_context()
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        logger.debug("Drawing window")

        self.window = glfw.create_window(self.window_width, self.window_height, self.WINDOW_TITLE, None, None)

        assert self.window, "GLFW failed to open the window"

        glfw.make_context_current(self.window)
        self.imgui_impl = GlfwRenderer(self.window, attach_callbacks=True)

        glfw.set_mouse_button_callback(self.window, self.mouse_callback)
        glfw.set_scroll_callback(self.window, self.scroll_callback)
        glfw.set_cursor_pos_callback(self.window, self.mouse_move_callback)
        glfw.set_window_size_callback(self.window, self.window_size_callback)

        glClearColor(*self.background_color)

        self.input = Input()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.info("Destroying imgui")
        self.imgui_impl.shutdown()

        logger.info("Destroying glfw")
        glfw.terminate()

    @property
    def any_window_focused(self):
        return self.menu_window_focused or self.log_window_focused

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

    def window_size_callback(self, window, width, height):
        self.camera.resize(width, height)
        self.imgui_impl.io.display_size = width, height
        self.window_width = width
        self.window_height = height

        if self.renderer is not None:
            self.renderer.resize(self.window_width, self.window_height, self.camera)

    def menu_window(self):
        imgui.set_next_window_size(self.menu_window_width, self.menu_window_height)
        imgui.set_next_window_position(*self.menu_window_pos)
        imgui.begin("Options", flags=imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_COLLAPSE)
        self.menu_window_focused = imgui.is_window_focused()
        if imgui.begin_menu_bar():
            if imgui.begin_menu(label="Menu"):
                imgui.end_menu()
        if imgui.button(label="Load Mesh"):
            file_dialog.file_dialog_open()
        imgui.same_line()
        if imgui.button(label="Save Mesh"):
            file_dialog.file_dialog_save_mesh(self.mesh)
        imgui.end()

    def log_window(self):
        imgui.set_next_window_size(self.log_window_width, self.log_window_height)
        imgui.set_next_window_position(*self.log_window_pos)
        imgui.begin("Program", flags=imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_COLLAPSE)
        self.log_window_focused = imgui.is_window_focused()
        imgui.text_colored("Log", 1, 1, 0)
        imgui.begin_child("Scrolling")
        with open("femflow.log", "r+") as f:
            for line in f.readlines():
                imgui.text(line)
        imgui.set_scroll_y(imgui.get_scroll_max_y())
        imgui.end_child()
        imgui.end()

    def launch(self):
        folder = os.path.dirname(os.path.abspath(__file__))
        tex = Texture.from_file(f"{folder}/assets/cube_texture.jpg")
        self.mesh = Mesh.from_file(f"{folder}/models/cube.obj")
        self.mesh.textures = tex
        self.renderer = Renderer(self.mesh)
        self.camera.resize(self.window_width, self.window_height)
        self.renderer.resize(self.window_width, self.window_height, self.camera)
        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            self.imgui_impl.process_inputs()

            imgui.new_frame()
            self.menu_window()
            self.log_window()

            self.renderer.render(self.camera)
            imgui.render()
            self.imgui_impl.render(imgui.get_draw_data())

            glfw.swap_buffers(self.window)
        self.renderer.destroy()

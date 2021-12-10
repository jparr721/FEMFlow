import os

import glfw
import imgui
from imgui.integrations.glfw import GlfwRenderer
from loguru import logger
from OpenGL.GL import *
from utils.filesystem import file_dialog

from .camera import Camera
from .input import Input
from .mesh import Mesh, Texture
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

        imgui.create_context()
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        logger.debug("Drawing window")

        self.window = glfw.create_window(self.WINDOW_WIDTH, self.WINDOW_HEIGHT, self.WINDOW_TITLE, None, None)

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

    def scroll_callback(self, window, xoffset, yoffset):
        self.input.scroll_event(window, xoffset, yoffset, self.camera)

    def mouse_callback(self, window, button, action, mods):
        self.input.handle_mouse(window, button, action, mods, self.camera)

    def mouse_move_callback(self, window, xpos, ypos):
        self.input.handle_mouse_move(window, xpos, ypos, self.camera)

    def window_size_callback(self, window, width, height):
        self.camera.resize(width, height)
        self.imgui_impl.io.display_size = width, height

        if self.renderer is not None:
            self.renderer.resize(width, height, self.camera)

    def menu_window(self):
        imgui.begin("FEMFlow Options")
        if imgui.begin_menu_bar():
            if imgui.begin_menu(label="Menu"):
                imgui.end_menu()

        if imgui.button(label="Load Mesh"):
            file_dialog.file_dialog_open()
        imgui.same_line()
        if imgui.button(label="Save Mesh"):
            file_dialog.file_dialog_save_mesh(self.mesh)
        imgui.end()

    def launch(self):
        folder = os.path.dirname(os.path.abspath(__file__))
        tex = Texture.from_file(f"{folder}/assets/cube_texture.jpg")
        self.mesh = Mesh.from_file(f"{folder}/models/cube.obj")
        self.renderer = Renderer(self.mesh)
        self.camera.resize(self.WINDOW_WIDTH, self.WINDOW_HEIGHT)
        self.renderer.resize(self.WINDOW_WIDTH, self.WINDOW_HEIGHT, self.camera)
        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            self.imgui_impl.process_inputs()

            imgui.new_frame()
            self.menu_window()

            self.renderer.render(self.camera)
            imgui.render()
            self.imgui_impl.render(imgui.get_draw_data())

            glfw.swap_buffers(self.window)
        self.renderer.destroy()

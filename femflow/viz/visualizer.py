import os
from typing import Callable

import glfw
import imgui
from imgui.integrations.glfw import GlfwRenderer
from loguru import logger
from OpenGL.GL import *
from utils.filesystem import file_dialog

from .camera import Camera
from .input import Input
from .mesh import Mesh
from .renderer import Renderer

logger.add("femflow.log", mode="w+")


class Visualizer(object):
    def __init__(
        self,
        callback_sim_parameters: Callable = None,
        callback_start_sim_button_pressed: Callable = None,
        callback_reset_sim_button_pressed: Callable = None,
    ):
        self.WINDOW_TITLE = "FEMFlow Viewer"
        self.window_width = 1200
        self.window_height = 800
        self.background_color = [1, 1, 1, 0]

        self.camera = Camera()
        self.camera.resize(self.window_width, self.window_height)

        # IMGUI
        self.log_window_focused = False
        self.menu_window_focused = False

        # Parameter-Specific Menus
        self.sim_parameters_expanded = True
        self.sim_parameters_visible = True

        self.callback_sim_parameters = (
            self.placeholder_sim_param_menu if callback_sim_parameters is None else callback_sim_parameters
        )
        self.callback_start_sim_button_pressed = (
            lambda: logger.error("No functionality implemented yet!")
            if callback_start_sim_button_pressed is None
            else callback_start_sim_button_pressed
        )
        self.callback_reset_sim_button_pressed = (
            lambda: logger.error("No functionality implemented yet!")
            if callback_reset_sim_button_pressed is None
            else callback_reset_sim_button_pressed
        )
        # IMGUI

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

    @property
    def log_window_dimensions(self):
        height = self.window_height * 0.2 if self.window_height >= 800 else 160
        return (
            self.window_width,
            height,
            0,
            self.window_height - height,
        )

    @property
    def menu_window_dimensions(self):
        _, log_window_height, _, _ = self.log_window_dimensions
        return (
            self.window_width * 0.15 if self.window_width >= 800 else 130,
            self.window_height - log_window_height,
            0,
            0,
        )

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
        menu_window_width, menu_window_height, menu_window_x, menu_window_y = self.menu_window_dimensions
        imgui.set_next_window_size(menu_window_width, menu_window_height)
        imgui.set_next_window_position(menu_window_x, menu_window_y)

        imgui.begin("Options", flags=imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_COLLAPSE)
        self.menu_window_focused = imgui.is_window_focused()
        if imgui.button(label="Load Mesh"):
            path = file_dialog.file_dialog_open()
            self.mesh = Mesh.from_file(path)
            self.renderer = Renderer(self.mesh)
            self.renderer.resize(self.window_width, self.window_height, self.camera)
        imgui.same_line()
        if imgui.button(label="Save Mesh"):
            file_dialog.file_dialog_save_mesh(self.mesh)
        self.sim_param_menu()
        imgui.end()

    def log_window(self):
        log_window_width, log_window_height, log_window_x, log_window_y = self.log_window_dimensions
        imgui.set_next_window_size(log_window_width, log_window_height)
        imgui.set_next_window_position(log_window_x, log_window_y)

        imgui.begin(
            "Logs",
            flags=imgui.WINDOW_NO_MOVE
            | imgui.WINDOW_NO_RESIZE
            | imgui.WINDOW_NO_COLLAPSE
            | imgui.WINDOW_HORIZONTAL_SCROLLING_BAR,
        )
        self.log_window_focused = imgui.is_window_focused() or imgui.is_item_clicked()
        imgui.begin_child("Scrolling")
        with open("femflow.log", "r+") as f:
            for line in f.readlines():
                imgui.text(line)
        imgui.set_scroll_y(imgui.get_scroll_max_y())
        imgui.end_child()
        imgui.end()

    def sim_param_menu(self):
        if not self.sim_parameters_visible:
            self.sim_parameters_visible = True
        self.sim_parameters_expanded, self.sim_parameters_visible = imgui.collapsing_header(
            "Parameters", self.sim_parameters_visible, imgui.TREE_NODE_DEFAULT_OPEN
        )
        if self.sim_parameters_expanded:
            self.callback_sim_parameters()
            if imgui.button(label="Start Sim"):
                self.callback_start_sim_button_pressed()
            imgui.same_line()
            if imgui.button(label="Reset Sim"):
                self.callback_reset_sim_button_pressed()

    def placeholder_sim_param_menu(self):
        imgui.text("Settings Here")

    def launch(self):
        folder = os.path.dirname(os.path.abspath(__file__))
        self.mesh = Mesh.from_file(f"{folder}/models/cube.obj")
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

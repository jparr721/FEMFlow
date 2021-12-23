import os
import threading

import glfw
import igl
import imgui
import numpy as np
from femflow.meshing.implicit import gyroid
from femflow.numerics.bintensor3 import bintensor3
from femflow.reconstruction.behavior_matching import BehaviorMatching
from femflow.simulation.environment import Environment
from femflow.utils.filesystem import file_dialog
from imgui.integrations.glfw import GlfwRenderer
from loguru import logger
from OpenGL.GL import *

from .camera import Camera
from .input import Input
from .mesh import Mesh
from .renderer import Renderer

RED = [1, 0, 0]
GREEN = [0, 1, 0]


class Visualizer(object):
    def __init__(self, environment: Environment):
        self.WINDOW_TITLE = "FEMFlow Viewer"
        self.window_width = 1800
        self.window_height = 1200
        self.background_color = [1, 1, 1, 0]

        self.camera = Camera()
        self.camera.resize(self.window_width, self.window_height)

        # IMGUI
        self.log_window_focused = False
        self.menu_window_focused = False
        self.capture_window_focused = False

        # Simulation-Specific Menus
        self.simulation_window_focused = False
        self.current_timestep = 0
        self.n_timesteps = 100

        # Parameter-Specific Menus
        self.sim_parameters_expanded = True
        self.sim_parameters_visible = True
        self.behavior_matching_expanded = True
        self.behavior_matching_visible = True
        self.simulation_spec_expanded = True
        self.simulation_spec_visible = False

        self.capture_window_visible = False

        self.callback_environment_loader = lambda: logger.error("No functionality implemented yet!")
        self.callback_start_sim_button_pressed = lambda: logger.error("No functionality implemented yet!")
        self.callback_reset_sim_button_pressed = lambda: logger.error("No functionality implemented yet!")

        self.simulation_environment: Environment = environment

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

        self.behavior_matching: BehaviorMatching = BehaviorMatching()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.info("Destroying imgui")
        self.imgui_impl.shutdown()

        logger.info("Destorying behavior matching")
        self.behavior_matching.destroy()

        logger.info("Destroying glfw")
        glfw.terminate()

    @property
    def any_window_focused(self):
        return (
            self.menu_window_focused
            or self.log_window_focused
            or self.simulation_window_focused
            or self.capture_window_focused
        )

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
    def simulation_window_dimensions(self):
        height = self.window_height * 0.12 if self.window_height >= 800 else 130
        menu_window_width, _, _, _ = self.menu_window_dimensions
        width = self.window_width - menu_window_width
        return (
            width,
            height,
            menu_window_width,
            0,
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
            self.mesh.tetrahedralize()
            self.renderer = Renderer(self.mesh)
            self.renderer.resize(self.window_width, self.window_height, self.camera)
        imgui.same_line()
        if imgui.button(label="Save Mesh"):
            file_dialog.file_dialog_save_mesh(self.mesh)
        self.sim_param_menu()
        self.capture_param_menu()
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
        imgui.set_scroll_y(imgui.get_scroll_max_y() + 2)
        imgui.end_child()
        imgui.end()

    def sim_param_menu(self):
        self.sim_parameters_expanded, self.sim_parameters_visible = imgui.collapsing_header(
            "Parameters", self.sim_parameters_visible
        )
        if self.sim_parameters_expanded:
            imgui.text_colored(
                f"Sim Status: {'Not Loaded' if not self.simulation_environment.loaded else 'Loaded'}",
                *(RED if not self.simulation_environment.loaded else GREEN),
            )
            self.simulation_environment.menu()
            if imgui.button(label="Load"):
                threading.Thread(target=self.simulation_environment.load, args=(self.mesh,)).start()

            self.simulation_spec_visible = self.simulation_environment.loaded

            self.simulation_spec_expanded, self.simulation_spec_visible = imgui.collapsing_header(
                "Sim Config", self.simulation_spec_visible, imgui.TREE_NODE_DEFAULT_OPEN
            )
            if self.simulation_spec_expanded:
                imgui.text("Timesteps")
                _, self.n_timesteps = imgui.input_int("##timesteps", self.n_timesteps)

                if imgui.button(label="Start Sim"):
                    self.sim_runtime_focused = True
                    self.start_simulation()
                imgui.same_line()
                if imgui.button(label="Reset Sim"):
                    self.reset_simulation()

    def capture_param_menu(self):
        self.behavior_matching_expanded, self.behavior_matching_visible = imgui.collapsing_header(
            "Behavior Match", self.behavior_matching_visible
        )
        if self.behavior_matching_expanded:
            if imgui.button(label="Calibrate"):
                if self.behavior_matching.streaming:
                    self.behavior_matching.stop_matching()
                self.behavior_matching.calibrate()

            _, self.capture_window_visible = imgui.checkbox("Capturing", self.capture_window_visible)
            if self.capture_window_visible:
                self.capture_window()
                self.behavior_matching.start_matching()
            else:
                self.behavior_matching.stop_matching()

            if self.behavior_matching.streaming:
                if imgui.button(label="Capture Shape"):
                    mask = np.clip(self.behavior_matching.mask, 0, 1)
                    scalar_field = gyroid(0.2, 60)
                    scalar_field = bintensor3(scalar_field)
                    scalar_field.padding(0)
                    scalar_field.padding(1)
                    scalar_field.padding(2)
                    v, f = scalar_field.tomesh()
                    igl.write_obj("out.obj", v, f)

    def capture_window(self):
        imgui.begin("Capture Window")
        self.capture_window_focused = imgui.is_window_focused() or imgui.is_item_clicked()
        imgui.end()

    def simulation_window(self):
        (
            simulation_window_width,
            simulation_window_height,
            simulation_window_x,
            simulation_window_y,
        ) = self.simulation_window_dimensions
        imgui.set_next_window_size(simulation_window_width, simulation_window_height)
        imgui.set_next_window_position(simulation_window_x, simulation_window_y)

        imgui.begin(
            "Simulation",
            flags=imgui.WINDOW_NO_MOVE
            | imgui.WINDOW_NO_RESIZE
            | imgui.WINDOW_NO_COLLAPSE
            | imgui.WINDOW_HORIZONTAL_SCROLLING_BAR,
        )
        self.simulation_window_focused = imgui.is_window_focused() or imgui.is_item_clicked()
        imgui.text("Timestep")
        imgui.text_colored(
            "No Displacements, Please Start Sim First."
            if len(self.simulation_environment.displacements) < self.n_timesteps
            else "Displacements Ready",
            *(RED if len(self.simulation_environment.displacements) < self.n_timesteps else GREEN),
        )
        imgui.push_item_width(-1)
        _, self.current_timestep = imgui.slider_int(
            "##timestep", self.current_timestep, min_value=0, max_value=self.n_timesteps
        )
        self.mesh.transform(self.simulation_environment.displacements[self.current_timestep])
        imgui.pop_item_width()
        imgui.end()

    def placeholder_sim_param_menu(self):
        imgui.text("Settings Here")

    def start_simulation(self):
        if self.simulation_environment is None:
            logger.error("Sim environment is empty, cannot start sim")
            return

        logger.info("Running simulation")
        threading.Thread(target=self.simulation_environment.simulate, args=(self.mesh, self.n_timesteps)).start()

    def reset_simulation(self):
        if self.simulation_environment is None:
            logger.error("Sim environment is empty, cannot reset sim")
            return

        self.simulation_environment.reset(self.mesh)
        self.current_timestep = 0

        logger.success("Simulation was reset")

    def launch(self):
        folder = os.path.dirname(os.path.abspath(__file__))
        self.mesh = Mesh.from_file(f"{folder}/models/cube.obj")
        self.mesh.tetrahedralize()
        self.renderer = Renderer(self.mesh)
        self.camera.resize(self.window_width, self.window_height)
        self.renderer.resize(self.window_width, self.window_height, self.camera)
        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            self.imgui_impl.process_inputs()

            imgui.new_frame()
            self.menu_window()
            self.log_window()

            # TODO(@jparr721) This variable is mis-named
            if self.simulation_spec_visible:
                self.simulation_window()

            self.renderer.render(self.camera)
            imgui.render()
            self.imgui_impl.render(imgui.get_draw_data())

            glfw.swap_buffers(self.window)
        self.renderer.destroy()

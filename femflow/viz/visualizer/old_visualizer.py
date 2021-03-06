import copy
import os
import threading
from typing import Dict, Iterable, Union

import glfw
import imgui
from imgui.integrations.glfw import GlfwRenderer
from loguru import logger
from OpenGL.GL import *

from femflow.meshing.implicit import gyroid
from femflow.numerics.bintensor3 import bintensor3
from femflow.reconstruction.behavior_matching import BehaviorMatching
from femflow.simulation.galerkin_optimizer import GalerkinOptimizer
from femflow.simulation.linear_fem_simulation import LinearFemSimulation

from .. import models
from ..camera import Camera
from ..input import Input
from ..mesh import Mesh
from ..renderer import Renderer, RenderMode
from ._builtin import (
    LogWindow,
    MenuWindow,
    ShapeCaptureConfigMenu,
    ShapeCaptureExperimentMenu,
    ShapeCaptureWindow,
    SimParametersMenu,
    SimulationConfigMenu,
    SimulationWindow,
)
from .visualizer_window import VisualizerWindow


class Visualizer(object):
    def __init__(self, environment: LinearFemSimulation):
        self.WINDOW_TITLE = "FEMFlow Viewer"
        self.background_color = [1, 1, 1, 0]

        self.camera = Camera()
        self.mesh: Mesh = Mesh.from_file(os.path.join(models.model_paths(), "cube.obj"))
        self.mesh.tetrahedralize()
        self.renderer: Renderer = None

        self.callback_environment_loader = lambda: logger.error(
            "No functionality implemented yet!"
        )
        self.callback_start_sim_button_pressed = lambda: logger.error(
            "No functionality implemented yet!"
        )
        self.callback_reset_sim_button_pressed = lambda: logger.error(
            "No functionality implemented yet!"
        )

        self.simulation_environment: LinearFemSimulation = environment
        self.sim_parameter_state = dict()

        assert glfw.init(), "GLFW is not initialized!"

        logger.info("GLFW Initialized.")

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

        logger.debug("Drawing window")

        self.window = glfw.create_window(
            self.window_width, self.window_height, self.WINDOW_TITLE, None, None
        )

        assert self.window, "GLFW failed to open the window"

        glfw.make_context_current(self.window)
        self.imgui_impl = GlfwRenderer(self.window, attach_callbacks=True)

        glfw.set_mouse_button_callback(self.window, self.mouse_callback)
        glfw.set_scroll_callback(self.window, self.scroll_callback)
        glfw.set_cursor_pos_callback(self.window, self.mouse_move_callback)
        glfw.set_window_size_callback(self.window, self.window_size_callback)

        glClearColor(*self.background_color)

        self.input = Input()

        # TODO(@jparr721) Everything below this should be abstracted later on.
        self.behavior_matching: BehaviorMatching = BehaviorMatching()

        # BUILTIN WINDOWS
        self.windows: Dict[str, VisualizerWindow] = dict()
        self.add_window(LogWindow())
        self.add_window(MenuWindow())
        self.add_window(ShapeCaptureWindow())
        self.add_window(SimulationWindow())
        self._init_builtins()

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
        return any([window.focused for window in self.windows.values()])

    def add_window(self, windows: Union[VisualizerWindow, Iterable[VisualizerWindow]]):
        if isinstance(windows, Iterable):
            for window in windows:
                self.add_window(window)
        elif isinstance(windows, VisualizerWindow):
            self.windows[windows.name] = windows
        else:
            raise TypeError(
                f"Windows must be iterable or window type, got {type(windows)}"
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
        self._resize_windows()

        if self.renderer is not None:
            self.renderer.resize(self.window_width, self.window_height, self.camera)

    def start_simulation(self):
        if self.simulation_environment is None:
            logger.error("Sim environment is empty, cannot start sim")
            return

        logger.info("Running simulation")
        if self.sim_parameter_state["current_timestep"] > 0:
            self.sim_parameter_state["current_timestep"] = 0
        threading.Thread(
            target=self.simulation_environment.simulate,
            args=(self.sim_parameter_state["n_timesteps"],),
            daemon=True,
        ).start()

    def run_static_simulation(self):
        if self.simulation_environment is None:
            logger.error("Sim environment is empty, cannot start sim")
            return

        logger.info("Running static form")
        threading.Thread(
            target=self.simulation_environment.solve_static, daemon=True
        ).start()

    def reset_simulation(self):
        if self.simulation_environment is None:
            logger.error("Sim environment is empty, cannot reset sim")
            return

        self.simulation_environment.reset(self.mesh)
        self.current_timestep = 0

        logger.success("Simulation was reset")

    def launch(self):
        self.mesh.tetrahedralize()
        self.renderer = Renderer(self.mesh)
        self.camera.resize(self.window_width, self.window_height)
        self.renderer.resize(self.window_width, self.window_height, self.camera)
        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            self.imgui_impl.process_inputs()

            imgui.new_frame()
            self._render_builtins()

            self.renderer.render(self.camera)
            imgui.render()
            self.imgui_impl.render(imgui.get_draw_data())

            glfw.swap_buffers(self.window)
        self.renderer.destroy()

    def _resize_windows(self):
        # Logs
        logs = self.windows["Logs"]
        height = self.window_height * 0.2 if self.window_height >= 800 else 160
        logs.dimensions = (self.window_width, height)
        logs.position = (0, self.window_height - height)

        # Menu
        menu = self.windows["Menu"]
        _, height = logs.dimensions
        menu.dimensions = (
            self.window_width * 0.15 if self.window_width >= 800 else 130,
            self.window_height - height,
        )

        # Simulation
        sim = self.windows["Simulation"]
        height = 150
        menu_window_width, _ = menu.dimensions
        width = self.window_width - menu_window_width
        sim.dimensions = (width - menu.dimensions[0], height)
        sim.position = (menu_window_width, 0)

        # Shape Capture
        shape_capture_experiment = ShapeCaptureExperimentMenu()
        shape_capture = self.windows["Shape Capture"]
        shape_capture.dimensions = copy.deepcopy(menu.dimensions)
        shape_capture.position = (self.window_width - menu.dimensions[0], 0)
        shape_capture.add_menu(shape_capture_experiment)

    def _init_builtins(self):
        self._resize_windows()
        sim_params_menu = SimParametersMenu()
        sim_config_menu = SimulationConfigMenu()
        sim_params_menu.add_submenu(sim_config_menu)

        menu = self.windows["Menu"]
        shape_capture_config_menu = ShapeCaptureConfigMenu()
        menu.add_menu([sim_params_menu, shape_capture_config_menu])

    def _render_builtins(self):
        self._extract_input_vars()
        logs = self.windows["Logs"]
        menu = self.windows["Menu"]
        sim = self.windows["Simulation"]
        shape_capture = self.windows["Shape Capture"]

        def sim_params_menu_load_button_cb():
            self.simulation_environment.loaded = False
            self.sim_parameter_state["current_timestep"] = 0
            self.simulation_environment.load(self.mesh)

        def static_sim_button_cb():
            menu.menus["Sim Params"].submenus["Sim Config"].n_timesteps = 1
            self.run_static_simulation()

        def sim_params_menu_sim_environment_menu_cb():
            self.simulation_environment.menu()

        def timestep_changed_cb(t):
            if len(self.simulation_environment.displacements) == 0:
                return
            if self.simulation_environment.loaded:
                self.mesh.transform(self.simulation_environment.displacements[t])

        def generate_geometry_cb(radius: float, thickness: float, size: int):
            scalar_field = gyroid(radius, size)
            scalar_field = bintensor3(scalar_field)
            scalar_field.padding(0)
            scalar_field.padding(1)
            scalar_field.padding(2)
            self.mesh.reload_from_surface(*scalar_field.tomesh(thickness))
            logger.info("Created gyroid, tetrahedralizing")
            self.mesh.tetrahedralize()

        def compute_coefficients_cb(load: float, strain_pct: float, height: int):
            target_height = ((100 - strain_pct) / 100) * height
            galerkin_optimizer = GalerkinOptimizer(
                self.mesh, -load, target_height, learning_rate=1
            )
            galerkin_optimizer.train()

        def set_sim_render_mode(render_mode: int):
            self.renderer.render_mode = RenderMode(render_mode)

        logs()
        menu(
            mesh=self.mesh,
            load_button_cb=sim_params_menu_load_button_cb,
            sim_environment_menu_cb=sim_params_menu_sim_environment_menu_cb,
            sim_status=self.simulation_environment.loaded,
            behavior_matching_streaming=self.behavior_matching.streaming,
            start_sim_button_cb=self.start_simulation,
            reset_sim_button_cb=self.reset_simulation,
            static_sim_button_cb=static_sim_button_cb,
            render_mode_cb=set_sim_render_mode,
        )
        sim(
            sim_status=len(self.simulation_environment.displacements) > 1,
            timestep_changed_cb=timestep_changed_cb,
            max_timesteps=menu.menus["Sim Params"].submenus["Sim Config"].n_timesteps,
        )
        if self.sim_parameter_state["capturing"]:
            self.behavior_matching.start_streaming()
            shape_capture(
                generate_geometry_cb=generate_geometry_cb,
                compute_coefficients_cb=compute_coefficients_cb,
            )
        else:
            self.behavior_matching.stop_streaming()

    def _extract_input_vars(self):
        for window in self.windows.values():
            for key in window.attr_keys:
                self.sim_parameter_state[key] = window.__dict__[key]
            for menu in window.menus.values():
                for key in menu.attr_keys:
                    self.sim_parameter_state[key] = menu.__dict__[key]
                for submenu in menu.submenus.values():
                    for key in submenu.attr_keys:
                        self.sim_parameter_state[key] = submenu.__dict__[key]

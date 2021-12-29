import threading
from typing import Callable, List

import imgui
from femflow.utils.filesystem import file_dialog
from femflow.viz.mesh import Mesh
from loguru import logger

from .visualizer_core import Colors


class VisualizerMenus(object):
    def __init__(self, window_height: int, window_width: int, windows: List[Callable] = [], menus: List[Callable] = []):
        self.windows = windows
        self.menus = menus

        # Windows
        # Log Window
        self.log_window_focused = False

        # Menu Window
        self.menu_window_focused = False

        # Capture Window
        self.capture_window_focused = False

        # Simulation Window
        self.simulation_window_focused = False

        # Menus
        # Simulation Parameters
        self.current_timestep = 0
        self.n_timesteps = 100

        # Behavior Matching
        self.behavior_matching_menu_expanded = True
        self.behavior_matching_menu_visible = True

        # Simulation Specification
        self.simulation_spec_menu_expanded = True
        self.simulation_spec_menu_visible = True

        # Simulation Parameters
        self.simulation_parameters_menu_expanded = True
        self.simulation_parameters_menu_visible = True

        # Meshing
        self.meshing_menu_expanded = True
        self.meshing_menu_visible = True

        # Overall Window
        self.window_height = window_height
        self.window_width = window_width

    @property
    def any_window_focused(self):
        return (
            self.log_window_focused
            or self.menu_window_focused
            or self.capture_window_focused
            or self.simulation_window_focused
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

    @property
    def capture_window_dimensions(self):
        _, log_window_height, _, _ = self.log_window_dimensions
        width = self.window_width * 0.15 if self.window_width >= 800 else 130
        return (
            width,
            self.window_height - log_window_height,
            self.window_width - width,
            0,
        )

    def menu_window(self, mesh: Mesh):
        menu_window_width, menu_window_height, menu_window_x, menu_window_y = self.menu_window_dimensions
        imgui.set_next_window_size(menu_window_width, menu_window_height)
        imgui.set_next_window_position(menu_window_x, menu_window_y)

        imgui.begin("Options", flags=imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_COLLAPSE)
        self.menu_window_focused = imgui.is_window_focused()
        if imgui.button(label="Load Mesh"):
            filename = file_dialog.file_dialog_open()
            mesh.reload_from_file(filename)
        imgui.same_line()
        if imgui.button(label="Save Mesh"):
            file_dialog.file_dialog_save_mesh(mesh)
        self.sim_param_menu()
        self.capture_param_menu()

        self.mesh_parameters_expanded, self.mesh_parameters_visible = imgui.collapsing_header("Mesh Parameters")
        if self.mesh_parameters_expanded:
            pass

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
        self.simulation_parameters_menu_expanded, self.simulation_parameters_menu_visible = imgui.collapsing_header(
            "Parameters", self.sim_parameters_visible
        )
        if self.sim_parameters_expanded:
            imgui.text_colored(
                f"Sim Status: {'Not Loaded' if not self.simulation_environment.loaded else 'Loaded'}",
                *(Colors.red if not self.simulation_environment.loaded else Colors.green),
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

    def _no_op_menu(self):
        logger.error("No functionality implemented yet!")

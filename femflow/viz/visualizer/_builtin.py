from typing import Callable, List

import imgui
from femflow.utils.filesystem import file_dialog

from ..mesh import Mesh
from . import ui_colors
from .visualizer_menu import VisualizerMenu
from .visualizer_window import VisualizerWindow

_IMGUIFLAGS_NOCLOSE_NOMOVE_NORESIZE = [imgui.WINDOW_NO_MOVE, imgui.WINDOW_NO_RESIZE, imgui.WINDOW_NO_COLLAPSE]
_IMGUIFLAGS_TREENODE_OPEN = [imgui.TREE_NODE_DEFAULT_OPEN]


class ShapeCaptureConfigMenu(VisualizerMenu):
    def __init__(self, name="Shape Matching", flags: List[int] = _IMGUIFLAGS_TREENODE_OPEN):
        super().__init__(name, flags)
        self._register_input("capturing", False)

    def render(self, **kwargs) -> None:
        behavior_matching_streaming = self._unpack_kwarg("behavior_matching_streaming", bool, **kwargs)
        if imgui.button(label="Calibrate"):
            calibrate_button_cb: Callable = self._unpack_kwarg("calibrate_button_cb", callable, **kwargs)
            calibrate_button_cb()

        self._generate_imgui_input("capturing", imgui.checkbox, use_key_as_label=True)

        if behavior_matching_streaming:
            if imgui.button(label="Capture Shape"):
                capture_shape_button_cb: Callable = self._unpack_kwarg("capture_shape_button_cb", callable, **kwargs)
                capture_shape_button_cb()


class SimulationConfigMenu(VisualizerMenu):
    def __init__(self, name="Sim Config", flags: List[int] = _IMGUIFLAGS_TREENODE_OPEN):
        super().__init__(name, flags)
        self._register_input("n_timesteps", 100)

    def render(self, **kwargs) -> None:
        sim_status = self._unpack_kwarg("sim_status", bool, **kwargs)
        self.visible = sim_status
        imgui.text("Timestamps")
        self._generate_imgui_input("n_timesteps", imgui.input_int, step=50)

        if imgui.button(label="Start Sim"):
            start_sim_button_cb: Callable = self._unpack_kwarg("start_sim_button_cb", callable, **kwargs)
            start_sim_button_cb()
        imgui.same_line()  # Maybe we can parameterize the layouts at some point.
        if imgui.button(label="Reset Sim"):
            reset_sim_button_cb: Callable = self._unpack_kwarg("reset_sim_button_cb", callable, **kwargs)
            reset_sim_button_cb()


class SimParametersMenu(VisualizerMenu):
    def __init__(self, *, name: str = "Sim Params", flags: List[int] = _IMGUIFLAGS_TREENODE_OPEN):
        super().__init__(name, flags)

    def render(self, **kwargs) -> None:
        sim_environment_menu_cb: Callable = self._unpack_kwarg("sim_environment_menu_cb", callable, **kwargs)
        sim_status: bool = self._unpack_kwarg("sim_status", bool, **kwargs)

        sim_status_text = ("Sim Status: Not Loaded", *ui_colors.error)
        if sim_status:
            sim_status_text = ("Sim Status: Loaded", *ui_colors.success)

        imgui.text_colored(*sim_status_text)

        sim_environment_menu_cb()

        if imgui.button(label="Load"):
            load_button_cb: Callable = self._unpack_kwarg("load_button_cb", callable, **kwargs)
            load_button_cb()


class LogWindow(VisualizerWindow):
    def __init__(self, *, name="Logs", flags: List[int] = _IMGUIFLAGS_NOCLOSE_NOMOVE_NORESIZE):
        super().__init__(name, flags)

    def render(self, **kwargs) -> None:
        imgui.begin_child("Scrolling")
        with open("femflow.log", "r+") as f:
            for line in f.readlines():
                imgui.text(line)
        imgui.set_scroll_y(imgui.get_scroll_max_y() + 2)
        imgui.end_child()


class MenuWindow(VisualizerWindow):
    def __init__(self, *, name="Menu", flags: List[int] = _IMGUIFLAGS_NOCLOSE_NOMOVE_NORESIZE):
        super().__init__(name, flags)

    def render(self, **kwargs) -> None:
        self.menu_window_focused = imgui.is_window_focused()
        if imgui.button(label="Load Mesh"):
            mesh = self._unpack_kwarg("mesh", Mesh, **kwargs)
            filename = file_dialog.file_dialog_open()
            mesh.reload_from_file(filename)
        imgui.same_line()
        if imgui.button(label="Save Mesh"):
            mesh = self._unpack_kwarg("mesh", Mesh, **kwargs)
            file_dialog.file_dialog_save_mesh(mesh)


class SimulationWindow(VisualizerWindow):
    def __init__(self, *, name="Simulation", flags: List[int] = _IMGUIFLAGS_NOCLOSE_NOMOVE_NORESIZE):
        super().__init__(name, flags)
        self._register_input("current_timestep", 0)

    def render(self, **kwargs) -> None:
        imgui.text("Timestep")

        max_timesteps = self._unpack_kwarg("max_timesteps", int, **kwargs)
        sim_status = self._unpack_kwarg("sim_status", bool, **kwargs)
        timestep_changed_cb = self._unpack_kwarg("timestep_changed_cb", callable, **kwargs)

        displacements_text = ("No displacements, please start sim first", *ui_colors.error)

        if sim_status:
            displacements_text = ("Displacements ready", *ui_colors.success)

        imgui.text_colored(*displacements_text)
        imgui.push_item_width(-1)
        self._generate_imgui_input("current_timestep", imgui.slider_int, min_value=0, max_value=max_timesteps)
        timestep_changed_cb(self.current_timestep)
        imgui.pop_item_width()


class ShapeCaptureWindow(VisualizerWindow):
    def __init__(self, *, name="Shape Capture", flags: List[int] = _IMGUIFLAGS_NOCLOSE_NOMOVE_NORESIZE):
        super().__init__(name, flags)

    def render(self, **kwargs) -> None:
        radius_converged = self._unpack_kwarg("radius_converged", bool, **kwargs)
        thickness_converged = self._unpack_kwarg("thickness_converged", bool, **kwargs)

        radius = self._unpack_kwarg("radius", float, **kwargs)
        thickness = self._unpack_kwarg("thickness", float, **kwargs)

        radius_text = (f"Radius: {radius}", *(ui_colors.success if radius_converged else ui_colors.error))
        thickness_text = (f"Thickness: {thickness}", *(ui_colors.success if thickness_converged else ui_colors.error))

        imgui.text_colored(*radius_text)
        imgui.text_colored(*thickness_text)

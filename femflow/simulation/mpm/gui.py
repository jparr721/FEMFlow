from typing import Callable, List

import imgui

from femflow.numerics.linear_algebra import matrix_to_vector
from femflow.simulation.mpm.simulation import MPMSimulation
from femflow.viz.mesh import Mesh
from femflow.viz.visualizer.visualizer_menu import VisualizerMenu
from femflow.viz.visualizer.visualizer_window import VisualizerWindow


class MPMSimulationMeshMenu(VisualizerMenu):
    def __init__(
        self, name="MPM Mesh", flags: List[int] = [imgui.TREE_NODE_DEFAULT_OPEN]
    ):
        super().__init__(name, flags)
        self.mesh_options = ["gyroid", "diamond", "primitive"]
        self.mesh_type = 0
        self.resolution = 30
        self.k = 0.3
        self.t = 0.3

    def render(self, **kwargs) -> None:
        mesh = self._unpack_kwarg("mesh", Mesh, **kwargs)
        imgui.text(f"Points {mesh.vertices.size}")


class MPMSimulationConfigMenu(VisualizerMenu):
    def __init__(
        self, name="Sim Config", flags: List[int] = [imgui.TREE_NODE_DEFAULT_OPEN]
    ):
        super().__init__(name, flags)
        self.n_timesteps = 500

    def render(self, **kwargs) -> None:
        imgui.text("Timestamps")
        self._generate_imgui_input("n_timesteps", imgui.input_int, step=100)

        sim = self._unpack_kwarg("sim", MPMSimulation, **kwargs)

        if imgui.button(label="Start Sim"):
            start_sim_button_cb: Callable = self._unpack_kwarg(
                "start_sim_button_cb", callable, **kwargs
            )
            start_sim_button_cb(n_timesteps=self.n_timesteps)

        imgui.same_line()

        if imgui.button(label="Reset Sim"):
            mesh = self._unpack_kwarg("mesh", Mesh, **kwargs)
            reset_sim_button_cb: Callable = self._unpack_kwarg(
                "reset_sim_button_cb", callable, **kwargs
            )
            reset_sim_button_cb(mesh)

        status = self._unpack_kwarg("sim_status", bool, **kwargs)

        if status:
            imgui.text("Sim Running")
        else:
            imgui.text("Sim Not Running")


class MPMDisplacementsWindow(VisualizerWindow):
    def __init__(self):
        name = "Displacements"
        flags = [imgui.TREE_NODE_DEFAULT_OPEN]
        super().__init__(name, flags)

        self._register_input("current_timestep", 0)

    def render(self, **kwargs) -> None:
        sim: MPMSimulation = self._unpack_kwarg("sim", MPMSimulation, **kwargs)
        mesh: Mesh = self._unpack_kwarg("mesh", Mesh, **kwargs)
        imgui.push_item_width(-1)
        self._generate_imgui_input(
            "current_timestep",
            imgui.slider_int,
            min_value=0,
            max_value=len(sim.displacements) - 1,
        )
        imgui.pop_item_width()

        if sim.loaded:
            sim.mesh.replace(matrix_to_vector(sim.displacements[self.current_timestep]))

        if imgui.button("Run"):
            sim.load(mesh=mesh)
            sim.start()

    def resize(self, parent_width: int, parent_height: int, **kwargs):
        self.dimensions = (
            parent_width,
            100,
        )
        self.position = (0, 0)


# TODO CHANGE TO A WINDOW TYPE
class BehaviorMatchingMenu(VisualizerMenu):
    def __init__(
        self,
        name: str = "Behavior Capture",
        flags: List[int] = [imgui.TREE_NODE_DEFAULT_OPEN],
    ):
        super().__init__(name, flags)
        self.capturing = False
        self.applied_load = 0.0
        self.bhm = BehaviorMatching()

    def render(self, **kwargs) -> None:
        self._generate_imgui_input("capturing", imgui.checkbox, use_key_as_label=True)

        if self.capturing:
            self.bhm.start_streaming()
        else:
            if self.bhm.streaming:
                self.bhm.stop_streaming()

        if self.capturing:
            if imgui.button("Set Start"):
                self.bhm.set_starting_dimensions()

            imgui.text(f"Height Difference: {self.bhm.height_diff}")
            imgui.text(f"Width Difference: {self.bhm.width_diff}")

            imgui.text("Applied Load")
            self._generate_imgui_input("applied_load", imgui.input_float)

            if self.applied_load > 0.0:
                if imgui.button("Optimize"):
                    print("Add optimizer here")

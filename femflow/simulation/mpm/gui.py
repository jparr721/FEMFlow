from typing import Callable, List

import imgui

from femflow.numerics.linear_algebra import matrix_to_vector
from femflow.simulation.mpm.simulation import MPMSimulation
from femflow.viz.mesh import Mesh
from femflow.viz.visualizer.visualizer_menu import VisualizerMenu
from femflow.viz.visualizer.visualizer_window import VisualizerWindow


class MPMDisplacementsWindow(VisualizerWindow):
    def __init__(self):
        name = "Displacements"
        flags = [imgui.TREE_NODE_DEFAULT_OPEN]
        super().__init__(name, flags)

        self._register_input("current_timestep", 0)

    def render(self, **kwargs) -> None:
        sim: MPMSimulation = self._unpack_kwarg("sim", MPMSimulation, **kwargs)

        mesh: Mesh = self._unpack_kwarg("mesh", Mesh, **kwargs)
        imgui.text(f"Points: {mesh.vertices.size // 3}")

        # gyroid_mesh: Mesh = self._unpack_kwarg("gyroid_mesh", Mesh, **kwargs)
        # collider_mesh: Mesh = self._unpack_kwarg("collider_mesh", Mesh, **kwargs)

        imgui.push_item_width(-1)
        self._generate_imgui_input(
            "current_timestep",
            imgui.slider_int,
            min_value=0,
            max_value=len(sim.displacements) - 1,
        )
        imgui.pop_item_width()

        if sim.loaded:
            mesh.replace(matrix_to_vector(sim.displacements[self.current_timestep]))

        if imgui.button("Run"):
            meshes = self._unpack_kwarg("meshes", list, **kwargs)
            params = self._unpack_kwarg("params", list, **kwargs)
            sim.load(mesh=mesh, meshes=meshes, params=params)
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

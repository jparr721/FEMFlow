# An Invertible Method For Material Characterization via the Inverse Material Point Method
from typing import List

import imgui
import numpy as np
from loguru import logger

from femflow.numerics.linear_algebra import vector_to_matrix
from femflow.reconstruction.behavior_matching import BehaviorMatching
from femflow.simulation.mpm.gui import MPMDisplacementsWindow, MPMSimulationWindow
from femflow.simulation.mpm.primitives import (
    generate_cube_points,
    generate_implicit_points,
)
from femflow.simulation.mpm.simulation import MPMSimulation
from femflow.viz.mesh import Mesh
from femflow.viz.visualizer.visualizer_menu import VisualizerMenu
from femflow.viz.visualizer.window import Window


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


def load_view(mesh_type: str, k, t, res):
    with Window("mpm", "Paper 1") as window:

        def generate_mesh_cb(fn: str, k: float, t: float, resolution: int):
            points = generate_implicit_points(fn, k, t, resolution)
            window.renderer.mesh["vertices"] = points

        gyroid_mesh = Mesh(generate_implicit_points(mesh_type, k, t, res))
        v = vector_to_matrix(gyroid_mesh.vertices, 3)
        minx, miny, minz = np.amin(v, axis=0)
        maxx, maxy, maxz = np.amax(v, axis=0)
        collider_mesh = Mesh(
            generate_cube_points((minx, maxx), (miny, maxy / 2), (minz, maxz), res)
        )
        collider_mesh.translate_y(maxy + (maxy * 0.5))
        collider_mesh.set_color(np.array((237.0 / 255.0, 85.0 / 255.0, 59.0 / 255.0)))

        sim = MPMSimulation([gyroid_mesh, collider_mesh])
        sim_menu_window = MPMSimulationWindow()
        sim_menu_window.add_menu(BehaviorMatchingMenu())
        window.add_mesh(sim.full_mesh)
        window.add_window(
            sim_menu_window,
            sim=sim,
            sim_status=sim.running,
            load_sim_cb=sim.load,
            start_sim_button_cb=sim.start,
            reset_sim_button_cb=sim.reset,
            mesh=sim.full_mesh,
            collider_mesh=collider_mesh,
            generate_mesh_cb=generate_mesh_cb,
        )
        window.add_window(MPMDisplacementsWindow(), sim=sim)
        window.launch()


def run_experiment(mesh_type: str, k: float, t: float, res: int):
    load_view(mesh_type, k, t, res)

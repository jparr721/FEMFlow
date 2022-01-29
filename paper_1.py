# An Invertible Method For Material Characterization via the Inverse Material Point Method
from typing import List

import imgui
from loguru import logger

from femflow.reconstruction.behavior_matching import BehaviorMatching
from femflow.simulation.mpm.gui import MPMDisplacementsWindow, MPMSimulationWindow
from femflow.simulation.mpm.primitives import generate_implicit_points
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
        self.bhm = BehaviorMatching()

    def render(self, **kwargs) -> None:
        self._generate_imgui_input("capturing", imgui.checkbox, use_key_as_label=True)

        if self.capturing:
            self.bhm.start_matching()


def load_view():
    with Window("mpm", "Paper 1") as window:

        def generate_mesh_cb(fn: str, k: float, t: float, resolution: int):
            points = generate_implicit_points(fn, k, t, resolution)
            window.renderer.mesh["vertices"] = points

        mesh = Mesh(generate_implicit_points("gyroid", 0.1, 0.3, 60))
        # mid: float = np.median(mesh.vertices[np.arange(0, len(mesh.vertices), 3)])
        # collider_mesh = Mesh(generate_cube_points((np.array((mid - 1, mid + 1))), 20))
        sim = MPMSimulation()
        sim_menu_window = MPMSimulationWindow()
        sim_menu_window.add_menu(BehaviorMatchingMenu())
        window.add_mesh(mesh)
        window.add_window(
            sim_menu_window,
            sim=sim,
            sim_status=sim.running,
            load_sim_cb=sim.load,
            start_sim_button_cb=sim.start,
            reset_sim_button_cb=sim.reset,
            mesh=mesh,
            generate_mesh_cb=generate_mesh_cb,
        )
        window.add_window(MPMDisplacementsWindow(), sim=sim)
        window.launch()


def run_experiment(debug: bool):
    try:
        load_view()
    except Exception as e:
        logger.error(f"Simulation encountered an error: {e}")
        if debug:
            raise e

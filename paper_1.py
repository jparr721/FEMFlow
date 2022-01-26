# An Invertible Method For Material Characterization via the Inverse Material Point Method
import numpy as np
from loguru import logger

from femflow.reconstruction.behavior_matching import BehaviorMatching
from femflow.simulation.mpm.gui import MPMDisplacementsWindow, MPMSimulationWindow
from femflow.simulation.mpm.primitives import (
    generate_cube_points,
    generate_implicit_points,
)
from femflow.simulation.mpm.simulation import MPMSimulation
from femflow.viz.mesh import Mesh
from femflow.viz.visualizer.window import Window


def load_view():

    with Window("mpm", "Paper 1") as window:

        def generate_mesh_cb(fn: str, k: float, t: float, resolution: int):
            points = generate_implicit_points(fn, k, t, resolution)
            window.renderer.mesh["vertices"] = points

        mesh = Mesh(generate_implicit_points("gyroid", 0.15, 0.4, 50))
        # mid: float = np.median(mesh.vertices[np.arange(0, len(mesh.vertices), 3)])
        # collider_mesh = Mesh(generate_cube_points((np.array((mid - 1, mid + 1))), 20))
        sim = MPMSimulation()
        window.add_mesh(mesh)
        window.add_window(
            MPMSimulationWindow(),
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

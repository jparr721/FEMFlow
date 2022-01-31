# An Invertible Method For Material Characterization via the Inverse Material Point Method
import numpy as np
from loguru import logger

from femflow.numerics.linear_algebra import vector_to_matrix
from femflow.reconstruction.behavior_matching import BehaviorMatching
from femflow.simulation.mpm.gui import BehaviorMatchingMenu, MPMDisplacementsWindow
from femflow.simulation.mpm.primitives import generate_implicit_points
from femflow.simulation.mpm.simulation import MPMSimulation
from femflow.solvers.mpm.particle import Particle
from femflow.viz.mesh import Mesh
from femflow.viz.visualizer.window import Window


def run_experiment(
    outdir: str,
    mesh_type: str,
    k: float,
    t: float,
    mesh_res: int,
    steps: int,
    dt: float,
    gyroid_mass: float,
    collider_mass: float,
    volume: float,
    force: float,
    gyroid_E: float,
    collider_E: float,
    gyroid_v: float,
    collider_v: float,
    hardening: float,
    grid_res: int,
    tightening_coeff: float,
) -> None:
    try:
        with Window("mpm", "Paper 1") as window:
            mesh = Mesh(generate_implicit_points(mesh_type, k, t, mesh_res))
            # sim_menu_window.add_menu(BehaviorMatchingMenu())
            window.add_mesh(mesh)
            sim = MPMSimulation(
                outdir,
                steps,
                dt,
                gyroid_mass,
                collider_mass,
                volume,
                force,
                gyroid_E,
                collider_E,
                gyroid_v,
                collider_v,
                hardening,
                grid_res,
                tightening_coeff,
            )
            window.add_window(MPMDisplacementsWindow(), sim=sim, mesh=mesh)
            window.launch()
    except Exception as e:
        logger.error(f"Simulation encountered an error: {e}")

# An Invertible Method For Material Characterization via the Inverse Material Point Method
import numpy as np
from loguru import logger

from femflow.numerics.linear_algebra import vector_to_matrix
from femflow.reconstruction.behavior_matching import BehaviorMatching
from femflow.simulation.mpm.gui import BehaviorMatchingMenu, MPMDisplacementsWindow
from femflow.simulation.mpm.primitives import (
    generate_cube_points,
    generate_implicit_points,
)
from femflow.simulation.mpm.simulation import MPMSimulation
from femflow.viz.mesh import Mesh
from femflow.viz.visualizer.window import Window


def multi_drop_experiment(experiment: int):
    def prepare_sim(
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
    ):
        gyroid_mesh = Mesh(generate_implicit_points(mesh_type, k, t, mesh_res))
        gyroid_mesh.translate_y(0.1)

        v = vector_to_matrix(gyroid_mesh.vertices, 3)
        minx, miny, minz = np.amin(v, axis=0)
        maxx, maxy, maxz = np.amax(v, axis=0)
        collider_mesh = Mesh(
            generate_cube_points((minx, maxx), (miny, maxy), (minz, maxz), mesh_res)
        )
        collider_mesh.translate_y(3)
        collider_mesh.set_color(np.array((237.0 / 255.0, 85.0 / 255.0, 59.0 / 255.0)))
        mesh = gyroid_mesh + collider_mesh

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

        return (
            mesh,
            [gyroid_mesh, collider_mesh],
            [(gyroid_mass, gyroid_E, gyroid_v), (collider_mass, collider_E, collider_v)],
            sim,
        )

    outdir = "tmp"
    dt = 1e-4
    gyroid_mass = 1.0
    k = 0.1
    t = 0.3
    volume = 1.0
    force = -9.8  # Gravity
    gyroid_v = 0.2
    collider_v = 0.4
    hardening = 0.7
    grid_res = 64
    mesh_type = "gyroid"
    if experiment == 0:
        steps = 2500
        collider_mass = 10.0
        mesh_res = 15
        tightening_coeff = 0.05
        collider_mass = 10.0
        collider_E = 1000
        gyroid_E = 140
        return prepare_sim(
            outdir,
            mesh_type,
            k,
            t,
            mesh_res,
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
    elif experiment == 1:
        mesh_res = 40
        tightening_coeff = 0.10
        collider_mass = 10.0
        collider_E = 1000
        gyroid_E = 140
        steps = 1000

        return prepare_sim(
            outdir,
            mesh_type,
            k,
            t,
            mesh_res,
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
    else:
        raise ValueError(f"Experiment {experiment} is not valid")


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
            mesh, meshes, params, sim = multi_drop_experiment(
                0,
                outdir,
                mesh_type,
                k,
                t,
                mesh_res,
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

            window.add_mesh(mesh)
            window.add_window(
                MPMDisplacementsWindow(),
                sim=sim,
                mesh=mesh,
                meshes=meshes,
                params=params,
            )
            window.launch()
    except Exception as e:
        logger.error(f"Simulation encountered an error: {e}")
        raise e

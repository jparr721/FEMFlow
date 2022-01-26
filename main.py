import os

import numpy as np
import typer
from loguru import logger

app = typer.Typer(help="FEMFlow simulation runner")

logger.add("femflow.log", mode="w+")


@app.command()
def calibrate(type: str, opt=typer.Option(0, "--camera")):

    """Calibrate with type "mask" "hsv" or "bb"

    Args:
        type (str): The mask type "mask" "hsv" "bb"
    """
    from femflow.reconstruction.calibration import (
        calibrate_bb,
        calibrate_hsv,
        calibrate_mask,
    )

    if type == "mask":
        calibrate_mask()
    elif type == "hsv":
        calibrate_hsv(int(opt))
    elif type == "bb":
        calibrate_bb()
    else:
        raise ValueError(f"Invalid calibration type {type} supplied")


@app.command()
def paper(debug: bool = typer.Option(False)):
    from paper_1 import run_experiment

    run_experiment(debug)


@app.command()
def fem(debug: bool = typer.Option(False)):
    """Launches the FEM simulation
    """
    if debug:
        import OpenGL

        OpenGL.ERROR_LOGGING = True
        OpenGL.FULL_LOGGING = True
    from femflow.viz.models import model_paths
    from femflow.viz.visualizer.window import Window

    with Window("fem") as window:
        window.add_mesh_from_file(os.path.join(model_paths(), "cube.obj"))
        window.launch()


@app.command()
def mpm(debug: bool = typer.Option(False)):
    """Launches the MPM simulation
    """
    if debug:
        import OpenGL

        OpenGL.ERROR_LOGGING = True
        OpenGL.FULL_LOGGING = True
    # from femflow.simulation.mpm_simulation import sim_3d

    # sim_3d()

    from femflow.simulation.mpm.gui import MPMDisplacementsWindow, MPMSimulationWindow
    from femflow.simulation.mpm.primitives import generate_implicit_points, gyroid
    from femflow.simulation.mpm.simulation import MPMSimulation
    from femflow.viz.mesh import Mesh
    from femflow.viz.visualizer.window import Window

    with Window("mpm") as window:
        sim = MPMSimulation()
        window.add_mesh(Mesh(generate_implicit_points(gyroid, 0.3, 0.3, 30)))
        window.add_window(
            MPMSimulationWindow(),
            sim_status=sim.loaded,
            load_sim_cb=sim.load,
            start_sim_button_cb=sim.start,
            mesh=window.renderer.mesh,
        )
        window.add_window(MPMDisplacementsWindow(), sim=sim)
        window.launch()


@app.command()
def headless(
    mesh_file: str,
    solver_type: str = typer.Option("dynamic", "--solver"),
    dt: float = typer.Option(0.001, "--dt"),
    mass: int = typer.Option(10, "--mass"),
    force: int = typer.Option(-100, "--force"),
    youngs_modulus: float = typer.Option(50000, "--youngs_modulus"),
    poissons_ratio: float = typer.Option(0.3, "--poissons_ratio"),
    rayleigh_lambda: float = typer.Option(0.0, "--rayleigh_lambda"),
    rayleigh_mu: float = typer.Option(0.0, "--rayleigh_mu"),
    timesteps: int = typer.Option(100, "--timesteps"),
):
    """Launches a headless version of the visualizer app
    """
    from femflow.simulation.linear_fem_simulation import LinearFemSimulationHeadless
    from femflow.viz.mesh import Mesh

    simulation = LinearFemSimulationHeadless(
        "linear_galerkin_headless",
        dt,
        mass,
        force,
        youngs_modulus,
        poissons_ratio,
        0,
        0,
        rayleigh_lambda,
        rayleigh_mu,
    )

    mesh = Mesh.from_file(mesh_file)
    if not mesh.tetrahedralized:
        mesh.tetrahedralize()
    simulation.load(mesh)

    if solver_type == "dynamic":
        simulation.solve_dynamic(timesteps)
    else:
        logger.info("Solving static problem")
        simulation.solve_static()

    logger.info("Saving displacements.")

    simulation_results_folder = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "simulation_results"
    )
    if not os.path.exists(simulation_results_folder):
        os.mkdir(simulation_results_folder)
    last_entry_index = len(os.listdir(simulation_results_folder))
    batch_folder_name = os.path.join(
        simulation_results_folder, f"batch_{last_entry_index}"
    )

    if not os.path.exists(batch_folder_name):
        os.mkdir(batch_folder_name)
    else:
        raise ValueError(f"{batch_folder_name} already exists")

    mesh.save("mesh.obj")
    np.savez(os.path.join(batch_folder_name, "displacements"), simulation.displacements)


@app.command()
def make_gyroid(
    path: str,
    amplitude: float = typer.Option(0.3),
    resolution: float = typer.Option(0.1),
    dimension: int = typer.Option(50),
):
    """Make a gyroid .obj file. Note that this is always saved to assets/path

    Args:
        path (str): The path to save to
        amplitude (float): The amplitude of the gyroid
        resolution (float): The resolution of the gyroid
        dimension (int): The dimension of the gyroid (mm)
    """
    from femflow.viz.mesh import Mesh

    mesh = Mesh.from_type(
        "gyroid", amplitude=amplitude, resolution=resolution, dimension=dimension
    )

    if not os.path.exists("cli_assets"):
        os.mkdir("cli_assets")

    if not mesh.save(os.path.join("cli_assets", path)):
        logger.error("Mesh failed to save")


@app.command()
def list_cameras():
    import cv2

    dev_port = 0
    working_ports = []
    available_ports = []
    for _ in range(30):
        camera = cv2.VideoCapture(dev_port)
        if camera.isOpened():
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                logger.info(f"Port {dev_port} is working and reads images ({h} x {w})")
                working_ports.append(dev_port)
        dev_port += 1
    logger.success(f"Available Ports: {available_ports}")
    logger.success(f"Working Ports: {working_ports}")


if __name__ == "__main__":
    app()


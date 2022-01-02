import typer
from loguru import logger
import os
import numpy as np

from femflow.reconstruction.calibration import calibrate_bb, calibrate_hsv, calibrate_mask
from femflow.simulation.linear_fem_simulation import LinearFemSimulation
from femflow.viz.mesh import Mesh
from femflow.viz.visualizer.visualizer import Visualizer

app = typer.Typer(help="FEMFlow simulation runner")

logger.add("femflow.log", mode="w+")


@app.command()
def calibrate(type: str):
    """Calibrate with type "mask" "hsv" or "bb"

    Args:
        type (str): The mask type "mask" "hsv" "bb"
    """
    if type == "mask":
        calibrate_mask()
    elif type == "hsv":
        calibrate_hsv()
    elif type == "bb":
        calibrate_bb()
    else:
        raise ValueError(f"Invalid calibration type {type} supplied")


@app.command()
def visualize():
    """Launches the visualizer
    """
    logger.info("Warming up...")
    with Visualizer(LinearFemSimulation()) as visualizer:
        visualizer.launch()


@app.command()
def headless(
    mesh_file: str,
    solver_type: str = typer.Option("dynamic", "--solver"),
    dt: float = typer.Option(0.001, "--dt"),
    mass: int = typer.Option(10, "--mass"),
    force: int = typer.Option(-100, "--force"),
    youngs_modulus: str = typer.Option("50000", "--youngs_modulus"),
    poissons_ratio: str = typer.Option("0.3", "--poissons_ratio"),
    shear_modulus: str = typer.Option("1000", "--shear_modulus"),
    use_damping: bool = typer.Option(False, "--use_damping"),
    material_type: str = typer.Option("isotropic", "--material_type"),
    rayleigh_lambda: float = typer.Option(0.0, "--rayleigh_lambda"),
    rayleigh_mu: float = typer.Option(0.0, "--rayleigh_mu"),
    timesteps: int = typer.Option(100, "--timesteps"),
):
    """Launches a headless version of the visualizer app

    Args:
        mesh_file (str): mesh_file
        solver_type (str): solver_type
        dt (float): dt
        mass (int): mass
        force (int): force
        youngs_modulus (str): youngs_modulus
        poissons_ratio (str): poissons_ratio
        shear_modulus (str): shear_modulus
        use_damping (bool): use_damping
        material_type (str): material_type
        rayleigh_lambda (float): rayleigh_lambda
        rayleigh_mu (float): rayleigh_mu
        timesteps (int): timesteps
    """
    simulation = LinearFemSimulation()
    simulation.dt = dt
    simulation.mass = mass
    simulation.force = force
    simulation.youngs_modulus = youngs_modulus
    simulation.poissons_ratio = poissons_ratio
    simulation.shear_modulus = shear_modulus
    simulation.use_damping = use_damping
    simulation.material_type = 1 if material_type == "orthotropic" else 0
    simulation.rayleigh_lambda = rayleigh_lambda
    simulation.rayleigh_mu = rayleigh_mu

    mesh = Mesh.from_file(mesh_file)
    if not mesh.tetrahedralized:
        mesh.tetrahedralize()
    simulation.load(mesh)

    if solver_type == "dynamic":
        simulation.simulate(mesh, timesteps)
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
    mesh = Mesh.from_type(
        "gyroid", amplitude=amplitude, resolution=resolution, dimension=dimension
    )

    if not os.path.exists("cli_assets"):
        os.mkdir("cli_assets")

    if not mesh.save(os.path.join("cli_assets", path)):
        logger.error("Mesh failed to save")


if __name__ == "__main__":
    app()

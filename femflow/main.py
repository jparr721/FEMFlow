import typer
from loguru import logger

from reconstruction.calibration import calibrate_bb, calibrate_hsv, calibrate_mask
from simulation.linear_fem_simulation import LinearFemSimulation
from viz.visualizer import Visualizer

app = typer.Typer()


@app.command()
def calibrate(type: str):
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
    logger.info("Warming up...")
    with Visualizer(LinearFemSimulation()) as visualizer:
        visualizer.launch()


if __name__ == "__main__":
    app()

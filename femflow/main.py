from loguru import logger

from simulation.linear_fem_simulation import LinearFemSimulation
from viz.visualizer import Visualizer

if __name__ == "__main__":
    logger.info("Warming up...")
    with Visualizer(LinearFemSimulation()) as visualizer:
        visualizer.launch()

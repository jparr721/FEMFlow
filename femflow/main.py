from loguru import logger

from simulation.linear_fem_simulation import make_linear_galerkin_parameter_menu, make_linear_galerkin_simulation
from viz.visualizer import Visualizer

if __name__ == "__main__":
    logger.info("Warming up...")
    with Visualizer(make_linear_galerkin_parameter_menu) as visualizer:
        visualizer.launch()

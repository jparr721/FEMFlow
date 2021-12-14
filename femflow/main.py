from loguru import logger

from simulation.linear_fem_simulation import make_linear_galerkin_parameter_menu, make_linear_galerkin_simulation
from viz.visualizer import Visualizer

if __name__ == "__main__":
    logger.info("Warming up...")
    with Visualizer(
        callback_sim_parameters=make_linear_galerkin_parameter_menu,
        callback_environment_loader=make_linear_galerkin_simulation,
    ) as visualizer:
        visualizer.launch()

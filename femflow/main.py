from loguru import logger

from viz.visualizer import Visualizer

if __name__ == "__main__":
    logger.info("Warming up...")
    with Visualizer() as visualizer:
        visualizer.launch()

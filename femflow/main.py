import glfw
import imgui
import OpenGL
from imgui.integrations.glfw import GlfwRenderer
from loguru import logger

OpenGL.ERROR_LOGGING = True
OpenGL.FULL_LOGGING = True
from OpenGL.GL import *

from viz.visualizer import Visualizer

if __name__ == "__main__":
    logger.info("Warming up...")
    with Visualizer() as visualizer:
        visualizer.launch()

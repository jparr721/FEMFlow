import os
from enum import IntEnum

import numpy as np
from loguru import logger
from OpenGL.GL import *
from OpenGL.GLU import *

from .camera import Camera
from .mesh import Mesh
from .shader_program import ShaderProgram

FRAG_SHADER_PATH = os.path.join(
    os.getcwd(), "femflow", "viz", "shaders", "core.frag.glsl"
)
VERTEX_SHADER_PATH = os.path.join(
    os.getcwd(), "femflow", "viz", "shaders", "core.vs.glsl"
)


from typing import Tuple

import numpy as np
from OpenGL.GL import *


def build_texture(
    buffer: GLint, data: np.ndarray, tex_u: int, tex_v: int, tex_wrap: GLint, tex_filter: GLint, refresh: bool = True
):
    if refresh:
        glBindTexture(GL_TEXTURE_2D, buffer)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, tex_wrap)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, tex_wrap)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, tex_filter)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, tex_filter)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, tex_u, tex_v, 0, GL_RGB, GL_UNSIGNED_BYTE, data)


def load_texture_from_image(buffer: GLint, image: np.ndarray) -> Tuple[int, int, int]:
    h, w, _ = image.shape
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, image)
    return buffer, w, h

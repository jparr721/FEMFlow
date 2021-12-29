import os
from typing import Tuple


def shader_paths() -> Tuple[str, str]:
    """Generate the shader paths

    Returns:
        Tuple[str, str]: The frg and vert paths
    """
    root = os.path.dirname(os.path.abspath(__file__))
    frag_path = os.path.join(root, "core.frag.glsl")
    vert_path = os.path.join(root, "core.vs.glsl")

    return frag_path, vert_path

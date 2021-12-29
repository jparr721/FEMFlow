import os
from typing import Tuple


def model_paths() -> str:
    """Generate the model root dir path

    Returns:
        str: The path to the models directory as a global path
    """
    return os.path.dirname(os.path.abspath(__file__))

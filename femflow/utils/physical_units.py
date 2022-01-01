import numpy as np


# TODO(@jparr721) - Add other physical units later
def numpy_bytes_human_readable(a: np.ndarray) -> str:
    return f"{a.nbytes / 1e-9}gb"

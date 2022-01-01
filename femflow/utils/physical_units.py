from typing import Union

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix


# TODO(@jparr721) - Add other physical units later
def numpy_bytes_human_readable(a: Union[np.ndarray, csr_matrix, csc_matrix]) -> str:
    if isinstance(a, np.ndarray):
        return f"{a.nbytes / 1e9}gb"
    elif isinstance(a, csr_matrix) or isinstance(a, csc_matrix):
        return f"{a.data.nbytes / 1e9}gb"

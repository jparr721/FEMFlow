from typing import Tuple, Union

import numpy as np
from numba import prange, types
from numba.experimental import jitclass
from scipy.sparse import csr_matrix

spec = [
    ("rows", types.Array(types.int64, 1, "A")),
    ("cols", types.Array(types.int64, 1, "A")),
    ("data", types.Array(types.float32, 1, "A")),
    ("shape", types.UniTuple(types.int64, 2)),
]


def index_sparse_matrix_by_indices(X, R: np.ndarray, C: np.ndarray = None) -> np.ndarray:
    if C is None:
        C = R.copy()
    assert R.ndim == 1 and C.ndim == 1, "Rows and cols must be vectors"
    rows = R.size
    cols = C.size
    RR = []
    CC = []
    for row in range(rows):
        for col in range(cols):
            RR.append(R[row])
            CC.append(C[col])
    return X[RR, CC].reshape(rows, cols)


def from_triplets(triplets: Tuple[np.ndarray, np.ndarray, np.ndarray], shape) -> "sparse_matrix":
    i, j, v = zip(*triplets)
    return sparse_matrix(i, j, v, shape)


def from_dense(dense: np.ndarray) -> "sparse_matrix":
    try:
        dense = np.atleast_2d(np.asarray(dense))

        if dense.ndim != 2:
            raise TypeError("Expected dimension of 2 for input dense matrix")

        if dense.dtype != np.float32:
            raise TypeError("Input dense type must be a float32")

        shape = dense.shape
        rows, cols = dense.nonzero()
        data = dense[rows, cols]
        return sparse_matrix(rows, cols, data, shape)
    except Exception as e:
        raise ValueError(f"Unrecognized input type '{type(dense)}' for matrix constructor") from e


@jitclass(spec)
class sparse_matrix(object):
    """A numba compatible sparse matrix class (limited scope).

    Args:
        rows (types.Array(types.int64, 1, "A")): Numba integer64 array of rows
        cols (types.Array(types.int64, 1, "A")): Numba integer64 array of cols
        data (types.Array(types.float32, 1, "A")): Numba float32 array of data (float32 for performance)
        shape (types.UniTuple(types.int64, 2)): Tuple of the shape of the matrix.
    """

    def __init__(self, rows, cols, data, shape):
        self._set_self(rows, cols, data)
        self.shape = shape

    @property
    def nnz(self):
        return self.data.size

    def submat(self, R, C) -> "sparse_matrix":
        n_rows = R.size
        n_cols = C.size

        rows = []
        cols = []
        data = []
        for row in prange(n_rows):
            for col in range(n_cols):
                r = R[row]
                c = C[col]
                rows.append(r)
                cols.append(c)
                data.append(self.data[r, c])
        return sparse_matrix(rows, cols, data, (n_rows, n_cols))

    def full(self) -> np.ndarray:
        """Returns the full matrix of the internal structure.

        Returns:
            np.ndarray: The dense numpy array.

        Todo:
            Caching.
        """
        out = np.zeros(self.shape)
        for i in prange(self.rows.size):
            r = self.rows[i]
            c = self.cols[i]
            d = self.data[i]
            out[r, c] = d
        return out

    def _make_bijective_map(self):
        pass

    def _set_self(self, rows, cols, data):
        self.rows = rows
        self.cols = cols
        self.data = data


def sparse_matrix_repr(s: sparse_matrix) -> str:
    return f"sparse_matrix(shape={s.shape}, data={s.data}, rows={s.rows} cols={s.cols}, nnz={s.nnz})"


def t():
    r = np.random.rand(5, 5)
    r = r.astype(np.float32)
    a = csr_matrix(r)
    a = index_sparse_matrix_by_indices(a, np.array([0, 1, 2]))
    print(a)

    a = from_dense(r)
    a = a.submat(np.array([0, 1, 2]), np.array([0, 1, 2]))
    print(a.full())


if __name__ == "__main__":
    t()

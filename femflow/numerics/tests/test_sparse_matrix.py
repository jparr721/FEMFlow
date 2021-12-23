import numpy as np

from ..sparse_matrix import from_dense, from_triplets, sparse_matrix
from ..geometry import index_sparse_matrix_by_indices


def test_constructor_1():
    a = np.array([[1.0, 2.0]], dtype=np.float32)
    rows, cols = a.nonzero()
    a = a[rows, cols]
    s = sparse_matrix(rows, cols, a, (10, 10))

    assert s.shape == (10, 10)
    assert (s.rows == np.array([0, 0])).all()
    assert (s.cols == np.array([0, 1])).all()
    assert s.nnz == 2


def test_full():
    a = np.array([[1.0, 2.0]], dtype=np.float32)
    s = from_dense(a)

    assert (s.full() == a).all()

def test_submat():
    a = np.random.rand()
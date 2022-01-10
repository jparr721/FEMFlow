import numpy as np

from ..circumcenter import circumcenter


def test_circumcenter():
    t = np.array([[0, 0, 0], [0, 1, 0], [0, 1, 1], [1, 1, 1]])
    t = circumcenter(t)
    assert np.isclose(t, np.array((0.5, 0.5, 0.5)), atol=0.0001).all()

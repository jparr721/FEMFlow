import numpy as np


class MPMMesh(object):
    def __init__(self, positions: np.ndarray = np.array([])):
        self.positions = positions
        self.world_coordinates = self.positions.copy()

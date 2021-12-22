from functools import cache
from typing import Tuple

import numpy as np


class Mask(object):
    def __init__(self, mask: np.ndarray):
        self.mask = mask

    @cache
    def shape(self) -> Tuple[int, int, Tuple[int, int], Tuple[int, int]]:
        """Computes the shape of the mask embedded in a scalar field. Note that a discontinuous shape will have problems.
        Result is cached for performance.

        Returns:
            Tuple[int, int, Tuple[int, int], Tuple[int, int]]: Shape and bounds
        """
        rows, row_bounds = self._extract_bounds_binary(self.mask)
        cols, col_bounds = self._extract_bounds_binary(self.mask.T)
        return rows, cols, row_bounds, col_bounds

    @cache
    def extract_embedded(self) -> np.ndarray:
        w, h, row_bounds, col_bounds = self.shape()
        out = np.zeros((w, h))
        for i, row in enumerate(range(*row_bounds), 0):
            for j, col in enumerate(range(*col_bounds), 0):
                out[i, j] = self.mask[row, col]
        return out

    @staticmethod
    def _extract_bounds_binary(mask: np.ndarray) -> Tuple[int, Tuple[int, int]]:
        found = False
        start = 0
        end = 0
        for i, row in enumerate(mask):
            if 1 in row and not found:
                start = i
                found = True

            if 1 not in row and found:
                end = i - 1
                break

        return end - start, (start, end)

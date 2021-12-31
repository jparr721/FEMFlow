from copy import deepcopy
from itertools import groupby
from typing import Any, Tuple, Union

import numpy as np
from skimage import measure


class bintensor3(object):
    def __init__(self, shape: Union[np.ndarray, Tuple[int, int, int]]):
        """A binary three-dimensional tensor object, gluten free.

        Args:
            shape (Tuple[int, int, int]): The shape of the tensor

        Raises:
            ValueError: If the shape is not a vector3 this function throws
        """
        if isinstance(shape, tuple):
            if len(shape) != 3:
                raise ValueError("bintensor3 shape must be a vector3")

            self.shape = shape
            self.data = np.zeros(self.shape)
        elif isinstance(shape, np.ndarray):
            self.shape = shape.shape
            self.data = deepcopy(shape)
        else:
            raise ValueError(f"Invalid input data type: {type(shape)}")

        _g = groupby(self.shape)
        self.square = next(_g, True) and not next(_g, False)

    def __eq__(self, other: Union[np.ndarray, "bintensor3"]) -> bool:
        """Checks equality.

        Args:
            other (Union[np.ndarray, bintensor3]): Other is a numpy array or another binary tensor object.

        Raises:
            ValueError: If the types don't align, we get an exception.

        Returns:
            bool: Equality.
        """
        t = type(other)
        if t == np.ndarray:
            return (self.data == other.data).all()
        elif t == bintensor3:
            return (self.data == other.data).all()
        else:
            raise ValueError("'Other' must be a numpy array or bintensor3.")

    def __getitem__(self, i: Union[int, Tuple[int, int], Tuple[int, int, int]]):
        return self.data[i]

    def __setitem__(self, i: Union[int, Tuple[int, int], Tuple[int, int, int]], v: Any):
        self.data[i] = v

    def __repr__(self):
        return f"bintensor3(shape={repr(self.shape)}, data={self.data}, square={repr(self.square)}"

    def set(self, mask: np.ndarray, axis: int = 0, layer: int = 0):
        """Set a given axis with a mask. If the layers are > 0 or < 0 then it will set for multiple layers in this axis.

        Args:
            mask (np.ndarray): Binary field mask
            axis (int, optional): The axis to set. Defaults to 0.
            layer (int, optional): The layer to set. Defaults to 0.
        """
        if axis == 0:
            self.data[layer] = mask
        if axis == 1:
            self.data[:, layer, :] = mask
        if axis == 2:
            self.data[:, :, layer] = mask

    def padding(self, axis: int = 0, loc: Tuple[int, int] = (-1, -1)):
        """Sets the zero padding for marching cubes on a given axis. Loc specifies where in the axis the start and
        end padding will go.

        Args:
            axis (int, optional): The axis to set zero padding on. Defaults to 0.
            loc (Tuple[int, int], optional): The padding start and end in this axis. Defaults to (-1, -1).
        """
        start, end = loc
        if start > end:
            start, end = end, start

        mask = np.zeros(self.shape[1:])

        if start < 0 and end < 0:
            self.set(mask, axis, 0)
            self.set(mask, axis, self.shape[0] - 1)
        elif start < 0 and end >= 0:
            raise ValueError("Start cannot come before end.")
        elif start >= 0 and end < 0:
            self.set(mask, axis, start)
            self.set(mask, axis, self.shape[0] - 1)
        else:
            self.set(mask, axis, start)
            self.set(mask, axis, end)

    def tomesh(self, iso=0) -> Tuple[np.ndarray, np.ndarray]:
        """Convert the mask into a surface mesh (via marching cubes)

        Args:
            iso (int, optional): The iso value for marching cubes. Defaults to 0.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The vertices and faces
        """
        v, f, _, _ = measure.marching_cubes(self.data, iso)
        return v, f

    def save(self, filename: str):
        """Save the data object to a file

        Args:
            filename (str): The name of the file
        """
        np.save(filename, self.data)

    def savetxt(self, filename: str):
        """Save the data object to a text file

        Args:
            filename (str): The name of the file
        """
        np.savetxt(filename, self.data)

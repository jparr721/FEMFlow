import numba as nb
import numpy as np


class CollisionObject(object):
    def __init__(self):
        pass

    def detect_and_resolve_collision(
        self, x: np.ndarray, v: np.ndarray, n: np.ndarray
    ) -> bool:
        """Takes a position and its velocity. We project the velocity and update the provided
        normal if a collision has occurred. This only supports sticky boundary conditions.

        Derivation from mpm siggraph course, section 12:

        x = phi(X,t) = R(t)s(t)X+b(t)
        X = phi^{-1}(x,t) = (1/s) R^{-1} (x-b)
        V(X,t) = frac{partial phi}{partial t}
             = R'sX + Rs'X + RsX' + b'
        v(x,t) = V(phi^{-1}(x,t),t)
             = R'R^{-1}(x-b) + (s'/s)(x-b) + RsX' + b'
             = omega cross (x-b) + (s'/s)(x-b) +b'

        This code was adapted from Wolper et. al. and their _excellent_ work on the
        ziran2020 project:

        https://github.com/penn-graphics-research/ziran2020

        Thank you Joshuah and UPenn for making your code open source!

        Args:
            x (np.ndarray): The position vector
            v (np.ndarray): The velocity vector
            n (np.ndarray): The normal vector

        Returns:
            (bool): True if collision is happening.
        """
        # If normal is all nan, then we know there's no collision.
        n.fill(np.nan)

        return True


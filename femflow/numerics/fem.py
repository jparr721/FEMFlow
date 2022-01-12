from typing import Tuple


def Ev_to_lame_coefficients(E: float, v: float) -> Tuple[float, float]:
    lambda_ = E * v / ((1 + v) * (1 - 2 * v))
    mu = E / (2 * (1 + v))
    return mu, lambda_

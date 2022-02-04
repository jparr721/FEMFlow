def Ev_to_mu(E: float, v: float) -> float:
    return E / (2 * (1 + v))


def Ev_to_lambda(E: float, v: float) -> float:
    return E * v / ((1 + v) * (1 - 2 * v))

from dataclasses import dataclass


@dataclass
class MPMParameters(object):
    mass: float
    volume: float
    hardening: float

    E: float
    v: float
    mu_0: float
    lambda_0: float
    gravity: float

    dt: float
    # dx is always 1 / grid_resolution
    dx: float

    grid_resolution: int
    dimensions: int

    # Turn on debug mode
    debug: bool

from typing import Callable, NamedTuple


class Environment(NamedTuple):
    step_function: Callable
    integrator: Callable
    reset_function: Callable

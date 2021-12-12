from typing import Callable, NamedTuple


class Environment(NamedTuple):
    step_function: Callable
    integrator: Callable
    parameter_menu_items: Callable
    reset_function: Callable
    start_function: Callable

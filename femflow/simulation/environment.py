from typing import Callable


class Environment(object):
    def __init__(self, step_function: Callable, integrator: Callable):
        self.step_function = step_function
        self.integrator = integrator

    def step_forward(self, *args, **kwargs):
        self.step_function(*args, **kwargs)

    def integrate(self, *args, **kwargs):
        self.integrator(*args, **kwargs)

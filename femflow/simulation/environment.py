from typing import Callable, NamedTuple


class Environment(object):
    def __init__(self, name: str):
        self.name = name

    def load(self):
        raise NotImplementedError()

    def menu(self):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def simulate(self, timesteps: int):
        raise NotImplementedError()

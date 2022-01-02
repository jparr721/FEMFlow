import abc

from femflow.viz.mesh import Mesh


class Environment(abc.ABC):
    def __init__(self, name: str):
        self.name = name
        self.loaded = False
        self.displacements = []

    @abc.abstractmethod
    def load(self, mesh: Mesh):
        raise NotImplementedError()

    def menu(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def reset(self, mesh: Mesh):
        raise NotImplementedError()

    @abc.abstractmethod
    def simulate(self, mesh: Mesh, timesteps: int):
        raise NotImplementedError()

    @abc.abstractmethod
    def solve_static(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def loss(self, batch):
        raise NotImplementedError()

    @abc.abstractmethod
    def accuracy(self, batch):
        raise NotImplementedError()

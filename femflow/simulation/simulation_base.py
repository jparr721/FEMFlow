import abc
from typing import List

from femflow.viz.mesh import Mesh


class SimulationBase(object):
    def __init__(self, meshes: List[Mesh]):
        self.meshes = meshes

    @abc.abstractmethod
    def load(self, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def start(self, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def reset(self, **kwargs):
        raise NotImplementedError()

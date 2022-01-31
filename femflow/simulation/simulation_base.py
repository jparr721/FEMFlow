import abc
from enum import Enum


class SimulationRunType(Enum):
    OFFLINE = 0
    ONLINE = 1


class SimulationBase(object):
    def __init__(self, run_type: SimulationRunType):
        self.run_type = run_type

    @abc.abstractmethod
    def load(self, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def start(self, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def reset(self, **kwargs):
        raise NotImplementedError()

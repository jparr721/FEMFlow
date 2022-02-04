import abc
from enum import Enum


class SimulationBase(object):
    def __init__(self):
        pass

    @abc.abstractmethod
    def load(self, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def start(self, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def reset(self, **kwargs):
        raise NotImplementedError()

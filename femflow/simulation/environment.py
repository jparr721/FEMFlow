from femflow.viz.mesh import Mesh


class Environment(object):
    def __init__(self, name: str):
        self.name = name
        self.loaded = False
        self.displacements = []

    def load(self, mesh: Mesh):
        raise NotImplementedError()

    def menu(self):
        raise NotImplementedError()

    def reset(self, mesh: Mesh):
        raise NotImplementedError()

    def simulate(self, mesh: Mesh, timesteps: int):
        raise NotImplementedError()

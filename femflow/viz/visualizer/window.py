from ..camera import Camera
from ..input import Input
from ..renderer import Renderer, RenderMode


class Window(object):
    def __init__(self, name: str = "FEMFlow GUI"):
        self.name = name

        self.background_color = [1.0, 1.0, 1.0, 0.0]

        self.camera = Camera()

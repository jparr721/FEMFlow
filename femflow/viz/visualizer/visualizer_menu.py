import abc
from typing import Dict, Iterable, List, Union

import imgui

from .visualizer_core import VisualizerCore


class VisualizerMenu(VisualizerCore):
    def __init__(self, name: str, flags: List[int]):
        super().__init__()
        self.name = name

        if not isinstance(flags, list):
            raise TypeError(
                f"Type: {type(flags)} is unsupported for flags, type must be a list."
            )
        self.flags = sum(flags)

        self.exapnded = True
        self.visible = True

        self.submenus: Dict[str, VisualizerMenu] = dict()

    def __eq__(self, name: str) -> bool:
        return self.name == name

    def __call__(self, **kwargs):
        self.expanded, self.visible = imgui.collapsing_header(
            self.name, self.visible, self.flags
        )

        if self.expanded:
            self.render(**kwargs)

            for submenu in self.submenus.values():
                submenu(**kwargs)

    def add_submenu(self, menus: Union["VisualizerMenu", Iterable["VisualizerMenu"]]):
        if isinstance(menus, Iterable):
            for menu in menus:
                self.add_submenu(menu)
        elif isinstance(menus, VisualizerMenu):
            self.submenus[menus.name] = menus
        else:
            raise TypeError(f"Menus must be iterable or menu type, got {type(menus)}")

    @abc.abstractmethod
    def render(self, **kwargs) -> None:
        raise NotImplementedError()

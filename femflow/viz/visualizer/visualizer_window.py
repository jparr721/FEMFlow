import abc
from collections.abc import Iterable
from typing import Dict
from typing import Iterable as _Iterable
from typing import List, Tuple, Union

import imgui

from .visualizer_core import VisualizerCore
from .visualizer_menu import VisualizerMenu


class VisualizerWindow(VisualizerCore):
    def __init__(
        self,
        name: str,
        flags: List[int] = [imgui.WINDOW_NO_COLLAPSE],
        focused=False,
        visible=False,
    ):
        super().__init__()
        self.name = name

        if not isinstance(flags, list):
            raise TypeError(
                f"Type: {type(flags)} is unsupported for flags, type must be a list."
            )

        self.flags = flags
        if imgui.WINDOW_NO_COLLAPSE not in self.flags:
            self.flags.append(imgui.WINDOW_NO_COLLAPSE)
        self.flags = sum(self.flags)

        self.focused = focused
        self.visible = visible

        self.menus: Dict[str, VisualizerMenu] = dict()

        self.dimensions: Tuple[int, int] = (0, 0)
        self.position: Tuple[int, int] = (0, 0)

    def __eq__(self, name: str):
        return self.name == name

    def __call__(self, **kwargs):
        imgui.set_next_window_size(*self.dimensions)
        imgui.set_next_window_position(*self.position)

        imgui.begin(self.name, self.flags)
        self.focused = imgui.is_window_focused() or imgui.is_item_clicked()
        self.render(**kwargs)
        for menu in self.menus.values():
            menu(**kwargs)
        imgui.end()

    def add_menu(self, menus: Union[VisualizerMenu, _Iterable[VisualizerMenu]]):
        """Add a new menu to the visualizer window, automatically adds the headings
        and such.

        Args:
            menus (Union[VisualizerMenu, _Iterable[VisualizerMenu]]): Single menu
                or list of menus.
        """
        if isinstance(menus, Iterable):
            for menu in menus:
                self.add_menu(menu)
        elif isinstance(menus, VisualizerMenu):
            self.menus[menus.name] = menus
        else:
            raise TypeError(f"Menus must be iterable or menu type, got {type(menus)}")

    @abc.abstractmethod
    def render(self, **kwargs) -> None:
        raise NotImplementedError()

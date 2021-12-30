import abc
from typing import Any, Callable


class VisualizerCore(abc.ABC):
    def __init__(self):
        self.attr_keys = set()

    @abc.abstractmethod
    def __call__(self, **kwargs):
        pass

    def __repr__(self):
        kws = [f"{key}={value!r}" for key, value in self.__dict__.items()]
        return f"{type(self).__name__}({', '.join(kws)})"

    def _unpack_kwarg(self, key: str, type, **kwargs) -> Any:
        if key not in kwargs:
            raise KeyError(
                f"Key {key} with type {type.__name__} is an expected paramter for class {self.__class__.__name__}"
            )

        if type == callable:
            if not callable(kwargs[key]):
                raise TypeError(f"Expected key '{key}' to be a callable, but got {type(kwargs[key])} instead.")
        else:
            if not isinstance(kwargs[key], type):
                raise TypeError(f"Expected key '{key}' to be {type} but got {type(kwargs[key])} instead.")

        return kwargs[key]

    def _generate_imgui_input(self, key: str, fn: Callable, use_key_as_label=False, **kwargs):
        if key not in self.attr_keys:
            raise ValueError("Register an input before mapping to a class member")

        label = key if use_key_as_label else f"##{key}"
        _, self.__dict__[key] = fn(label, self.__dict__[key], **kwargs)

    def _register_input(self, name: str, default: Any):
        if name in self.attr_keys:
            raise ValueError(f"Key {name} already exists as an input handler.")
        setattr(self, name, default)
        self.attr_keys.add(name)

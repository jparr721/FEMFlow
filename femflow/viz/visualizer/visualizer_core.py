import abc
import re
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

    def _unpack_kwarg(self, key: str, type: Any, **kwargs) -> Any:
        """Safely unpack a kwarg of a given typed input

        Args:
            key (str): The attribute to unpack.
            type (Any): Thetype of the input to unpack.
            kwargs: The arguments to check.

        Returns:
            Any:
        """
        if key not in kwargs:
            raise KeyError(
                f"Key {key} with type {type.__name__} is an expected paramter for class {self.__class__.__name__}"
            )

        if type == callable:
            if not callable(kwargs[key]):
                raise TypeError(
                    f"Expected key '{key}' to be a callable, but got {type(kwargs[key])} instead."
                )
        else:
            if not isinstance(kwargs[key], type):
                raise TypeError(
                    f"Expected key '{key}' to be {type} but got {type(kwargs[key])} instead."
                )

        return kwargs[key]

    def _generate_imgui_input(
        self, key: str, fn: Callable, use_key_as_label=False, **kwargs
    ):
        """Generate the imgui input for an attribute.

        Args:
            key (str): Key of the input, must be an existing attribute.
            fn (Callable): The imgui function to call.
            use_key_as_label (bool): Whether or not to use the attr name as the label.
            kwargs: Kwargs for the imgui type.
        """

        def beautify_label(label: str):
            return " ".join(w for w in re.split(r"\W", label) if w).capitalize()

        label = beautify_label(key) if use_key_as_label else f"##{key}"
        try:
            _, self.__dict__[key] = fn(label, self.__dict__[key], **kwargs)
        except KeyError as ke:
            raise ValueError(f"Key {key} is not an attribute.") from ke

    def _register_input(self, name: str, default: Any):
        """Register a new input handler

        Args:
            name (str): The name of the attribute to add.
            default (Any): The default value (also adds the type)
        """
        if " " in name:
            raise ValueError(
                f"Key {name} is invalid due to being an invalid python variable name"
            )

        if name in self.attr_keys:
            raise ValueError(f"Key {name} already exists as an input handler.")

        setattr(self, name, default)
        self.attr_keys.add(name)

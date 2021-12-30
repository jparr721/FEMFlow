from typing import List


def color(r: float, g: float, b: float) -> List[float]:
    def _clamp(v: float):
        if v > 1:
            v /= 255

        return v

    return [_clamp(r), _clamp(g), _clamp(b)]


error = color(1, 0, 0)
success = color(0, 1, 0)
warning = color(0, 1, 1)

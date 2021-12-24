import time


class Timer(object):
    def __init__(self, name: str = "timer"):
        self.name = name
        self.t_start = 0
        self.t_end = 0

    def __repr__(self):
        return f"Timer(name={self.name}, elapsed={self.elapsed()})"

    def elapsed(self) -> float:
        return self.t_end - self.t_start

    def start(self):
        self.t_start = time.time()

    def end(self):
        self.t_end = time.time()

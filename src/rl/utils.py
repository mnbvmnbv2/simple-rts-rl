import time
from collections import defaultdict
from contextlib import contextmanager


class TimerLog:
    def __init__(self):
        self.store = defaultdict(list)

    @contextmanager
    def record(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.store[name].append(time.perf_counter() - t0)

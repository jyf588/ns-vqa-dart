import numpy as np
import time
from typing import *


class Profiler:
    def __init__(self, name=None):
        self.name = name
        self.start_time = None
        self.total_secs = 0
        self.n_iters = 0
        self.times = []

    def start(self):
        self.start_time = time.time()

    def end(self):
        iter_secs = time.time() - self.start_time
        self.times.append(iter_secs)

    def compute_stats(self) -> Tuple[float, float]:
        avg = np.mean(self.times)
        std = np.std(self.times)
        return avg, std

    def __str__(self):
        avg, std = self.compute_stats()
        return f"""
            {self.name} Profiler Results:
            \tFirst iteration: {to_str(self.times[0])}
            \tAverage: {to_str(avg)}
            \tStd: {to_str(std)}
            """
        print(f"{self.name} Profiler Results:")
        print(f"\tFirst iteration: {to_str(self.times[0])}")
        print(f"\tAverage: {to_str(avg)}")
        print(f"\tStd: {to_str(std)}")


def to_str(time: float) -> str:
    return f"{time * 1000:.2f} ms"

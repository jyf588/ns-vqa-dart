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

    def compute_stats(self, vec: List[float]) -> Tuple[float, float]:
        avg = np.mean(vec)
        std = np.std(vec)
        return avg, std

    def __str__(self):
        avg, std = self.compute_stats(self.times)
        avg_excl_first, std_excl_first = self.compute_stats(self.times[1:])
        return f"""{self.name} Profiler Results:
            N iterations: {len(self.times)}
            First iteration: {to_str(self.times[0])}
            Average (excl first): {to_str(avg_excl_first)}
            Std (excl first): {to_str(std_excl_first)}
            Average: {to_str(avg)}
            Std: {to_str(std)}"""


def to_str(time: float) -> str:
    return f"{time * 1000:.2f} ms"

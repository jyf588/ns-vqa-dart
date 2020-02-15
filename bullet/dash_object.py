import os
from typing import List, Optional


class DashObject:
    def __init__(
        self,
        shape: str,
        size: str,
        color: str,
        world_position: List[int],
        world_orientation: List[int],
    ):
        self.shape = shape
        self.size = size
        self.color = color
        self.world_position = world_position
        self.world_orientation = world_orientation


class DashTable(DashObject):
    def __init__(self):
        self.shape = "tabletop"
        self.size = None
        self.color = "grey"
        self.world_position = [0.25, 0.2, 0.0]
        self.world_orientation = [0.0, 0.0, 0.0, 1.0]

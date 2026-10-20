"""What a scripted task provides: a layout sampler, a plan, a success check."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class Waypoint:
    """A TCP pose to reach (base frame), the gripper while moving there (0 open, 1 closed), how fast
    (`slow` near objects), and how long to hold it once reached."""

    position: np.ndarray
    rotation: Rotation
    grip: float = 0.0
    slow: bool = False
    hold: float = 0.0


class Task:
    """A scripted manipulation task. Subclasses define the four methods; nothing here talks to ROS."""

    name: str
    scene: str  # baxter_ros --scene
    parked: dict = {}  # {arm: joints (s0 .. w2)} for the arms the task does not use

    def sample(self, rng):
        """A random layout (a flat list of numbers, as /sim/layout takes)."""
        raise NotImplementedError

    def plan(self, layout, home):
        """Waypoints from `home` (position, Rotation) through the task and back to `home`."""
        raise NotImplementedError

    def success(self, layout, objects):
        """Whether the episode succeeded, from the objects' final base-frame positions {name: xyz}."""
        raise NotImplementedError

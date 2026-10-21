"""What a scripted task provides: a layout sampler, a plan, a success check, and the plan again from
any state an attempt reaches (the expert taking over)."""

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


@dataclass(frozen=True)
class Replan:
    """The expert's plan from a state reached mid-attempt: hold where the arm is for `wait` s with the
    gripper commanded to `grip` (finishing a close or an open), then the waypoints. `origin`, the pose
    (position, Rotation) the expert's move to the first waypoint starts from in the full plan: an arm
    on that move is taken over where the expert would be on it, not started afresh."""

    waypoints: list
    grip: float = 0.0
    wait: float = 0.0
    origin: tuple | None = None


class Task:
    """A scripted manipulation task. Subclasses define the six methods; nothing here talks to ROS."""

    name: str
    scene: str  # baxter_ros --scene
    parked: dict = {}  # {arm: joints (s0 .. w2)} for the arms the task does not use

    def sample(self, rng):
        """A random layout (a flat list of numbers, as /sim/layout takes)."""
        raise NotImplementedError

    def plan(self, layout, home):
        """Waypoints from `home` (position, Rotation) through the task and back to `home`."""
        raise NotImplementedError

    def replan(self, tcp, gripper, objects, home, layout=None):
        """A Replan from the measured TCP (position, Rotation), the gripper (0 open .. 1 closed) and the
        objects' base-frame poses {name: (position, quaternion [x y z w])}, back to `home`; None where
        the expert has no plan (an object it cannot pick up). `layout`, the attempt's (where the objects
        started), if known."""
        raise NotImplementedError

    def layout(self, objects):
        """The layout (as sample() gives it) the objects' poses {name: (position, quaternion)} are in."""
        raise NotImplementedError

    def success(self, layout, objects):
        """Whether the episode succeeded, from the objects' final base-frame positions {name: xyz}."""
        raise NotImplementedError

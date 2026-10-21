"""Plan input: a policy's timed TCP plan as a joint trajectory, played as replay plays a demo.

The policy runtime (RethoughtAI) publishes, every policy step, its plan from now on: a nav_msgs/Path of
TCP poses in the base frame, each stamped with the time it is due. Each new plan's poses are turned
into joints by tcp_ik (from the joints being played, or the measured arm when none are: the least joint
motion), and those joints become a replay.Trajectory, which the teleop's replay stream plays. The
teleop's servo, velocity IK and setpoint smoothing are not used.
"""

import numpy as np
from nav_msgs.msg import Path
from scipy.spatial.transform import Rotation

from ..replay import Trajectory
from ..settings import LIMITS
from ..tcp_ik import joints_along

MIN_SPAN = 0.1  # s: replay.Trajectory's velocity fit needs this much of a plan


class PlanTarget:
    """The newest plan as (replay.Trajectory, t0), or None to hold."""

    def __init__(self, node, topic, kinematics, stale_after=0.2):
        self.node, self.kinematics, self.stale_after = node, kinematics, float(stale_after)
        self.goal = None  # (position, Rotation) due now, for telemetry
        self._moving = False  # read by the teleop's cuff handling
        self._path = None  # (t (N,), positions (N, 3), Rotations): the newest plan received
        self._played = None  # (path it was built from, (Trajectory, t0))
        node.create_subscription(Path, topic, self._on_path, 10)

    def _on_path(self, msg):
        t = np.array([p.header.stamp.sec + 1e-9 * p.header.stamp.nanosec for p in msg.poses])
        if len(t) < 2 or np.any(np.diff(t) <= 0) or t[-1] - t[0] < MIN_SPAN:
            return
        positions = np.array([[p.pose.position.x, p.pose.position.y, p.pose.position.z] for p in msg.poses])
        o = [p.pose.orientation for p in msg.poses]
        self._path = (t, positions, Rotation.from_quat([[q.x, q.y, q.z, q.w] for q in o]))

    def plan(self, q_measured, q_playing):
        """(Trajectory, t0) of the newest plan, or None to hold (no plan, or none fresh)."""
        self.goal, self._moving = None, False
        path = self._path
        now = self.node.get_clock().now().nanoseconds * 1e-9
        if path is None or now > path[0][-1] + self.stale_after:
            return None
        if self._played is None or self._played[0] is not path:
            t, positions, rotations = path
            seed = q_measured if q_playing is None else q_playing
            q, _ = joints_along(self.kinematics, seed, positions, rotations, LIMITS.position_low, LIMITS.position_high)
            self._played = (path, (Trajectory(t - t[0], q), t[0]))
        t, positions, rotations = path
        k = int(np.clip(np.searchsorted(t, now), 0, len(t) - 1))
        self.goal, self._moving = (positions[k], rotations[k]), True
        return self._played[1]

    def reset(self):
        """Drop the plan (a cuff squeeze, home): the next one is built from where the arm is."""
        self._path, self._played = None, None

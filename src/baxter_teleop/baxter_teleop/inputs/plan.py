"""Plan input: a policy's timed TCP plan as a joint trajectory, played as replay plays a demo.

The policy runtime (RethoughtAI) publishes, every policy step, its plan from now on: a nav_msgs/Path of
TCP poses in the base frame, each stamped with the time it is due. Each new plan's poses are turned
into joints by tcp_ik (from the joints being played, or the measured arm when none are: the least joint
motion), and the teleop's replay stream plays them. The teleop's servo, velocity IK and setpoint
smoothing are not used.

What is played is one curve through the joint knots already played (the last HISTORY s) and the new
plan's: a cubic smoothing spline per joint (penalty LAM on the second derivative), position, velocity
and acceleration from the same curve. The plan's future is known, so the smoothing is zero-phase and
adds no lag, and the knots already played keep the curve close to what is playing where a new plan
takes over. A plan played from its own start instead (replay.Trajectory per plan) kinked every 40 ms:
on the robot, a policy through the runtime's ramp merge, 1.00 rad/s^2 rms commanded acceleration
against 0.61 this way (offline, its own chunks), and 0.66 for a joint replay of a demo.

The smoothing spline passes near its knots, not through them, so a new curve need not start where the
old one is now. When a new plan's joints differ from the old (a policy's next chunk disagreeing with
the last: 7 to 39 mm and up to 18 deg on SweepTrash), that gap was a step of up to 0.05 rad in one
5 ms setpoint. Each new curve is therefore continued from the setpoint playing now: the difference
in position, velocity and acceleration decays to zero over BLEND s along a quintic, the polynomial
of least degree matching all three at both ends (Craig, Introduction to Robotics, 3rd ed., 7.3), so
the inverse dynamics feedforward sees a continuous acceleration too.
"""

import numpy as np
from nav_msgs.msg import Path
from scipy.interpolate import make_smoothing_spline
from scipy.spatial.transform import Rotation

from ..replay import Trajectory
from ..settings import LIMITS
from ..tcp_ik import joints_along

MIN_SPAN = 0.1  # s: a plan must cover this much to be played
HISTORY = 0.4  # s of knots already played kept in the curve
LAM = 1e-4  # smoothing spline penalty: within ~0.65 deg of the plan's joints on a recorded demo
BLEND = 0.2  # s for a new curve to reach its plan from the setpoint playing; plans arrive every 0.04 s


class SmoothJoints:
    """(q, qd, qdd) at t from one cubic smoothing spline per joint through knots (t (N,), q (N, 7)), as
    replay.Trajectory's interface; held at its ends outside them."""

    def __init__(self, t, q, lam=LAM):
        self.splines = [make_smoothing_spline(t, q[:, j], lam=lam) for j in range(q.shape[1])]
        self.velocity = [s.derivative(1) for s in self.splines]
        self.acceleration = [s.derivative(2) for s in self.splines]
        self.duration = float(t[-1])

    def __call__(self, t):
        t = float(np.clip(t, 0.0, self.duration))
        return (
            np.array([s(t) for s in self.splines]),
            np.array([v(t) for v in self.velocity]),
            np.array([a(t) for a in self.acceleration]),
        )


class Continued:
    """`curve` (replay.Trajectory's interface) from local time `t` on, offset so that at `t` it plays
    (q, qd, qdd): the offset decays to zero along a quintic over `span` s."""

    def __init__(self, curve, t, q, qd, qdd, span=BLEND):
        self.curve, self.t, self.span, self.duration = curve, float(t), float(span), curve.duration
        e = np.subtract(q, curve(t)[0])
        v = np.subtract(qd, curve(t)[1])
        a = np.subtract(qdd, curve(t)[2])
        T = self.span
        ends = np.array([[T**3, T**4, T**5], [3 * T**2, 4 * T**3, 5 * T**4], [6 * T, 12 * T**2, 20 * T**3]])
        start = np.array([e + v * T + a * T**2 / 2, v + a * T, a])  # what the first three terms reach at T
        self.coefficients = np.vstack([e, v, a / 2, np.linalg.solve(ends, -start)])  # (6, 7), c_k s^k

    def __call__(self, t):
        q, qd, qdd = self.curve(t)
        s = max(0.0, float(t) - self.t)  # nothing plays it before the switch but clock rounding
        if s < self.span:
            c, k = self.coefficients, np.arange(6)
            q = q + (s**k) @ c
            qd = qd + (k[1:] * s ** (k[1:] - 1)) @ c[1:]
            qdd = qdd + (k[2:] * (k[2:] - 1) * s ** (k[2:] - 2)) @ c[2:]
        return q, qd, qdd


class PlanTarget:
    """The newest plan as (trajectory, t0) for the replay stream, or None to hold."""

    def __init__(self, node, topic, kinematics, stale_after=0.2):
        self.node, self.kinematics, self.stale_after = node, kinematics, float(stale_after)
        self.goal = None  # (position, Rotation) due now, for telemetry
        self._moving = False  # read by the teleop's cuff handling
        self._path = None  # (t (N,), positions (N, 3), Rotations): the newest plan received
        self._played = None  # (path it was built from, (trajectory, t0))
        self._history = []  # (t, q): the knot due at each plan's start, as played
        node.create_subscription(Path, topic, self._on_path, 10)

    def _on_path(self, msg):
        t = np.array([p.header.stamp.sec + 1e-9 * p.header.stamp.nanosec for p in msg.poses])
        if len(t) < 2 or np.any(np.diff(t) <= 0) or t[-1] - t[0] < MIN_SPAN:
            return
        positions = np.array([[p.pose.position.x, p.pose.position.y, p.pose.position.z] for p in msg.poses])
        o = [p.pose.orientation for p in msg.poses]
        self._path = (t, positions, Rotation.from_quat([[q.x, q.y, q.z, q.w] for q in o]))

    def plan(self, q_measured, q_playing):
        """(trajectory, t0) of the newest plan, or None to hold (no plan, or none fresh)."""
        self.goal, self._moving = None, False
        path = self._path
        now = self.node.get_clock().now().nanoseconds * 1e-9
        if path is None or now > path[0][-1] + self.stale_after:
            return None
        if self._played is None or self._played[0] is not path:
            t, positions, rotations = path
            seed = q_measured if q_playing is None else q_playing
            q, _ = joints_along(self.kinematics, seed, positions, rotations, LIMITS.position_low, LIMITS.position_high)
            past = [(tk, qk) for tk, qk in self._history if t[0] - HISTORY <= tk < t[0] - 1e-6]
            knots_t = np.r_[[tk for tk, _ in past], t]
            knots_q = np.vstack([qk for _, qk in past] + [q])
            if len(knots_t) >= 5:
                played = (SmoothJoints(knots_t - knots_t[0], knots_q), knots_t[0])
            else:  # too few knots for the smoothing spline
                played = (Trajectory(t - t[0], q), t[0])
            if q_playing is not None and self._played is not None:  # continue from the setpoint playing
                old, old_t0 = self._played[1]
                played = (Continued(played[0], now - played[1], *old(now - old_t0)), played[1])
            self._played = (path, played)
            self._history = [*past, (t[0], q[0])]
        t, positions, rotations = path
        k = int(np.clip(np.searchsorted(t, now), 0, len(t) - 1))
        self.goal, self._moving = (positions[k], rotations[k]), True
        return self._played[1]

    def reset(self):
        """Drop the plan (a cuff squeeze, home): the next one is built from where the arm is."""
        self._path, self._played, self._history = None, None, []

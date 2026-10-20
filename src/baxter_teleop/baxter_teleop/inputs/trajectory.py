"""Trajectory adapter: servo the TCP along a timed TCP trajectory, with its own velocity as feedforward."""

import numpy as np
from nav_msgs.msg import Path
from scipy.interpolate import CubicSpline, make_smoothing_spline
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, RotationSpline, Slerp

from ..servo import servo_twist


def smoothing_splines(t, positions, rotations, lam):
    """(position spline, rotation spline) fitted, not interpolated: a cubic smoothing spline per axis
    (penalty lam on the second derivative) and a rotation spline through orientations smoothed on
    sign-continuous quaternions (Savitzky-Golay, cubic, 5 to 9 samples)."""
    position = SmoothedPosition([make_smoothing_spline(t, positions[:, k], lam=lam) for k in range(3)])
    q = rotations.as_quat()
    flips = np.cumsum(np.r_[0, np.sum(q[1:] * q[:-1], axis=1) < 0]) % 2 == 1
    q[flips] *= -1
    window = min(9, len(t) - (1 - len(t) % 2))  # odd, at most the samples there are
    q = savgol_filter(q, window, 3, axis=0)
    return position, RotationSpline(t, Rotation.from_quat(q / np.linalg.norm(q, axis=1, keepdims=True)))


class SmoothedPosition:
    """Three per-axis splines called as one: (t) position, (t, 1) velocity."""

    def __init__(self, splines):
        self.splines = splines

    def __call__(self, t, nu=0):
        return np.array([s(t, nu) for s in self.splines])


class TrajectoryTarget:
    """Turns a timed TCP trajectory into a base-frame TCP twist.

    Contract (e.g. /policy/{arm}/path): a nav_msgs/Path of TCP poses in the robot base frame, each
    PoseStamped stamped with the time it is due (a policy's chunk: its observation time plus k steps).
    Each new path replaces the last. Every cycle the servo chases the pose due now, r(t), with the
    path's velocity there, r'(t), as feedforward: twist = r' + gain * (r - x), so the tracking error
    decays as e' = -gain * e for any gain, with no standing lag to compensate. Poses are absolute:
    nothing is anchored on the arm, and a path that has run out holds its last pose.

    feedforward=False drops r' (the servo alone, lagging a moving goal by 1/gain). interpolation:
    'linear' (piecewise linear, velocity constant per segment), 'cubic' (C2 spline through every
    pose: continuous velocity, but any jitter in the poses is differentiated into r') or 'smooth' (a
    cubic smoothing spline, penalty `smoothing`, and a rotation spline through Savitzky-Golay
    smoothed orientations: r' from the fit, not from the jitter). end: 'hold' (stop at the last pose) or 'continue' (keep
    the last segment's velocity for up to stale_after, for a stream of short paths). shift (s)
    evaluates the path that much later.
    """

    def __init__(
        self,
        node,
        topic,
        gain,
        max_linear,
        max_angular,
        scale=None,
        feedforward=True,
        interpolation='linear',
        end='hold',
        shift=0.0,
        stale_after=0.2,
        smoothing=1e-3,
    ):
        if interpolation not in ('linear', 'cubic', 'smooth'):
            raise ValueError(f"interpolation must be 'linear', 'cubic' or 'smooth', got {interpolation}")
        self.smoothing = float(smoothing)
        if end not in ('hold', 'continue'):
            raise ValueError(f"end must be 'hold' or 'continue', got {end}")
        self.node, self.gain, self.max_linear, self.max_angular = (
            node,
            float(gain),
            float(max_linear),
            float(max_angular),
        )
        self.scale = None if scale is None else np.asarray(scale, dtype=float)
        self.feedforward, self.interpolation, self.end = bool(feedforward), interpolation, end
        self.shift, self.stale_after = float(shift), float(stale_after)
        self.goal = None  # (position, Rotation) the TCP is servoed toward this cycle, base frame
        self._moving = False  # read by the teleop's cuff handling, as VRTarget's
        self._path = None
        node.create_subscription(Path, topic, self._on_path, 10)

    def _on_path(self, msg):
        if not msg.poses:
            return
        t = np.array([p.header.stamp.sec + 1e-9 * p.header.stamp.nanosec for p in msg.poses])
        if np.any(np.diff(t) <= 0):
            self.node.get_logger().warn('trajectory: pose stamps must increase; path ignored')
            return
        positions = np.array([[p.pose.position.x, p.pose.position.y, p.pose.position.z] for p in msg.poses])
        o = [p.pose.orientation for p in msg.poses]
        rotations = Rotation.from_quat([[q.x, q.y, q.z, q.w] for q in o])
        splines = None
        if self.interpolation == 'cubic' and len(t) >= 3:
            splines = (CubicSpline(t, positions, axis=0), RotationSpline(t, rotations))
        elif self.interpolation == 'smooth' and len(t) >= 5:
            splines = smoothing_splines(t, positions, rotations, self.smoothing)
        self._path = (t, positions, rotations, splines)

    def _sample(self, now):
        """(position, Rotation, linear velocity, angular velocity) due at `now`, or None when stale."""
        t, positions, rotations, splines = self._path
        if now > t[-1] + self.stale_after:
            return None
        if len(t) == 1 or now <= t[0]:
            return positions[0], rotations[0], np.zeros(3), np.zeros(3)
        if now >= t[-1]:
            if self.end == 'hold':
                return positions[-1], rotations[-1], np.zeros(3), np.zeros(3)
            dt = t[-1] - t[-2]
            v = (positions[-1] - positions[-2]) / dt
            w = (rotations[-1] * rotations[-2].inv()).as_rotvec() / dt
            ahead = now - t[-1]
            return positions[-1] + v * ahead, Rotation.from_rotvec(w * ahead) * rotations[-1], v, w
        if splines is not None:
            position_spline, rotation_spline = splines
            rotation = rotation_spline(now)
            # RotationSpline's rate is in the rotating (TCP) frame; the servo's twist is in the base frame
            return position_spline(now), rotation, position_spline(now, 1), rotation.apply(rotation_spline(now, 1))
        i = int(np.searchsorted(t, now))  # t[i - 1] < now <= t[i]
        dt = t[i] - t[i - 1]
        s = (now - t[i - 1]) / dt
        position = positions[i - 1] + s * (positions[i] - positions[i - 1])
        rotation = Slerp([0.0, 1.0], rotations[[i - 1, i]])(s)
        v = (positions[i] - positions[i - 1]) / dt
        w = (rotations[i] * rotations[i - 1].inv()).as_rotvec() / dt
        return position, rotation, v, w

    def twist(self, tcp_position, tcp_rotation):
        """Base-frame TCP twist [v, w] toward the pose due now, or None to hold position."""
        self.goal = None
        self._moving = False
        if self._path is None:
            return None
        now = self.node.get_clock().now().nanoseconds * 1e-9 + self.shift
        sample = self._sample(now)
        if sample is None:
            return None
        position, rotation, v, w = sample
        self.goal = (position, rotation)
        self._moving = True
        return servo_twist(
            self.goal,
            tcp_position,
            Rotation.from_matrix(tcp_rotation),
            self.gain,
            self.max_linear,
            self.max_angular,
            self.scale,
            feedforward=np.concatenate([v, w]) if self.feedforward else None,
        )

    def reset(self):
        """Absolute poses: nothing to re-anchor; the next path is followed as it is."""

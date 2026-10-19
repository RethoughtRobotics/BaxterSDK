"""ISO 9283-style test trajectories for the teleop servo, as dense TCP offsets from the start pose.

ISO 9283:1998, "Manipulating industrial robots -- Performance criteria and related test methods",
characterises a robot with pose tests (approach a set of points and stop: accuracy, repeatability,
overshoot, stabilisation time) and path tests (lines and circles at set speeds: path accuracy,
cornering, velocity accuracy). The protocol here keeps those test types, scaled to Baxter's
workspace near the untuck pose and kept planar (constant height, orientation held):

  pose      5 points of a horizontal square (centre P1, corners P2-P5), visited P1..P5 in order for
            `cycles` cycles, each approached with a trapezoidal velocity profile and held for `dwell`.
  linear    out and back along a line with trapezoidal profiles: acceleration, cruise, deceleration
            to rest at each end, i.e. a full reversal of direction.
  circle    circle(s) traversed at constant speed between trapezoidal ramp-up and ramp-down (no
            velocity step at the start, unlike circle_protocol).

All motions use the same `speed` (m/s) and `accel` (m/s^2) limits, so one parameter sweeps all of
them. Offsets are in the robot base frame; `side` is -1 for the right arm (shapes extend toward the
robot's right, away from the body) and +1 for the left. Sampled at `dt`.

Each test returns (offsets (n, 3), phases, events). phases: [(name, t_start, t_end)]. events: for
the pose test, one dict per target {'target': 'P3', 'point': [...], 't_arrive', 't_leave'}; for the
linear test, the reversals {'kind': 'reversal', 't': ..., 'point': [...]}.
"""

import numpy as np

DT = 0.01


def trapezoid(distance, speed, accel, dt=DT):
    """Arc length s(t) from 0 to `distance` at rest at both ends: trapezoidal velocity (triangular
    when the distance is too short to reach `speed`). Samples at dt, excluding t=0, ending exactly
    at `distance`."""
    if distance <= 0:
        return np.zeros(0)
    t_acc = speed / accel
    if accel * t_acc**2 >= distance:  # triangular: never reaches speed
        t_acc = np.sqrt(distance / accel)
        speed = accel * t_acc
    t_cruise = (distance - accel * t_acc**2) / speed
    total = 2 * t_acc + t_cruise
    t = np.arange(1, int(np.ceil(total / dt)) + 1) * dt
    t = np.minimum(t, total)
    s = np.where(
        t < t_acc,
        0.5 * accel * t**2,
        np.where(
            t < t_acc + t_cruise,
            0.5 * accel * t_acc**2 + speed * (t - t_acc),
            distance - 0.5 * accel * (total - t) ** 2,
        ),
    )
    s[-1] = distance
    return s


def _move(start, end, speed, accel, dt):
    start, end = np.asarray(start, float), np.asarray(end, float)
    length = np.linalg.norm(end - start)
    s = trapezoid(length, speed, accel, dt)
    if not len(s):
        return np.zeros((0, 3))
    return start + np.outer(s / length, end - start)


def _hold(point, duration, dt):
    return np.repeat(np.asarray(point, float)[None], int(round(duration / dt)), axis=0)


class _Builder:
    """Accumulates offsets and phase/event times."""

    def __init__(self, dt):
        self.dt, self.points, self.phases, self.events = dt, [np.zeros(3)], [], []

    @property
    def t(self):
        return (len(self.points) - 1) * self.dt

    @property
    def here(self):
        return self.points[-1]

    def extend(self, block):
        self.points.extend(block)

    def result(self):
        return np.asarray(self.points), self.phases, self.events


def pose_test(speed, accel, side=-1, half=0.10, dwell=1.5, cycles=2, dt=DT, builder=None):
    """ISO 9283 pose test in a horizontal square of side 2 * half, centred `half` toward `side`."""
    b = builder or _Builder(dt)
    centre = np.array([0.0, side * half, 0.0])
    corners = [(1, 1), (1, -1), (-1, -1), (-1, 1)]
    targets = {
        'P1': centre,
        **{f'P{i + 2}': centre + half * np.array([cx, cy, 0.0]) for i, (cx, cy) in enumerate(corners)},
    }
    t0 = b.t
    for _ in range(cycles):
        for name, point in targets.items():
            b.extend(_move(b.here, point, speed, accel, dt))
            t_arrive = b.t
            b.extend(_hold(point, dwell, dt))
            b.events.append({'target': name, 'point': point.tolist(), 't_arrive': t_arrive, 't_leave': b.t})
    b.extend(_move(b.here, np.zeros(3), speed, accel, dt))
    b.phases.append(('pose', t0, b.t))
    return b.result() if builder is None else None


def linear_test(speed, accel, side=-1, length=0.30, reps=3, dwell=0.5, dt=DT, builder=None):
    """Out and back along the base y axis (toward `side`), `reps` times, resting `dwell` at the start."""
    b = builder or _Builder(dt)
    far = np.array([0.0, side * length, 0.0])
    t0 = b.t
    for _ in range(reps):
        b.extend(_move(b.here, far, speed, accel, dt))
        b.events.append({'kind': 'reversal', 't': b.t, 'point': far.tolist()})
        b.extend(_move(b.here, np.zeros(3), speed, accel, dt))
        b.events.append({'kind': 'reversal', 't': b.t, 'point': [0.0, 0.0, 0.0]})
        b.extend(_hold(np.zeros(3), dwell, dt))
    b.phases.append(('linear', t0, b.t))
    return b.result() if builder is None else None


def circle_test(speed, accel, side=-1, radius=0.12, loops=2, dt=DT, builder=None):
    """`loops` circles centred `radius` toward `side`, starting/ending at the start point; arc length
    follows a trapezoidal profile so the TCP starts and stops at rest."""
    b = builder or _Builder(dt)
    t0 = b.t
    s = trapezoid(loops * 2 * np.pi * radius, speed, accel, dt)
    angle = -side * np.pi / 2 + s / radius
    b.extend(np.c_[radius * np.cos(angle), side * radius + radius * np.sin(angle), np.zeros(len(s))])
    b.phases.append(('circle', t0, b.t))
    return b.result() if builder is None else None


def iso_protocol(speed=0.15, accel=0.5, side=-1, pause=1.0, dt=DT):
    """pose, linear and circle tests back to back, each preceded by a `pause` at rest at the start."""
    b = _Builder(dt)
    for test in (pose_test, linear_test, circle_test):
        b.extend(_hold(np.zeros(3), pause, dt))
        test(speed, accel, side=side, dt=dt, builder=b)
    b.extend(_hold(np.zeros(3), pause, dt))
    return b.result()

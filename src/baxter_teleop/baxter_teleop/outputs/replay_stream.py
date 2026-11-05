"""Stream a joint trajectory to Baxter as replay plays a demo, for the teleop's plan input.

Each period (200 Hz, replay.RATE) the trajectory is sampled at the current time, `q, qd, qdd =
trajectory(t - t0)` (replay.Trajectory: cubic spline, Savitzky-Golay velocity and acceleration), and
sent as Setpoint(q, qd, qdd) through `send`, replay's JointCommandPublisher: RAW_POSITION_MODE with
the inverse dynamics feedforward. The same sample and the same publisher as replay's pose mode, so a
plan is played as a demo is replayed.

It has the raw output's interface for the teleop (stop, position, velocity, active, close), so the
teleop's hold and cuff handling stop it as they stop the raw output. play() swaps in a new trajectory.

RAW_POSITION_MODE goes around Baxter's own command smoothing, so a step in the trajectory would reach the
joints as sent. Each setpoint therefore moves at most SPEED_RATIO times the joint velocity limits from
the last one sent, and the velocity sent with it is that of the clamped step.

`offset`, if set, is called each period with the period and the plan's joints and returns a joint offset
and its rate, added to the setpoint (inputs/steer.py): the path that moves at 200 Hz while plans arrive at 25 Hz. It stays
inside the joint limits.
"""

import threading
import time

import numpy as np

from ..settings import LIMITS
from .setpoint import Setpoint

RATE = 200.0  # Hz, as replay.RATE
SPEED_RATIO = 1.5  # of LIMITS.velocity


class ReplayStream:
    """A trajectory played at RATE through `send`; `clock()` is the time the trajectory's t0 is on."""

    def __init__(self, send, clock, rate=RATE):
        self.send, self.clock, self.period = send, clock, 1.0 / float(rate)
        self._lock = threading.Lock()
        self._trajectory, self._t0 = None, 0.0
        self._last = None  # the last Setpoint sent
        self.offset = None  # (period, plan's q) -> (joint offset, its rate), see the module docstring
        self._thread, self._running = None, False

    @property
    def active(self):
        return self._trajectory is not None

    @property
    def position(self):
        """Setpoint position now, or None while stopped."""
        with self._lock:
            return None if self._trajectory is None or self._last is None else self._last.position.copy()

    @property
    def commanded(self):
        """The position last sent, offset included, or None while stopped (the same as `position` here)."""
        return self.position

    @property
    def velocity(self):
        with self._lock:
            return np.zeros(7) if self._trajectory is None or self._last is None else self._last.velocity.copy()

    def play(self, trajectory, t0):
        """Play `trajectory` (replay.Trajectory), its time 0 at clock() == t0, from the next period on."""
        with self._lock:
            self._trajectory, self._t0 = trajectory, float(t0)

    def stop(self):
        """Send nothing more; the caller holds the arm (Limb.exit_control_mode)."""
        with self._lock:
            self._trajectory, self._last = None, None
            if self.offset is not None:
                self.offset.reset()

    def tick(self):
        """One period: sample the trajectory now and send it, as replay's pose mode does."""
        with self._lock:
            if self._trajectory is None:
                return None
            q, qd, qdd = self._trajectory(self.clock() - self._t0)
            q, qd, qdd = np.asarray(q), np.asarray(qd), np.asarray(qdd)
            if self.offset is not None:
                added, rate = self.offset(self.period, q)
                q, qd = np.clip(q + added, LIMITS.position_low, LIMITS.position_high), qd + rate
            if self._last is not None:
                largest = SPEED_RATIO * np.asarray(LIMITS.velocity) * self.period
                step = np.clip(q - self._last.position, -largest, largest)
                if not np.array_equal(step, q - self._last.position):
                    q, qd, qdd = self._last.position + step, step / self.period, np.zeros_like(qdd)
            self._last = Setpoint(q, qd, qdd)
            self.send(self._last)
            return self._last

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, name='replay_stream', daemon=True)
        self._thread.start()
        return self

    def close(self):
        self.stop()
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _run(self):
        deadline = time.monotonic()
        while self._running:
            self.tick()
            deadline += self.period
            delay = deadline - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            elif delay < -self.period:  # fell behind: skip rather than burst
                deadline = time.monotonic()

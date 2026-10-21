"""Stream a joint trajectory to Baxter as replay plays a demo, for the teleop's plan input.

Each period (200 Hz, replay.RATE) the trajectory is sampled at the current time, `q, qd, qdd =
trajectory(t - t0)` (replay.Trajectory: cubic spline, Savitzky-Golay velocity and acceleration), and
sent as Setpoint(q, qd, qdd) through `send`, replay's JointCommandPublisher: RAW_POSITION_MODE with
the inverse dynamics feedforward. The same sample and the same publisher as replay's pose mode, so a
plan is played as a demo is replayed.

It has the raw output's interface for the teleop (stop, position, velocity, active, close), so the
teleop's hold and cuff handling stop it as they stop the raw output. play() swaps in a new trajectory.
"""

import threading
import time

import numpy as np

from .setpoint import Setpoint

RATE = 200.0  # Hz, as replay.RATE


class ReplayStream:
    """A trajectory played at RATE through `send`; `clock()` is the time the trajectory's t0 is on."""

    def __init__(self, send, clock, rate=RATE):
        self.send, self.clock, self.period = send, clock, 1.0 / float(rate)
        self._lock = threading.Lock()
        self._trajectory, self._t0 = None, 0.0
        self._last = None  # the last Setpoint sent
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

    def tick(self):
        """One period: sample the trajectory now and send it, as replay's pose mode does."""
        with self._lock:
            if self._trajectory is None:
                return None
            q, qd, qdd = self._trajectory(self.clock() - self._t0)
            self._last = Setpoint(np.asarray(q), np.asarray(qd), np.asarray(qdd))
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

"""Joint velocity commands in Baxter's VELOCITY_MODE, ramped within the joint acceleration limits."""

import time

import numpy as np


class VelocityOutput:
    """Sends each cycle's joint velocity, changed from the last command by at most a_max * dt.

    The ramp starts from the previous command, not the (lagging) measured velocity, and scales the
    whole change uniformly, so its direction is kept. A gap of more than IDLE_RESET without a command
    means the arm was put back on position hold, so the ramp restarts from rest.
    """

    IDLE_RESET = 0.05  # s

    def __init__(self, limb, joint_names, acceleration_limits, dt):
        self._limb, self._joint_names = limb, joint_names
        self._acceleration = np.asarray(acceleration_limits, dtype=np.float32)
        self._dt = float(dt)
        self._last = np.zeros(len(joint_names), dtype=np.float32)
        self._last_time = 0.0

    def command(self, qdot):
        """Ramp toward qdot (rad/s), send it, and return what was sent."""
        now = time.monotonic()
        dt = now - self._last_time
        if dt > self.IDLE_RESET:
            self._last = np.zeros_like(self._last)
            dt = self._dt
        change = np.asarray(qdot, dtype=np.float32) - self._last
        ratio = float(np.max(np.abs(change) / (self._acceleration * dt)))
        sent = self._last + change / max(ratio, 1.0)
        self._last, self._last_time = sent, now
        self._limb.set_joint_velocities({name: float(sent[i]) for i, name in enumerate(self._joint_names)})
        return sent

    def at_rest(self):
        """The arm is held still in velocity mode right now (after going home): ramp from rest."""
        self._last, self._last_time = np.zeros_like(self._last), time.monotonic()

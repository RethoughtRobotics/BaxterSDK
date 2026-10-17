"""Jerk-limited joint setpoint that follows a commanded joint velocity.

The teleop's IK produces a joint velocity every control cycle (~57 Hz, jittery). Baxter's
RAW_POSITION_MODE wants positions, and its inverse dynamics feedforward wants (q, qdot, qddot)
that agree with each other. Integrating the IK velocity per cycle gives a staircase setpoint
and a noisy finite-difference qddot, which roughly doubled 2.5-50 Hz wrist vibration on the
robot (bags 20261016_200134 / _200606).

This generator runs at its own fixed rate and treats each IK velocity as a *target*: Ruckig's
velocity control interface plans a time-optimal, jerk-limited transition to it within the
joint velocity, acceleration and jerk limits, so q, qdot and qddot are continuous and
consistent by construction.

  Berscheid, Kroger, "Jerk-limited Real-time Trajectory Generation with Arbitrary Target
  States", RSS 2021. https://github.com/pantor/ruckig (pip: ruckig, community version).

Joint position limits are not part of the Ruckig problem here; they are enforced after each
step by stopping the joint at the limit (the IK already brakes toward limits, so this is a
backstop, not the primary mechanism).
"""

from dataclasses import dataclass

import numpy as np
from ruckig import ControlInterface, InputParameter, OutputParameter, Result, Ruckig


@dataclass(frozen=True)
class JointLimits:
    """Per-joint kinematic limits, all positive except the position bounds."""

    velocity: np.ndarray  # rad/s
    acceleration: np.ndarray  # rad/s^2
    jerk: np.ndarray  # rad/s^3
    position_low: np.ndarray  # rad
    position_high: np.ndarray  # rad

    def __post_init__(self):
        n = len(self.velocity)
        for name in ('velocity', 'acceleration', 'jerk', 'position_low', 'position_high'):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (n,):
                raise ValueError(f'{name} must have shape ({n},), got {value.shape}')
            object.__setattr__(self, name, value)
        if np.any(self.velocity <= 0) or np.any(self.acceleration <= 0) or np.any(self.jerk <= 0):
            raise ValueError('velocity, acceleration and jerk limits must be positive')
        if np.any(self.position_low >= self.position_high):
            raise ValueError('position_low must be below position_high')


@dataclass(frozen=True)
class Setpoint:
    """Setpoint at one instant: position, velocity and acceleration (rad, rad/s, rad/s^2)."""

    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray


class JointSetpointGenerator:
    """Steps a jerk-limited joint setpoint toward a target velocity at a fixed period `dt`.

    Usage: reset() from the measured arm, then set_target_velocity() whenever a new command
    arrives and step() once per period. Not thread-safe; the caller serialises access.
    """

    def __init__(self, limits: JointLimits, dt: float):
        if dt <= 0:
            raise ValueError('dt must be positive')
        self.limits = limits
        self.dt = float(dt)
        self.dof = len(limits.velocity)
        self._otg = Ruckig(self.dof, self.dt)
        self._input = InputParameter(self.dof)
        self._output = OutputParameter(self.dof)
        self._input.control_interface = ControlInterface.Velocity
        self._input.max_velocity = limits.velocity.tolist()
        self._input.max_acceleration = limits.acceleration.tolist()
        self._input.max_jerk = limits.jerk.tolist()
        self._input.target_acceleration = [0.0] * self.dof
        self._state = None

    @property
    def state(self) -> Setpoint:
        if self._state is None:
            raise RuntimeError('reset() the generator before reading its state')
        return self._state

    def reset(self, position, velocity=None) -> None:
        """Restart from `position` (e.g. the measured arm), at rest unless `velocity` is given."""
        position = np.clip(np.asarray(position, dtype=float), self.limits.position_low, self.limits.position_high)
        velocity = np.zeros(self.dof) if velocity is None else np.asarray(velocity, dtype=float)
        self._set_current(position, velocity, np.zeros(self.dof))
        self._input.target_velocity = velocity.tolist()

    def set_target_velocity(self, velocity) -> None:
        """Velocity to converge to, clipped to the velocity limits (keeps the direction per joint)."""
        velocity = np.clip(np.asarray(velocity, dtype=float), -self.limits.velocity, self.limits.velocity)
        self._input.target_velocity = velocity.tolist()

    def step(self) -> Setpoint:
        """Advance one period and return the new setpoint."""
        if self._state is None:
            raise RuntimeError('reset() the generator before step()')
        result = self._otg.update(self._input, self._output)
        if result not in (Result.Working, Result.Finished):
            raise RuntimeError(f'Ruckig failed: {result}')
        q = np.asarray(self._output.new_position)
        v = np.asarray(self._output.new_velocity)
        a = np.asarray(self._output.new_acceleration)
        # Position limit backstop: stop the joint at the bound. Zeroing its acceleration breaks
        # jerk continuity for that joint only, which is acceptable at a hard limit.
        low, high = self.limits.position_low, self.limits.position_high
        hit = ((q <= low) & (v < 0)) | ((q >= high) & (v > 0))
        if np.any(hit):
            q = np.clip(q, low, high)
            v, a = np.where(hit, 0.0, v), np.where(hit, 0.0, a)
        self._set_current(q, v, a)
        return self._state

    def hold_within(self, reference, max_lead) -> bool:
        """Clamp the setpoint to within max_lead (rad) of `reference` (the measured arm).

        Returns True if it had to clamp: the arm is not following (blocked, in contact), and the
        setpoint is pulled back instead of winding up. Clamped joints restart at rest.
        """
        q = self.state.position
        reference = np.asarray(reference, dtype=float)
        clamped = np.clip(q, reference - max_lead, reference + max_lead)
        moved = clamped != q
        if not np.any(moved):
            return False
        s = self.state
        self._set_current(clamped, np.where(moved, 0.0, s.velocity), np.where(moved, 0.0, s.acceleration))
        return True

    def _set_current(self, q, v, a) -> None:
        self._input.current_position = q.tolist()
        self._input.current_velocity = v.tolist()
        self._input.current_acceleration = a.tolist()
        self._state = Setpoint(q.copy(), v.copy(), a.copy())

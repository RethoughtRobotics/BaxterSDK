"""Validation policy for cartesian delta command streams."""

import numpy as np


class CartesianDeltaGuard:
    """Stateful guard that validates modality deltas before IK is applied."""

    def __init__(
        self,
        dt,
        max_linear_velocity,
        max_angular_velocity,
        max_linear_delta,
        max_angular_delta,
        max_delta_jump,
    ):
        self.dt = float(dt)
        self.max_linear_velocity = float(max_linear_velocity)
        self.max_angular_velocity = float(max_angular_velocity)
        self.max_linear_delta = float(max_linear_delta)
        self.max_angular_delta = float(max_angular_delta)
        self.max_delta_jump = float(max_delta_jump)
        self._last_accepted_delta = np.zeros(6, dtype=np.float32)

    def validate(self, delta, logger=None):
        """Return True when delta is safe to apply, otherwise False."""
        linear_norm = float(np.linalg.norm(delta[:3]))
        angular_norm = float(np.linalg.norm(delta[3:]))

        if linear_norm > self.max_linear_delta:
            self._warn(
                logger,
                f'Rejected delta: linear magnitude {linear_norm:.6f} exceeds {self.max_linear_delta:.6f}',
            )
            return False

        if angular_norm > self.max_angular_delta:
            self._warn(
                logger,
                f'Rejected delta: angular magnitude {angular_norm:.6f} exceeds {self.max_angular_delta:.6f}',
            )
            return False

        linear_velocity = linear_norm / self.dt
        angular_velocity = angular_norm / self.dt

        if linear_velocity > self.max_linear_velocity:
            self._warn(
                logger,
                f'Rejected delta: linear velocity {linear_velocity:.6f} exceeds {self.max_linear_velocity:.6f}',
            )
            return False

        if angular_velocity > self.max_angular_velocity:
            self._warn(
                logger,
                f'Rejected delta: angular velocity {angular_velocity:.6f} exceeds {self.max_angular_velocity:.6f}',
            )
            return False

        jump = float(np.linalg.norm(delta - self._last_accepted_delta))
        if jump > self.max_delta_jump:
            self._warn(logger, f'Rejected delta: jump {jump:.6f} exceeds {self.max_delta_jump:.6f}')
            return False

        return True

    def mark_accepted(self, delta):
        """Update internal continuity state after a command is accepted."""
        self._last_accepted_delta = np.asarray(delta, dtype=np.float32)

    @staticmethod
    def _warn(logger, msg):
        if logger is not None:
            logger.warn(msg)

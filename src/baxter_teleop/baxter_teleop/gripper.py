"""The arm's electric gripper as the teleop drives it: keyboard steps, the VR trigger, the cuff buttons."""

import time

import numpy as np

import baxter_interface


class GripperControl:
    """Commands one arm's electric gripper, as an opening in percent (0 closed, 100 open).

    With enabled False (--no-gripper) the VR trigger does not drive it and going home does not open
    it; the keyboard keys and the cuff buttons still do. Without an electric gripper every command
    is a logged no-op.
    """

    VELOCITY = 80.0  # % of the electric gripper's max speed (SDK default 50)

    def __init__(self, node, arm, step=2.0):
        self._node = node
        self.step = float(step)  # % per keypress
        self.enabled = True
        self.target = None  # last commanded opening (%)
        self._filtered = self._sent = self._filtered_at = None  # trigger filter state
        self._gripper = None
        try:
            self._gripper = baxter_interface.Gripper(arm, node=node)
        except Exception as exc:
            self._log().warn(f'Gripper control unavailable: {exc}')
            return
        if not self.electric:
            self._log().info(f'Gripper delta control only supports electric grippers; detected {self._gripper.type()}.')
            return
        self._gripper.set_velocity(self.VELOCITY)
        if self._gripper.calibrated():
            self.target = float(self._gripper.position())
        else:
            self._log().info('Gripper is not calibrated; press G before using delta controls.')

    def _log(self):
        return self._node.get_logger()

    @property
    def electric(self):
        return self._gripper is not None and self._gripper.type() == 'electric'

    @property
    def calibrated(self):
        return self.electric and self._gripper.calibrated()

    def calibrate(self):
        if self._gripper is None:
            self._log().warn('Gripper control is unavailable.')
            return
        if not self.electric:
            self._log().warn('Gripper calibration is only supported for electric grippers.')
            return
        if self._gripper.calibrate(block=True):
            self.target = float(self._gripper.position())

    def step_by(self, direction):
        """Open (+1) or close (-1) by one keyboard step."""
        if self._gripper is None:
            self._log().warn('Gripper control is unavailable.')
            return
        if not self.electric:
            self._log().warn('Gripper delta control only supports electric grippers.')
            return
        if not self._gripper.calibrated():
            self._log().warn('Gripper is not calibrated; press G to calibrate first.')
            return
        if self.target is None:
            self.target = float(self._gripper.position())
        self.target = float(np.clip(self.target + direction * self.step, 0.0, 100.0))
        self._gripper.command_position(self.target, block=False)

    def open_for_home(self):
        """Open fully on the way home (if enabled); the trigger filter restarts open."""
        if not (self.enabled and self.calibrated):
            return
        self._gripper.command_position(100.0, block=False)
        self.target = 100.0
        if self._filtered is not None:
            self._filtered, self._sent = 100.0, None

    def cuff_button(self, action):
        """Open or close fully from a cuff button ('open' or 'close'), if calibrated and ready."""
        if self._gripper is None or not (self._gripper.calibrated() and self._gripper.ready()):
            return
        (self._gripper.close if action == 'close' else self._gripper.open)()
        self.target = 0.0 if action == 'close' else 100.0

    def follow_trigger(self, trigger, smoothing=0.05, min_step=0.5, min_period=0.02, stale=0.5):
        """Opening from the VR trigger, (value 0..1, receive time) or None: first-order filtered
        (smoothing, s) and sent when it changed by min_step (%) at most every min_period (s); holds
        when the trigger is stale."""
        if not self.enabled or trigger is None or not self.calibrated:
            return
        value, received = trigger
        now = time.monotonic()
        if now - received > stale:
            return
        goal = 100.0 * (1.0 - float(np.clip(value, 0.0, 1.0)))  # 0 closed .. 100 open
        if self._filtered is None:
            self._filtered, self._sent, self._filtered_at = goal, None, now
        else:
            self._filtered += (1.0 - np.exp(-(now - self._filtered_at) / smoothing)) * (goal - self._filtered)
            self._filtered_at = now
        due = self._sent is None or (
            abs(self._filtered - self._sent[0]) >= min_step and now - self._sent[1] >= min_period
        )
        if due:
            self._gripper.command_position(self._filtered, block=False)
            self._sent = (self._filtered, now)
            self.target = self._filtered

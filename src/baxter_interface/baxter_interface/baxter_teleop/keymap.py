"""Keyboard mapping utilities for EE teleop."""

import time

import numpy as np
from scipy.spatial.transform import Rotation

from .pose_servo import servo_twist


def twist_from_key(key, linear_speed, angular_speed):
    """Convert a keyboard key to a 6D EE twist command."""
    v = np.zeros(6, dtype=np.float32)
    k = key.lower()

    # This map is intentionally explicit because the key layout is part of the
    # user interface contract for manual teleoperation.
    if k == 'w':
        v[0] = linear_speed
    elif k == 's':
        v[0] = -linear_speed
    elif k == 'a':
        v[1] = linear_speed
    elif k == 'd':
        v[1] = -linear_speed
    elif k == 'r':
        v[2] = linear_speed
    elif k == 'f':
        v[2] = -linear_speed
    elif k == 'x':
        v[3] = angular_speed
    elif k == 'v':
        v[3] = -angular_speed
    elif k == 'z':
        v[4] = angular_speed
    elif k == 'c':
        v[4] = -angular_speed
    elif k == 'q':
        v[5] = angular_speed
    elif k == 'e':
        v[5] = -angular_speed
    return v


class KeyTarget:
    """Closed-loop keyboard: a held key moves a goal pose at the key speed, and the TCP is servoed
    onto it (servo_twist, with the key twist as feedforward). Lag, blocked motion and IK error are
    pulled back instead of accumulating, and the orientation holds when only translating.

    Same interface as VRTarget: twist(tcp_position, tcp_rotation) -> base-frame twist or None to
    hold, and goal. A terminal cannot see key release, so a key stays held for hold_initial after
    the first character (the OS auto-repeat delay) and hold_repeat after each repeat. Released,
    the TCP settles onto the goal while it keeps getting closer; once it reaches the goal, stops
    getting closer for `stall` seconds (a joint limit or an obstacle makes the goal unreachable)
    or settle_timeout passes, the goal is dropped and the arm holds where it is.
    """

    def __init__(
        self,
        frame,
        linear_speed,
        angular_speed,
        gain=5.0,
        hold_initial=0.55,
        hold_repeat=0.10,
        max_lead=(0.05, 0.2),
        settle=(0.002, 0.01),
        stall=0.2,
        settle_timeout=2.0,
    ):
        self.frame = frame  # 'base' or 'tool': the axes keys move in
        self.linear_speed, self.angular_speed = float(linear_speed), float(angular_speed)
        self.gain = float(gain)
        self.hold_initial, self.hold_repeat = float(hold_initial), float(hold_repeat)
        self.max_lead, self.settle, self.settle_timeout = max_lead, settle, float(settle_timeout)
        self.stall = float(stall)
        self.closest, self.closer_at = None, 0.0  # settling progress: best error so far, and when
        self.goal = None  # (position, Rotation), base frame
        self.key, self.key_twist, self.held_until, self.released_at, self.last = None, None, 0.0, 0.0, 0.0

    def press(self, key):
        """A motion key character arrived; returns False if `key` is not a motion key."""
        now = time.monotonic()
        twist = twist_from_key(key, self.linear_speed, self.angular_speed)
        if not np.any(twist):
            return False
        repeating = key == self.key and now <= self.held_until
        self.key, self.key_twist = key, twist
        self.held_until = now + (self.hold_repeat if repeating else self.hold_initial)
        return True

    def reset(self):
        """Drop the goal and any held key: the arm holds where it is (SPACE, H)."""
        self.goal, self.key, self.held_until = None, None, 0.0

    def twist(self, tcp_position, tcp_rotation):
        """Base-frame TCP twist toward the goal, or None to hold position."""
        now = time.monotonic()
        tcp_rotation = Rotation.from_matrix(tcp_rotation)
        held = now < self.held_until
        if self.goal is None:
            if not held:
                return None
            self.goal, self.last = (np.asarray(tcp_position, float).copy(), tcp_rotation), now
        dt, self.last = min(now - self.last, 0.1), now
        feedforward = np.zeros(6)
        if held:
            v, w = self.key_twist[:3].astype(float), self.key_twist[3:].astype(float)
            if self.frame == 'tool':
                R = tcp_rotation.as_matrix()
                v, w = R @ v, R @ w
            position = self.goal[0] + v * dt
            rotation = Rotation.from_rotvec(w * dt) * self.goal[1]
            # Keep the goal within max_lead of the TCP, so holding a key against a limit or an
            # obstacle does not wind it up.
            lead = position - tcp_position
            position = tcp_position + lead * min(1.0, self.max_lead[0] / max(np.linalg.norm(lead), 1e-9))
            turn = (rotation * tcp_rotation.inv()).as_rotvec()
            rotation = (
                Rotation.from_rotvec(turn * min(1.0, self.max_lead[1] / max(np.linalg.norm(turn), 1e-9))) * tcp_rotation
            )
            self.goal, feedforward = (position, rotation), np.concatenate([v, w])
            self.released_at, self.closest = now, None
        else:
            error = (np.linalg.norm(self.goal[0] - tcp_position), (self.goal[1] * tcp_rotation.inv()).magnitude())
            if self.closest is None or error[0] < self.closest[0] - 1e-3 or error[1] < self.closest[1] - 5e-3:
                self.closest, self.closer_at = error, now  # still getting closer (by 1 mm or 0.3 deg)
            settled = error[0] < self.settle[0] and error[1] < self.settle[1]
            if settled or now - self.closer_at > self.stall or now - self.released_at > self.settle_timeout:
                self.goal = None
                return None
        return servo_twist(
            self.goal,
            tcp_position,
            tcp_rotation,
            self.gain,
            2.0 * self.linear_speed,
            2.0 * self.angular_speed,
            feedforward=feedforward,
        )

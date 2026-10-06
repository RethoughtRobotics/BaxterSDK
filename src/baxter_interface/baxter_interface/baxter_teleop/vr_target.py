"""VR adapter: servo the TCP toward the clutched target published by libsurvive_ros2."""

import time

import numpy as np
import yaml
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation

from .pose_servo import servo_twist


def signed_permutation(axes, config):
    """3x3 matrix whose row i picks the signed controller axis named by axes[i], e.g. '-y'."""
    matrix = np.zeros((3, 3))
    for row, axis in enumerate(axes):
        matrix[row, 'xyz'.index(axis[1])] = -1.0 if axis[0] == '-' else 1.0
    if not np.array_equal(np.abs(matrix).sum(axis=0), np.ones(3)):
        raise ValueError(f'{config}: axes must use each of x, y, z once, got {axes}')
    return matrix


class VRTarget:
    """Turns the VR target pose into a base-frame TCP twist.

    Contract (libsurvive_ros2, /vive/{arm}/target): a PoseStamped in the hand's
    anchor frame, republished with every tracker update, that is exactly frozen
    while the trackpad clutch is released. A new message with an unchanged pose
    therefore means "released": the arm holds and is re-referenced so the current
    target maps onto the current TCP, and the next stroke continues from there.

    Rotation is applied in TCP axes. Translation follows `linear_frame` in `config`:
      tool (default)  controller axes at clutch -> TCP axes at clutch (config/vr_axes.yaml).
                      The map follows the wrist, so rotating it re-aims later strokes.
      base            anchor axes (set by pressing A) -> robot base axes, one fixed map
                      (config/vr_axes_base.yaml). Neither the grip nor the wrist re-aims it.
    Either way it does not depend on where the base stations are. `config` also holds the
    signed axis permutations for translation and rotation, per TCP axis velocity scales,
    per TCP axis deadbands subtracted from the stroke delta, and the smoothing applied to
    the target while the clutch is held.
    """

    def __init__(self, node, topic, config, gain, max_linear, max_angular, stale_after=0.2):
        with open(config) as f:
            cfg = yaml.safe_load(f)
        self.linear_frame = cfg.get('linear_frame', 'tool')
        if self.linear_frame not in ('tool', 'base'):
            raise ValueError(f"{config}: linear_frame must be 'tool' or 'base', got {self.linear_frame}")
        self.linear_axes = signed_permutation(cfg['linear_axes'], config)
        self.angular_axes = signed_permutation(cfg['angular_axes'], config)
        self.scale = np.asarray(cfg['linear_scale'] + cfg['angular_scale'], dtype=float)
        self.deadband = np.asarray(cfg['linear_deadband'] + cfg['angular_deadband'], dtype=float)
        self.smoothing = float(cfg['smoothing'])
        self.gain = float(gain)
        self.max_linear = float(max_linear)
        self.max_angular = float(max_angular)
        self.stale_after = float(stale_after)
        self.reference = None  # (TCP position, TCP Rotation, offset, rotation) when the clutch engaged
        self.goal = None  # (position, Rotation) the TCP is servoed toward this cycle, base frame
        self._pose = None
        self._filtered = None  # (offset, Rotation), smoothed while the clutch is held
        self._moving = False
        self._received = 0.0
        node.create_subscription(PoseStamped, topic, self._on_target, 10)

    def _on_target(self, msg):
        p, o = msg.pose.position, msg.pose.orientation
        pose = (p.x, p.y, p.z, o.x, o.y, o.z, o.w)
        self._moving = self._pose is not None and pose != self._pose
        self._pose = pose
        self._received = time.monotonic()
        # Clutch state is decided on the raw pose; only the stroke itself is smoothed.
        offset, rotation = np.asarray(pose[:3]), Rotation.from_quat(pose[3:])
        if not self._moving or self._filtered is None:
            self._filtered = (offset, rotation)
        else:
            s = self.smoothing
            filtered_offset, filtered_rotation = self._filtered
            self._filtered = (
                s * filtered_offset + (1.0 - s) * offset,
                filtered_rotation * Rotation.from_rotvec((1.0 - s) * (filtered_rotation.inv() * rotation).as_rotvec()),
            )

    def twist(self, tcp_position, tcp_rotation):
        """Base-frame TCP twist [v, w] toward the target, or None to hold position."""
        self.goal = None
        if self._pose is None or time.monotonic() - self._received > self.stale_after:
            return None
        offset, rotation = self._filtered
        tcp_rotation = Rotation.from_matrix(tcp_rotation)

        if not self._moving or self.reference is None:
            self.reference = (tcp_position, tcp_rotation, offset, rotation)
            return None

        # Stroke delta in the controller's axes at clutch time, mapped onto TCP axes, minus a
        # per-axis deadband
        p0, R0, offset0, rotation0 = self.reference
        base = self.linear_frame == 'base'
        delta = np.concatenate([
            self.linear_axes @ (offset - offset0 if base else rotation0.inv().apply(offset - offset0)),
            self.angular_axes @ (rotation0.inv() * rotation).as_rotvec(),
        ])
        delta = np.sign(delta) * np.maximum(np.abs(delta) - self.deadband, 0.0)
        self.goal = (p0 + (delta[:3] if base else R0.apply(delta[:3])), R0 * Rotation.from_rotvec(delta[3:]))
        return servo_twist(
            self.goal, tcp_position, tcp_rotation, self.gain, self.max_linear, self.max_angular, self.scale
        )

    def reset(self):
        """Re-anchor on the next cycle (e.g. after the arm was moved home)."""
        self.reference = None

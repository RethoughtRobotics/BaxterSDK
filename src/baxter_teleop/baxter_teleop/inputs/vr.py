"""VR adapter: servo the TCP toward the clutched target published by libsurvive_ros2."""

import time

import numpy as np
import yaml
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation
from std_msgs.msg import Empty, Float32

from ..servo import servo_twist


def signed_permutation(axes, config):
    """3x3 matrix whose row i picks the signed controller axis named by axes[i], e.g. '-y'."""
    if len(axes) != 3 or not all(isinstance(a, str) and len(a) == 2 and a[0] in '+-' and a[1] in 'xyz' for a in axes):
        raise ValueError(f"{config}: axes must be three of '+x', '-y', ... (sign and axis), got {axes}")
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
    while the trackpad clutch is released. A pose unchanged for `release_after` therefore
    means "released": the arm holds and is re-referenced so the current target maps onto
    the current TCP, and the next stroke continues from there. A single repeated pose does
    not: a controller tracked at a high rate (~230 Hz) repeats its pose now and then while
    clutched (left_v_right bag: 11% of messages), and treating each as a release dropped
    the stroke every few cycles.

    Translation and rotation follow `linear_frame` in `config`:
      tool (default)  controller axes at clutch -> TCP axes at clutch (config/vr_axes.yaml), for
                      both translation and rotation. The map follows the wrist, so rotating it
                      re-aims later strokes.
      base            anchor axes (set by pressing A; libsurvive_ros2 levels the anchor: x your
                      heading, y your left, z up) -> robot base axes (config/vr_axes_base.yaml),
                      for both: a hand move or wrist turn about your forward/left/up moves or
                      turns the TCP along/about the base axes it maps to. Neither the grip nor
                      the wrist re-aims it; pressing A again does.
    Either way it does not depend on where the base stations are. `config` also holds the
    signed axis permutations for translation and rotation, per TCP axis velocity scales,
    per TCP axis deadbands subtracted from the stroke delta, and the smoothing applied to
    the target while the clutch is held.
    """

    def __init__(
        self, node, topic, config, gain, max_linear, max_angular, stale_after=0.2, release_after=0.05, swap_hands=None
    ):
        with open(config) as f:
            cfg = yaml.safe_load(f)
        self.linear_frame = cfg.get('linear_frame', 'tool')
        if self.linear_frame not in ('tool', 'base'):
            raise ValueError(f"{config}: linear_frame must be 'tool' or 'base', got {self.linear_frame}")
        # base_yaw turns the anchor about up before the translation map, e.g. 180 when facing Baxter.
        base_yaw = float(cfg.get('base_yaw', 0.0))
        if base_yaw and self.linear_frame != 'base':
            raise ValueError(f'{config}: base_yaw needs linear_frame: base, got {self.linear_frame}')
        yaw = Rotation.from_euler('z', base_yaw, degrees=True).as_matrix()
        self.linear_axes = signed_permutation(cfg['linear_axes'], config) @ yaw
        self.angular_axes = signed_permutation(cfg['angular_axes'], config)
        # swap_hands: true is the operator facing Baxter (hands crossed to the arms, mirrored), for which
        # linear_axes/angular_axes are written. swap_hands: false is the operator on Baxter's side, a
        # 180-degree turn about up from that: x and y of both maps are negated (z kept, so up stays
        # up and the rotation map stays proper). Base frame only: tool-frame maps follow the wrist.
        # swap_hands (not None) overrides the config's, e.g. run_teleop.sh --swap.
        self.swap_hands = bool(cfg.get('swap_hands', False)) if swap_hands is None else bool(swap_hands)
        if self.linear_frame == 'base' and not self.swap_hands:
            turn = np.diag([-1.0, -1.0, 1.0])
            self.linear_axes = turn @ self.linear_axes
            self.angular_axes = turn @ self.angular_axes
        self.scale = np.asarray(cfg['linear_scale'] + cfg['angular_scale'], dtype=float)
        self.deadband = np.asarray(cfg['linear_deadband'] + cfg['angular_deadband'], dtype=float)
        # Motion scaling (precision mode): the TCP moves motion_scale x the hand's translation, so hand
        # sway shrinks by the same factor without the lag a low-pass filter would add, as surgical
        # teleoperation scales master motion. Rotation stays 1:1. Applied before the deadband (robot units).
        self.motion_scale = float(cfg.get('motion_scale', 1.0))
        if not self.motion_scale > 0.0:
            raise ValueError(f'{config}: motion_scale must be positive, got {self.motion_scale}')
        self.smoothing = float(cfg['smoothing'])
        self.gain = float(gain)
        self.max_linear = float(max_linear)
        self.max_angular = float(max_angular)
        self.stale_after = float(stale_after)
        # A config may lengthen it, e.g. a policy stream, where a delivery hiccup must not re-anchor.
        self.release_after = float(cfg.get('release_after', release_after))
        self.reference = None  # (TCP position, TCP Rotation, offset, rotation) when the clutch engaged
        self.goal = None  # (position, Rotation) the TCP is servoed toward this cycle, base frame
        self._pose = None
        self._filtered = None  # (offset, Rotation), smoothed while the clutch is held
        self._moving = False
        self._received = 0.0
        self._changed = -np.inf  # when the pose last changed
        node.create_subscription(PoseStamped, topic, self._on_target, 10)

    def _on_target(self, msg):
        p, o = msg.pose.position, msg.pose.orientation
        pose = (p.x, p.y, p.z, o.x, o.y, o.z, o.w)
        if not np.all(np.isfinite(pose)):  # lost tracking: ignore, so the target goes stale and the arm holds
            return
        now = time.monotonic()
        if self._pose is not None and pose != self._pose:
            self._changed = now
        self._moving = now - self._changed < self.release_after
        self._pose = pose
        self._received = now
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
            self.reference = None  # the next stream anchors at the TCP where it starts
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
            self.motion_scale
            * (self.linear_axes @ (offset - offset0 if base else rotation0.inv().apply(offset - offset0))),
            # base: the wrist turn in anchor axes; tool: in the controller's axes at clutch
            self.angular_axes @ ((rotation * rotation0.inv()) if base else (rotation0.inv() * rotation)).as_rotvec(),
        ])
        delta = np.sign(delta) * np.maximum(np.abs(delta) - self.deadband, 0.0)
        turn = Rotation.from_rotvec(delta[3:])
        self.goal = (p0 + delta[:3], turn * R0) if base else (p0 + R0.apply(delta[:3]), R0 * turn)
        return servo_twist(
            self.goal, tcp_position, tcp_rotation, self.gain, self.max_linear, self.max_angular, self.scale
        )

    def reset(self):
        """Re-anchor on the next cycle (e.g. after the arm was moved home)."""
        self.reference = None


class VRController:
    """The controller's analog trigger and B button (libsurvive_ros2 /vive/{hand}/trigger, b_button).

    The trigger sets the gripper opening (released open, pulled closed): gripper_opening() gives it
    first-order filtered (smoothing, s), when it changed by min_step (%) at most every min_period
    (s), and nothing while the trigger is stale.
    """

    def __init__(self, node, trigger_topic, b_button_topic, smoothing=0.05, min_step=0.5, min_period=0.02, stale=0.5):
        self.trigger = None  # (value 0 released .. 1 pulled, receive time)
        self.smoothing, self.min_step, self.min_period, self.stale = smoothing, min_step, min_period, stale
        self._filtered = self._filtered_at = self._sent = None  # opening filter state
        self._b_pressed = False
        node.create_subscription(Float32, trigger_topic, self._on_trigger, 10)
        node.create_subscription(Empty, b_button_topic, self._on_b_button, 10)

    def gripper_opening(self):
        """The opening to command now (%: 0 closed, 100 open), or None."""
        if self.trigger is None:
            return None
        value, received = self.trigger
        now = time.monotonic()
        if now - received > self.stale:
            return None
        goal = 100.0 * (1.0 - float(np.clip(value, 0.0, 1.0)))
        if self._filtered is None:
            self._filtered, self._sent, self._filtered_at = goal, None, now
        else:
            self._filtered += (1.0 - np.exp(-(now - self._filtered_at) / self.smoothing)) * (goal - self._filtered)
            self._filtered_at = now
        due = self._sent is None or (
            abs(self._filtered - self._sent[0]) >= self.min_step and now - self._sent[1] >= self.min_period
        )
        if not due:
            return None
        self._sent = (self._filtered, now)
        return self._filtered

    def restart_opening(self, opening):
        """The gripper was commanded elsewhere (opened on the way home): filter on from there."""
        if self._filtered is not None:
            self._filtered, self._sent = opening, None

    def take_b_press(self):
        """True once for each B press since the last call."""
        pressed, self._b_pressed = self._b_pressed, False
        return pressed

    def _on_trigger(self, msg):
        self.trigger = (msg.data, time.monotonic())

    def _on_b_button(self, msg):
        self._b_pressed = True

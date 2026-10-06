#!/usr/bin/env python3

"""Cartesian delta teleop for either Baxter arm using frax differential IK."""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, TwistStamped
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import String

import baxter_interface

try:
    import frax
    import jax
    import jax.numpy as jnp
except ImportError as exc:  # pragma: no cover
    frax = None
    jax = None
    jnp = None
    FRAX_IMPORT_ERROR = exc
else:
    FRAX_IMPORT_ERROR = None

from baxter_external_devices.getch import getch

from .delta_guard import CartesianDeltaGuard
from .diff_ik import DiffIKSolver
from .keymap import KeyTarget, twist_from_key
from .urdf_tools import extract_arm_chain_urdf, resolve_baxter_urdf
from .vr_target import VRTarget


class CartesianDeltaTeleop:
    """Runtime that maps cartesian deltas to Baxter joint commands.

    This class is intentionally modality agnostic. Any input source can be used
    as long as it yields a six element cartesian delta vector.

    Two control modes are supported:
      velocity  joint velocity commands from the measured state (default)
      position  accumulated joint position targets
    """

    JOINT_SUFFIXES = ['s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2']
    # URDF joint velocity limits (rad/s) scaled by 0.8 for margin.
    JOINT_VELOCITY_LIMITS = [1.2, 1.2, 1.2, 1.2, 3.2, 3.2, 3.2]
    # Velocity command acceleration limits (rad/s^2), as franka-vr-teleop's Ruckig limits. Step
    # commands (up to 500 rad/s^2 in circle_clockwise_20261001_184937) ring Baxter's wrist.
    JOINT_ACCELERATION_LIMITS = [4.0, 4.0, 4.0, 4.0, 6.0, 6.0, 6.0]
    # URDF joint position limits (rad), except e1: the elbow is kept from straightening past
    # 0.35 rad, where the arm becomes singular (condition number 9-14 bent vs 100-1000+ straight
    # in circle_clockwise_20261001_184937). config/sns_joint_limits.yaml sets the same for SNS.
    JOINT_POSITION_LIMITS = (
        [-1.70168, -2.147, -3.05418, 0.35, -3.059, -1.5708, -3.059],
        [1.70168, 1.047, 3.05418, 2.618, 3.059, 2.094, 3.059],
    )
    MODE_SPEEDS = {'velocity': (0.20, 0.8), 'position': (1.20, 12.0)}
    # VR servo speed caps (m/s, rad/s): above the speeds reached in circle_clockwise_20261001_184937
    # (1.46 m/s, 4.1 rad/s) so they rarely bind; the joint velocity limits still do.
    VR_SPEEDS = (2.0, 5.0)
    # URDF {arm}_gripper frame, used when the robot endpoint cannot be read.
    URDF_TCP_OFFSET = [0.0, 0.0, 0.025]

    def __init__(
        self,
        arm='right',
        node=None,
        mode='velocity',
        ik='dls',
        sns_limits=None,
        frame='base',
        tcp_offset=None,
        linear_speed=None,
        angular_speed=None,
        key_hold_initial=0.55,
        key_hold_repeat=0.10,
        max_linear_velocity=1.40,
        max_angular_velocity=14.00,
        max_linear_delta=0.020,
        max_angular_delta=0.200,
        max_delta_jump=0.300,
        gripper_step=2.0,
    ):
        if frax is None or jax is None or jnp is None:
            raise RuntimeError(
                'frax + jax are required. Install in your runtime env, e.g. uv pip install frax jax jaxlib'
            ) from FRAX_IMPORT_ERROR

        if arm not in ('left', 'right'):
            raise ValueError("arm must be 'left' or 'right'")
        if mode not in self.MODE_SPEEDS:
            raise ValueError("mode must be 'velocity' or 'position'")
        if ik not in ('dls', 'sns'):
            raise ValueError("ik must be 'dls' or 'sns'")
        if ik == 'sns' and mode != 'velocity':
            raise ValueError('SNS-IK is only wired for velocity mode')
        if frame not in ('tool', 'base'):
            raise ValueError("frame must be 'tool' or 'base'")

        self.arm = arm
        self.mode = mode
        self.frame = frame
        self.joint_names = [f'{arm}_{suffix}' for suffix in self.JOINT_SUFFIXES]

        self.node = node or rclpy.create_node(f'{arm}_arm_ee_teleop')
        self.limb = baxter_interface.Limb(arm, node=self.node)
        self.limb.set_joint_position_speed(1.0)
        self.key_source = getch  # (timeout) -> character or None; key_replay swaps in recorded keys
        # Telemetry for record_teleop.sh: what the teleop asked for and computed, every cycle.
        ns = f'/teleop/{arm}/'
        self.tcp_twist_pub = self.node.create_publisher(TwistStamped, ns + 'tcp_twist', 10)
        self.ik_pub = self.node.create_publisher(JointState, ns + 'ik', 10)
        self.goal_pub = self.node.create_publisher(PoseStamped, ns + 'goal', 10)
        self.key_pub = self.node.create_publisher(String, ns + 'key', 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.config_pub = self.node.create_publisher(String, ns + 'config', latched)
        self.gripper_step = float(gripper_step)
        self.gripper = None
        self.gripper_target = None

        try:
            self.gripper = baxter_interface.Gripper(arm, node=self.node)
        except Exception as exc:
            self.node.get_logger().warn(f'Gripper control unavailable: {exc}')
        else:
            if self.gripper.type() != 'electric':
                self.node.get_logger().info(
                    f'Gripper delta control only supports electric grippers; detected {self.gripper.type()}.'
                )
            elif self.gripper.calibrated():
                self.gripper_target = float(self.gripper.position())
            else:
                self.node.get_logger().info('Gripper is not calibrated; press G before using delta controls.')

        baxter_urdf = resolve_baxter_urdf()
        arm_urdf = extract_arm_chain_urdf(baxter_urdf, arm, self.joint_names)

        manipulator_cls = getattr(getattr(frax, 'core', object()), 'manipulator', None)
        if not manipulator_cls or not hasattr(manipulator_cls, 'Manipulator'):
            raise RuntimeError('This frax version does not expose core.manipulator.Manipulator')

        # Build a single arm model so Jacobian dimensions match a seven joint chain.
        self.robot = manipulator_cls.Manipulator(arm_urdf, joint_ordering=self.joint_names)
        self.ee_jacobian = jax.jit(self.robot.ee_jacobian)
        self.ee_transform = jax.jit(self.robot.ee_transform)
        # Compile now so the first keypress does not stall the control loop.
        self.home = baxter_interface.settings.UNTUCK_POSITIONS[arm]
        self.ee_jacobian(jnp.asarray(self.home, dtype=jnp.float32)).block_until_ready()
        self.ee_transform(jnp.asarray(self.home, dtype=jnp.float32)).block_until_ready()

        # TCP as an offset in the {arm}_hand frame. By default it is measured from the
        # robot's own endpoint, which includes the configured gripper and fingers.
        if tcp_offset is None:
            self.tcp_offset = self._measure_tcp_offset()
        else:
            self.tcp_offset = np.array([0.0, 0.0, float(tcp_offset)], dtype=np.float32)

        self.dt = 0.01
        self.ik = DiffIKSolver(dt=self.dt, damping=0.05, max_joint_step=0.04, max_joint_velocity=6.0)
        if ik == 'sns':
            # Rethink's SNS-IK replaces the DLS solve; it brings its own Jacobian and joint limits.
            from .sns_ik import SNSIKSolver

            # Its nullspace bias holds the elbow near the home (untuck) posture.
            self.ik = SNSIKSolver(
                baxter_urdf,
                arm,
                self.joint_names,
                self.dt,
                joint_limits=sns_limits,
                posture=baxter_interface.settings.UNTUCK_POSITIONS[arm],
            )
        self.joint_velocity_limits = np.asarray(self.JOINT_VELOCITY_LIMITS, dtype=np.float32)
        self.joint_acceleration_limits = np.asarray(self.JOINT_ACCELERATION_LIMITS, dtype=np.float32)
        self.joint_position_limits = tuple(np.asarray(limit, dtype=np.float32) for limit in self.JOINT_POSITION_LIMITS)
        self.qdot_cmd = np.zeros(7, dtype=np.float32)  # last velocity command, the ramp's state
        self.qdot_cmd_time = 0.0
        self.ramp_idle_reset = 0.05  # s without a velocity command: the arm was holding, ramp from rest
        # Commanded target accumulates while input streams, so steps add up instead of
        # restarting from the (lagging) measured position each cycle. Position mode only.
        self.q_cmd = None
        self.last_cmd_time = 0.0
        self.cmd_idle_reset = 0.2  # seconds without input before re-syncing to measured
        self.max_cmd_lead = 0.2  # max rad any joint target may lead the measured arm
        # A terminal cannot see key release, so a key stays active for a hold window after
        # each character. The first window covers the OS auto-repeat delay (500 ms default).
        self.key_hold_initial = float(key_hold_initial)
        self.key_hold_repeat = float(key_hold_repeat)
        default_linear, default_angular = self.MODE_SPEEDS[mode]
        self.linear_speed = default_linear if linear_speed is None else float(linear_speed)
        self.angular_speed = default_angular if angular_speed is None else float(angular_speed)

        self.delta_guard = CartesianDeltaGuard(
            dt=self.dt,
            max_linear_velocity=max_linear_velocity,
            max_angular_velocity=max_angular_velocity,
            max_linear_delta=max_linear_delta,
            max_angular_delta=max_angular_delta,
            max_delta_jump=max_delta_jump,
        )

    def _read_key(self):
        key = self.key_source(timeout=self.dt)
        if key is not None:
            self.key_pub.publish(String(data=key))
        return key

    def _publish_cycle(self, twist, frame, q, qdot_ik, qdot_sent):
        """tcp_twist: the TCP twist handed to the IK, in `frame` (base or tool, i.e. {arm}_gripper).
        ik: joint positions used (position), IK joint velocities (velocity), velocities or
        position steps actually sent after the ramp / step limits (effort)."""
        stamp = self.node.get_clock().now().to_msg()
        tw = TwistStamped()
        tw.header.stamp, tw.header.frame_id = stamp, 'base' if frame == 'base' else f'{self.arm}_gripper'
        (
            tw.twist.linear.x,
            tw.twist.linear.y,
            tw.twist.linear.z,
            tw.twist.angular.x,
            tw.twist.angular.y,
            tw.twist.angular.z,
        ) = (float(x) for x in twist)
        self.tcp_twist_pub.publish(tw)
        js = JointState(
            name=self.joint_names,
            position=np.asarray(q, float).tolist(),
            velocity=np.asarray(qdot_ik, float).tolist(),
            effort=np.asarray(qdot_sent, float).tolist(),
        )
        js.header.stamp = stamp
        self.ik_pub.publish(js)

    def publish_goal(self, goal):
        """VR servo goal (position, scipy Rotation) in the base frame."""
        msg = PoseStamped()
        msg.header.stamp, msg.header.frame_id = self.node.get_clock().now().to_msg(), 'base'
        p, o = msg.pose.position, msg.pose.orientation
        p.x, p.y, p.z = (float(x) for x in goal[0])
        o.x, o.y, o.z, o.w = (float(x) for x in goal[1].as_quat())
        self.goal_pub.publish(msg)

    def publish_config(self, config):
        self.config_pub.publish(String(data=json.dumps(config, indent=1, default=str)))

    def _current_q(self):
        """Return the current arm configuration in the model joint order."""
        angles = self.limb.joint_angles()
        return jnp.array([angles[name] for name in self.joint_names], dtype=jnp.float32)

    def _measure_tcp_offset(self, num_samples=10, sample_period=0.02, timeout=3.0):
        """Return the robot endpoint position expressed in the hand frame, averaged over samples."""
        samples = []
        deadline = time.monotonic() + timeout
        next_sample = time.monotonic()
        while len(samples) < num_samples and time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.005)
            if time.monotonic() < next_sample:
                continue
            pose = self.limb.endpoint_pose()
            if pose and 'position' in pose:
                q = jnp.asarray(self._current_q())
                T = np.asarray(self.ee_transform(q), dtype=np.float32)
                p = np.asarray(pose['position'], dtype=np.float32)
                samples.append(T[:3, :3].T @ (p - T[:3, 3]))
                next_sample = time.monotonic() + sample_period
        if not samples:
            self.node.get_logger().warn('No endpoint state received; using URDF gripper frame as TCP.')
            return np.asarray(self.URDF_TCP_OFFSET, dtype=np.float32)
        samples = np.asarray(samples)
        spread_mm = float(np.max(np.linalg.norm(samples - samples.mean(axis=0), axis=1))) * 1000.0
        self.node.get_logger().info(f'TCP offset from {len(samples)} samples, max deviation {spread_mm:.1f} mm')
        return samples.mean(axis=0).astype(np.float32)

    def _hand_twist(self, q, twist, frame):
        """Convert a TCP twist into the base-frame twist of the hand link used by the Jacobian.

        In tool frame the twist is expressed in the TCP axes; in base frame it is
        expressed in the robot base axes. Either way rotation is about the TCP.
        """
        R = np.asarray(self.ee_transform(jnp.asarray(q)), dtype=np.float32)[:3, :3]
        v = np.asarray(twist[:3], dtype=np.float32)
        w = np.asarray(twist[3:], dtype=np.float32)
        if frame == 'tool':
            v = R @ v
            w = R @ w
        # v_tcp = v_hand + w x r, with r the hand-to-TCP vector in base frame.
        v_hand = v - np.cross(w, R @ self.tcp_offset)
        return np.concatenate([v_hand, w]).astype(np.float32)

    def apply_delta_cartesian(self, delta_cartesian):
        """Apply a 6D cartesian delta from any input modality (position mode)."""
        q_meas = np.asarray(self._current_q(), dtype=np.float32)
        now = time.monotonic()
        if self.q_cmd is None or now - self.last_cmd_time > self.cmd_idle_reset:
            self.q_cmd = q_meas
        self.last_cmd_time = now

        J = self.ee_jacobian(jnp.asarray(self.q_cmd))
        delta_hand = self._hand_twist(self.q_cmd, delta_cartesian, self.frame)
        q_next = self.ik.step_delta(q=self.q_cmd, jacobian=J, delta_cartesian=delta_hand)
        q_ik = q_next
        q_next = np.clip(q_next, q_meas - self.max_cmd_lead, q_meas + self.max_cmd_lead)
        self._publish_cycle(
            np.asarray(delta_cartesian) / self.dt,
            self.frame,
            self.q_cmd,
            (q_ik - self.q_cmd) / self.dt,
            (q_next - self.q_cmd) / self.dt,
        )
        self.q_cmd = q_next
        self.limb.set_joint_positions({name: float(q_next[i]) for i, name in enumerate(self.joint_names)})

    def apply_twist_velocity(self, v_ee):
        """Command joint velocities realizing the base-frame TCP twist v_ee (velocity mode)."""
        q_meas = np.asarray(self._current_q(), dtype=np.float32)
        J = self.ee_jacobian(jnp.asarray(q_meas))
        qdot_ik = qdot = self.ik.joint_velocity(
            q=q_meas,
            jacobian=J,
            v_ee=self._hand_twist(q_meas, v_ee, 'base'),
            joint_velocity_limits=self.joint_velocity_limits,
            joint_position_limits=self.joint_position_limits,
            joint_acceleration_limits=self.joint_acceleration_limits,
        )
        # Ramp from the previous command (not the lagging measured velocity) within the joint
        # acceleration limits, scaling the change uniformly to keep its direction. A skipped cycle
        # means the arm was put back on position hold, so the ramp restarts from rest.
        now = time.monotonic()
        dt = now - self.qdot_cmd_time
        if dt > self.ramp_idle_reset:
            self.qdot_cmd = np.zeros(7, dtype=np.float32)
            dt = self.dt
        change = qdot - self.qdot_cmd
        ratio = float(np.max(np.abs(change) / (self.joint_acceleration_limits * dt)))
        qdot = self.qdot_cmd + change / max(ratio, 1.0)
        self.qdot_cmd, self.qdot_cmd_time = qdot, now
        self._publish_cycle(v_ee, 'base', q_meas, qdot_ik, qdot)
        self.limb.set_joint_velocities({name: float(qdot[i]) for i, name in enumerate(self.joint_names)})

    def _go_home(self):
        """Move to the untuck pose (as tuck_arms -u) with a slow, blocking position move."""
        self.q_cmd = None
        self.limb.exit_control_mode()
        self.node.get_logger().info('Moving to the untuck pose...')
        self.limb.set_joint_position_speed(0.3)
        try:
            self.limb.move_to_joint_positions(dict(zip(self.joint_names, self.home)), timeout=15.0)
        finally:
            self.limb.set_joint_position_speed(1.0)
        self.node.get_logger().info('At the untuck pose.')

    def _apply_gripper_delta(self, delta_percent):
        if self.gripper is None:
            self.node.get_logger().warn('Gripper control is unavailable.')
            return
        if self.gripper.type() != 'electric':
            self.node.get_logger().warn('Gripper delta control only supports electric grippers.')
            return
        if not self.gripper.calibrated():
            self.node.get_logger().warn('Gripper is not calibrated; press G to calibrate first.')
            return

        if self.gripper_target is None:
            self.gripper_target = float(self.gripper.position())

        self.gripper_target = float(np.clip(self.gripper_target + delta_percent, 0.0, 100.0))
        self.gripper.command_position(self.gripper_target, block=False)

    def _calibrate_gripper(self):
        if self.gripper is None:
            self.node.get_logger().warn('Gripper control is unavailable.')
            return
        if self.gripper.type() != 'electric':
            self.node.get_logger().warn('Gripper calibration is only supported for electric grippers.')
            return

        if self.gripper.calibrate(block=True):
            self.gripper_target = float(self.gripper.position())

    def _handle_keyboard_key(self, key):
        if key == 'g':
            self._calibrate_gripper()
            return True
        if key == '[':
            self._apply_gripper_delta(-self.gripper_step)
            return True
        if key == ']':
            self._apply_gripper_delta(self.gripper_step)
            return True
        if key == 'h':
            self._go_home()
            return True
        return False

    def run_delta_source(self, delta_source):
        """Run control loop for any modality that yields cartesian deltas (position mode).

        The callback may return None for no command, a six element vector for a
        command, or the string quit to stop the loop.
        """
        while rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.0)
            delta = delta_source()
            if delta is None:
                continue
            if isinstance(delta, str) and delta == 'quit':
                break

            delta = np.asarray(delta, dtype=np.float32)
            if delta.shape != (6,):
                self.node.get_logger().warn(f'Ignoring delta with invalid shape: {delta.shape}')
                continue

            # Small deltas are treated as no op to keep command traffic quiet.
            if np.linalg.norm(delta) < 1e-9:
                continue

            if not self.delta_guard.validate(delta, logger=self.node.get_logger()):
                continue

            self.apply_delta_cartesian(delta)
            self.delta_guard.mark_accepted(delta)

    def _spin_position(self):
        def keyboard_delta_source():
            key = self._read_key()
            if key is None:
                return None
            if key == '\x1b':
                return 'quit'
            if self._handle_keyboard_key(key):
                return None
            if key == ' ':
                self.q_cmd = None
                self.limb.set_joint_positions(self.limb.joint_angles())
                return None
            v_ee = twist_from_key(key, self.linear_speed, self.angular_speed)
            return v_ee * self.dt

        self.run_delta_source(keyboard_delta_source)

    def spin_target(self, target, on_key=None):
        """Servo the TCP to a target pose with velocity commands, for any input: KeyTarget
        (keyboard) or VRTarget. target.twist() gives the base-frame twist toward target.goal, or
        None to hold position. The keyboard keeps gripper keys, H (home), SPACE (stop) and ESC;
        other keys go to on_key.
        """
        self.limb.set_command_timeout(0.2)
        moving = False
        try:
            while rclpy.ok():
                rclpy.spin_once(self.node, timeout_sec=0.0)
                key = self._read_key()
                if key == '\x1b':
                    break
                if key is not None:
                    if key in ('h', ' '):
                        target.reset()
                        moving = False
                    if not self._handle_keyboard_key(key) and on_key is not None:
                        on_key(key)

                T = np.asarray(self.ee_transform(self._current_q()), dtype=np.float32)
                twist = target.twist(T[:3, 3] + T[:3, :3] @ self.tcp_offset, T[:3, :3])
                if target.goal is not None:
                    self.publish_goal(target.goal)
                if twist is not None:
                    self.apply_twist_velocity(twist)
                    moving = True
                elif moving:
                    self.limb.exit_control_mode()  # back to position mode holding the current pose
                    moving = False
        finally:
            if moving:
                self.limb.exit_control_mode()

    def spin_keyboard(self):
        """Closed-loop keyboard: held keys move a goal pose that the TCP is servoed onto."""
        target = KeyTarget(
            self.frame,
            self.linear_speed,
            self.angular_speed,
            hold_initial=self.key_hold_initial,
            hold_repeat=self.key_hold_repeat,
        )

        def on_key(key):
            v = twist_from_key(key, self.linear_speed, self.angular_speed)
            if np.any(v) and self.delta_guard.validate(v * self.dt, logger=self.node.get_logger()):
                target.press(key)

        self.spin_target(target, on_key)

    def spin_vr(self, target):
        """Follow a VRTarget; the keyboard keeps gripper/home/ESC."""
        print(f'\n{self.arm.capitalize()} arm VR teleop ({type(self.ik).__name__}, velocity mode)')
        print('Touch the trackpad to move, release to stop. A re-anchors. H home, ESC quit')
        self.spin_target(target)

    def spin(self):
        print(f'\n{self.arm.capitalize()} arm cartesian-delta teleop (frax diff-IK, {self.mode} mode)')
        print(f'speed {self.linear_speed:.2f} m/s, {self.angular_speed:.2f} rad/s')
        print(f'{self.frame} frame, TCP offset in hand frame {np.round(self.tcp_offset, 3)} m')
        print('W/S X, A/D Y, R/F Z, Q/E yaw, Z/C pitch, X/V roll, SPACE stop, ESC quit')
        print('H go to the untuck pose (home)')
        print('[/ ] gripper delta, G calibrate gripper')

        if self.mode == 'velocity':
            self.spin_keyboard()
        else:
            self._spin_position()


def main():
    if frax is None or jax is None or jnp is None:
        raise RuntimeError(
            'frax + jax are required. Install in your runtime env, e.g. uv pip install frax jax jaxlib'
        ) from FRAX_IMPORT_ERROR

    parser = argparse.ArgumentParser(description='Baxter cartesian-delta teleop')
    parser.add_argument('--arm', choices=['left', 'right'], default='right', help='arm to control (default: right)')
    parser.add_argument('--input', choices=['keyboard', 'vr'], default='keyboard', help='command source')
    parser.add_argument(
        '--vr-topic', default=None, help='VR target topic (default: /vive/{arm}/target from libsurvive_ros2)'
    )
    parser.add_argument(
        '--vr-config',
        default=os.path.join(get_package_share_directory('baxter_interface'), 'config', 'vr_axes.yaml'),
        help='VR mapping YAML, a path or a file in the installed config/ (default: vr_axes.yaml; '
        'vr_axes_base.yaml maps translation onto the robot base frame)',
    )
    parser.add_argument('--vr-gain', type=float, default=3.0, help='VR pose servo gain in 1/s')
    parser.add_argument('--mode', choices=['velocity', 'position'], default='velocity', help='joint control mode')
    parser.add_argument(
        '--ik',
        choices=['dls', 'sns'],
        default='dls',
        help="velocity IK: 'dls' damped least squares (default) or 'sns' Rethink's SNS-IK (velocity mode)",
    )
    parser.add_argument(
        '--sns-limits',
        default=None,
        help="SNS-IK joint limits YAML (default: Rethink's example_baxter_joint_limits.yaml; "
        'config/sns_joint_limits.yaml matches the DLS joint speeds)',
    )
    parser.add_argument(
        '--frame', choices=['tool', 'base'], default='base', help='frame keys are expressed in (default: base)'
    )
    parser.add_argument(
        '--tcp-offset',
        type=float,
        default=None,
        help='TCP distance along hand z in meters (default: measured from robot endpoint)',
    )
    parser.add_argument(
        '--linear-speed', type=float, default=None, help='key speed / VR cap in m/s (default: mode default, VR 2.0)'
    )
    parser.add_argument(
        '--angular-speed', type=float, default=None, help='key speed / VR cap in rad/s (default: mode default, VR 5.0)'
    )
    parser.add_argument(
        '--key-hold-initial', type=float, default=0.55, help='velocity mode: seconds a first keypress stays active'
    )
    parser.add_argument('--max-linear-velocity', type=float, default=1.40, help='max linear velocity in m/s')
    parser.add_argument('--max-angular-velocity', type=float, default=14.00, help='max angular velocity in rad/s')
    parser.add_argument('--max-linear-delta', type=float, default=0.020, help='max linear delta per cycle in meters')
    parser.add_argument('--max-angular-delta', type=float, default=0.200, help='max angular delta per cycle in radians')
    parser.add_argument(
        '--max-delta-jump', type=float, default=0.300, help='max Euclidean jump between consecutive deltas'
    )
    parser.add_argument(
        '--gripper-step', type=float, default=2.0, help='gripper position delta per keypress in percent'
    )
    args, ros_args = parser.parse_known_args(sys.argv[1:])

    if args.input == 'vr':
        if not os.path.exists(args.vr_config):
            args.vr_config = os.path.join(get_package_share_directory('baxter_interface'), 'config', args.vr_config)
        vr_linear, vr_angular = CartesianDeltaTeleop.VR_SPEEDS
        args.linear_speed = vr_linear if args.linear_speed is None else args.linear_speed
        args.angular_speed = vr_angular if args.angular_speed is None else args.angular_speed

    rclpy.init(args=[sys.argv[0], *ros_args])
    teleop = CartesianDeltaTeleop(
        arm=args.arm,
        mode=args.mode,
        ik=args.ik,
        sns_limits=args.sns_limits,
        frame=args.frame,
        tcp_offset=args.tcp_offset,
        linear_speed=args.linear_speed,
        angular_speed=args.angular_speed,
        key_hold_initial=args.key_hold_initial,
        max_linear_velocity=args.max_linear_velocity,
        max_angular_velocity=args.max_angular_velocity,
        max_linear_delta=args.max_linear_delta,
        max_angular_delta=args.max_angular_delta,
        max_delta_jump=args.max_delta_jump,
        gripper_step=args.gripper_step,
    )
    config = {'args': vars(args), 'tcp_offset': teleop.tcp_offset.tolist(), 'ik': type(teleop.ik).__name__}
    if args.input == 'vr':
        with open(args.vr_config) as f:
            config['vr_config'] = f.read()
    src = os.path.dirname(os.path.realpath(__file__))
    config['git'] = subprocess.run(
        ['git', '-C', src, 'describe', '--always', '--dirty'], capture_output=True, text=True
    ).stdout.strip()
    teleop.publish_config(config)
    try:
        if args.input == 'vr':
            teleop.spin_vr(
                VRTarget(
                    teleop.node,
                    args.vr_topic or f'/vive/{args.arm}/target',
                    args.vr_config,
                    args.vr_gain,
                    teleop.linear_speed,
                    teleop.angular_speed,
                )
            )
        else:
            teleop.spin()
    finally:
        teleop.node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

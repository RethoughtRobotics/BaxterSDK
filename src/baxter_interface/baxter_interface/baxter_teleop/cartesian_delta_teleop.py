#!/usr/bin/env python3

"""Cartesian delta teleop for either Baxter arm using frax differential IK."""

import argparse
import sys
import time

import numpy as np
import rclpy

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
from .keymap import twist_from_key
from .urdf_tools import extract_arm_chain_urdf, resolve_baxter_urdf


class CartesianDeltaTeleop:
    """Runtime that maps cartesian deltas to Baxter joint position commands.

    This class is intentionally modality agnostic. Any input source can be used
    as long as it yields a six element cartesian delta vector.
    """

    JOINT_SUFFIXES = ['s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2']
    # Nullspace rest posture: Baxter neutral with a more bent elbow, which roughly
    # doubles the smallest Jacobian singular value (0.087 -> 0.15) vs neutral.
    REST_POSTURE = [0.0, -0.55, 0.0, 1.2, 0.0, 1.26, 0.0]
    E0_INDEX = 2
    E0_LIMIT = 3.0

    def __init__(
        self,
        arm='right',
        node=None,
        null_gain=2.0,
        elbow_step=0.03,
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

        self.arm = arm
        self.joint_names = [f'{arm}_{suffix}' for suffix in self.JOINT_SUFFIXES]

        self.node = node or rclpy.create_node(f'{arm}_arm_ee_teleop')
        self.limb = baxter_interface.Limb(arm, node=self.node)
        self.limb.set_joint_position_speed(1.0)
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
        self.ee_jacobian(jnp.asarray(self.REST_POSTURE, dtype=jnp.float32)).block_until_ready()

        self.dt = 0.01
        self.ik = DiffIKSolver(
            dt=self.dt, damping=0.05, max_joint_step=0.04, max_joint_velocity=6.0, null_gain=null_gain
        )
        self.q_rest = np.asarray(self.REST_POSTURE, dtype=np.float32)
        self.elbow_step = float(elbow_step)
        # Commanded target accumulates while input streams, so steps add up instead of
        # restarting from the (lagging) measured position each cycle.
        self.q_cmd = None
        self.last_cmd_time = 0.0
        self.cmd_idle_reset = 0.2  # seconds without input before re-syncing to measured
        self.max_cmd_lead = 0.2  # max rad any joint target may lead the measured arm
        self.linear_speed = 1.20
        self.angular_speed = 12.0

        self.delta_guard = CartesianDeltaGuard(
            dt=self.dt,
            max_linear_velocity=max_linear_velocity,
            max_angular_velocity=max_angular_velocity,
            max_linear_delta=max_linear_delta,
            max_angular_delta=max_angular_delta,
            max_delta_jump=max_delta_jump,
        )

    def _current_q(self):
        """Return the current arm configuration in the model joint order."""
        angles = self.limb.joint_angles()
        return jnp.array([angles[name] for name in self.joint_names], dtype=jnp.float32)

    def apply_delta_cartesian(self, delta_cartesian):
        """Apply a 6D cartesian delta from any input modality."""
        q_meas = np.asarray(self._current_q(), dtype=np.float32)
        now = time.monotonic()
        if self.q_cmd is None or now - self.last_cmd_time > self.cmd_idle_reset:
            self.q_cmd = q_meas
        self.last_cmd_time = now

        J = self.ee_jacobian(jnp.asarray(self.q_cmd))
        q_next = self.ik.step_delta(q=self.q_cmd, jacobian=J, delta_cartesian=delta_cartesian, q_rest=self.q_rest)
        q_next = np.clip(q_next, q_meas - self.max_cmd_lead, q_meas + self.max_cmd_lead)
        self.q_cmd = q_next
        self.limb.set_joint_positions({name: float(q_next[i]) for i, name in enumerate(self.joint_names)})

    def _shift_elbow(self, direction):
        """Swing the elbow by moving the e0 rest target; the nullspace term follows it."""
        e0 = self.q_rest[self.E0_INDEX] + direction * self.elbow_step
        self.q_rest[self.E0_INDEX] = float(np.clip(e0, -self.E0_LIMIT, self.E0_LIMIT))

    def _go_home(self):
        """Move to the rest posture with a slow, blocking position move."""
        self.q_cmd = None
        self.q_rest = np.asarray(self.REST_POSTURE, dtype=np.float32)
        self.node.get_logger().info('Moving to rest posture...')
        self.limb.set_joint_position_speed(0.3)
        try:
            self.limb.move_to_joint_positions(
                {name: float(self.q_rest[i]) for i, name in enumerate(self.joint_names)}, timeout=15.0
            )
        finally:
            self.limb.set_joint_position_speed(1.0)
        self.node.get_logger().info('At rest posture.')

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
        """Run control loop for any modality that yields cartesian deltas.

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

    def spin(self):
        print(f'\n{self.arm.capitalize()} arm cartesian-delta teleop (frax diff-IK)')
        print('W/S X, A/D Y, R/F Z, Q/E yaw, Z/C pitch, X/V roll, SPACE stop, ESC quit')
        print('T/Y swing elbow, H go to rest posture')
        print('[/ ] gripper delta, G calibrate gripper')

        def keyboard_delta_source():
            key = getch(timeout=self.dt)
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
            if key in ('t', 'y'):
                # Nullspace-only step: zero EE delta, elbow follows the shifted rest target.
                self._shift_elbow(1.0 if key == 't' else -1.0)
                self.apply_delta_cartesian(np.zeros(6, dtype=np.float32))
                return None

            v_ee = twist_from_key(key, self.linear_speed, self.angular_speed)
            return v_ee * self.dt

        self.run_delta_source(keyboard_delta_source)


def main():
    if frax is None or jax is None or jnp is None:
        raise RuntimeError(
            'frax + jax are required. Install in your runtime env, e.g. uv pip install frax jax jaxlib'
        ) from FRAX_IMPORT_ERROR

    parser = argparse.ArgumentParser(description='Baxter cartesian-delta teleop')
    parser.add_argument('--arm', choices=['left', 'right'], default='right', help='arm to control (default: right)')
    parser.add_argument(
        '--null-gain', type=float, default=2.0, help='nullspace pull toward rest posture in 1/s (0 disables)'
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

    rclpy.init(args=[sys.argv[0], *ros_args])
    teleop = CartesianDeltaTeleop(
        arm=args.arm,
        null_gain=args.null_gain,
        max_linear_velocity=args.max_linear_velocity,
        max_angular_velocity=args.max_angular_velocity,
        max_linear_delta=args.max_linear_delta,
        max_angular_delta=args.max_angular_delta,
        max_delta_jump=args.max_delta_jump,
        gripper_step=args.gripper_step,
    )
    try:
        teleop.spin()
    finally:
        teleop.node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

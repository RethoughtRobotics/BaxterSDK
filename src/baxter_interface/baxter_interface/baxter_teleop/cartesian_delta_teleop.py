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
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, TwistStamped
from rclpy.executors import TimeoutException
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty, Float32, String

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
from .tcp import pose_msg, tcp_pose
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
    # Raw output (--output raw/raw_id): jerk limit = JERK_FACTOR x the acceleration limit, so full
    # acceleration builds up in 0.1 s. A tuning choice: short against the ~0.3 s velocity transitions
    # (adds little lag), long enough that the setpoint has no steps for the raw joint loop to ring on.
    JERK_FACTOR = 10.0
    RAW_OUTPUT_RATE = 200.0  # Hz, setpoints per second: ~3.5 per control cycle (~57 Hz)
    # URDF joint position limits (rad), except e1: the elbow is kept from straightening past
    # 0.35 rad, where the arm becomes singular (condition number 9-14 bent vs 100-1000+ straight
    # in circle_clockwise_20261001_184937). config/sns_joint_limits.yaml sets the same for SNS.
    JOINT_POSITION_LIMITS = (
        [-1.70168, -2.147, -3.05418, 0.35, -3.059, -1.5708, -3.059],
        [1.70168, 1.047, 3.05418, 2.618, 3.059, 2.094, 3.059],
    )
    MODE_SPEEDS = {'velocity': (0.20, 0.8), 'position': (1.20, 12.0)}
    GRIPPER_VELOCITY = 80.0  # % of the electric gripper's max speed (SDK default 50)
    # After a position-mode move Baxter ignores velocity commands until ~1.4-1.9 s after it ends
    # (rollouts Clean-SmolVLA*, ACT_Rollout; 20 ms otherwise): wait that out at home.
    HOME_SETTLE = 2.0  # s
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
        output='velocity',
        ik='dls',
        sns_limits=None,
        posture_gain=0.0,
        joint_accel_scale=1.0,
        servo_feedback='measured',
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
        # Kept on the global executor, so rclpy.spin_once (e.g. in Limb waits) no longer adds and
        # removes it, which would discard the callbacks still pending; see _spin_pending.
        self.executor = rclpy.get_global_executor()
        self.executor.add_node(self.node)
        self.limb = baxter_interface.Limb(arm, node=self.node)
        self.limb.set_joint_position_speed(1.0)
        self.key_source = getch  # (timeout) -> character or None; key_replay swaps in recorded keys
        # Telemetry for record_teleop.sh: what the teleop asked for and computed, every cycle.
        ns = f'/teleop/{arm}/'
        self.tcp_twist_pub = self.node.create_publisher(TwistStamped, ns + 'tcp_twist', 10)
        self.ik_pub = self.node.create_publisher(JointState, ns + 'ik', 10)
        self.goal_pub = self.node.create_publisher(PoseStamped, ns + 'goal', 10)
        self.tcp_pub = self.node.create_publisher(PoseStamped, ns + 'tcp', 10)  # measured TCP (tcp.py)
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
            else:
                self.gripper.set_velocity(self.GRIPPER_VELOCITY)
                if self.gripper.calibrated():
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
        # Going home opens the gripper, and the VR trigger drives it (--no-gripper: neither touches it)
        self.use_gripper = True
        self.ee_jacobian(jnp.asarray(self.home, dtype=jnp.float32)).block_until_ready()
        self.ee_transform(jnp.asarray(self.home, dtype=jnp.float32)).block_until_ready()

        # TCP as an offset in the {arm}_hand frame. By default it is measured from the
        # robot's own endpoint, which includes the configured gripper and fingers.
        if tcp_offset is None:
            self.tcp_offset = self._measure_tcp_offset()
        else:
            self.tcp_offset = np.array([0.0, 0.0, float(tcp_offset)], dtype=np.float32)

        self.dt = 0.01
        # posture_gain (1/s) pulls the spare DOF (elbow swing) toward untuck without moving the hand.
        self.ik = DiffIKSolver(
            dt=self.dt,
            damping=0.1,
            singular_threshold=0.1,
            max_joint_step=0.04,
            max_joint_velocity=6.0,
            posture=baxter_interface.settings.UNTUCK_POSITIONS[arm],
            posture_gain=posture_gain,
        )
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
        # joint_accel_scale scales every use of these limits: the IK's braking toward joint limits, the
        # velocity-mode ramp, and the raw-output setpoint generator (whose jerk limit follows, JERK_FACTOR).
        self.joint_acceleration_limits = float(joint_accel_scale) * np.asarray(
            self.JOINT_ACCELERATION_LIMITS, dtype=np.float32
        )
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
        # Velocity-mode output stage. 'velocity': the ramped IK velocity in VELOCITY_MODE. 'raw' /
        # 'raw_id': the IK velocity is the target of a jerk-limited setpoint streamed at
        # RAW_OUTPUT_RATE in RAW_POSITION_MODE, raw_id with inverse dynamics feedforward
        # (raw_output.py). Raw modes bypass Baxter's collision avoidance.
        # 'raw_id_step' is a test-only ablation (control benchmark): the stage raw_id replaced, which
        # integrates the ramped velocity into a setpoint once per control cycle and sends it with a
        # finite-difference qddot, i.e. a ~57 Hz staircase. Kept to measure what the jerk-limited
        # setpoint buys; not for teleoperation.
        if output not in ('velocity', 'raw', 'raw_id', 'raw_id_step'):
            raise ValueError("output must be 'velocity', 'raw', 'raw_id' or 'raw_id_step'")
        self.output = output
        # What the pose servo and IK close on. 'measured': the arm's joint states. 'setpoint': the raw
        # output's own setpoint (raw/raw_id only), so the Cartesian loop is purely kinematic and cannot
        # hunt on joint friction; Baxter's joint loop (with inverse dynamics) tracks the setpoint.
        if servo_feedback not in ('measured', 'setpoint'):
            raise ValueError("servo_feedback must be 'measured' or 'setpoint'")
        if servo_feedback == 'setpoint' and output in ('velocity', 'raw_id_step'):
            raise ValueError("servo_feedback='setpoint' needs a streamed raw output (--output raw or raw_id)")
        self.servo_feedback = servo_feedback
        self.raw_output = None
        self._step_sink, self._q_step, self._qdot_step = None, None, np.zeros(7, dtype=np.float32)
        if output == 'raw_id_step':
            from .raw_output import BaxterRawSink

            self._step_sink = BaxterRawSink(self.node, arm, self.joint_names, self.limb.joint_names())
        elif output != 'velocity':
            # Imported here: baxter_interface imports this module eagerly, and tools such as
            # tuck_arms run on the system Python, which has no ruckig (only .venv-ros does).
            from .joint_setpoint import JointLimits, JointSetpointGenerator
            from .raw_output import BaxterRawSink, RawJointOutput

            limits = JointLimits(
                velocity=self.joint_velocity_limits,
                acceleration=self.joint_acceleration_limits,
                jerk=self.JERK_FACTOR * self.joint_acceleration_limits,
                position_low=self.joint_position_limits[0],
                position_high=self.joint_position_limits[1],
            )
            sink = BaxterRawSink(
                self.node, arm, self.joint_names, self.limb.joint_names(), inverse_dynamics=output == 'raw_id'
            )
            self.raw_output = RawJointOutput(JointSetpointGenerator(limits, 1.0 / self.RAW_OUTPUT_RATE), sink).start()
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

    def _spin_pending(self):
        """Run every callback that is ready now. One per cycle (rclpy.spin_once) falls behind the
        VR target, trigger, joint states and endpoint state together, and the servo then acts on
        joint angles 0.1-0.3 s old, which on the real arm's lag makes the loop unstable. Each wait
        takes one message per subscription, so wait again until a wait finds nothing."""
        ran = True
        while ran:
            ran = False
            try:
                while True:
                    handler, _, _ = self.executor.wait_for_ready_callbacks(timeout_sec=0.0)
                    handler()
                    handler.result()  # raise any exception from the callback
                    ran = True
            except TimeoutException:
                pass

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
        self.goal_pub.publish(pose_msg(goal[0], goal[1], self.node.get_clock().now().to_msg()))

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

    def _servo_q(self, q_meas):
        """Configuration the servo and IK close on: the raw setpoint with servo_feedback='setpoint'
        while it streams, else the measured arm (also for the first cycle of a stroke, which the
        setpoint then starts from)."""
        if self.servo_feedback == 'setpoint' and self.raw_output is not None:
            q_set = self.raw_output.position
            if q_set is not None:
                return q_set.astype(np.float32)
        return np.asarray(q_meas, dtype=np.float32)

    def apply_twist_velocity(self, v_ee):
        """Command joint velocities realizing the base-frame TCP twist v_ee (velocity mode)."""
        q_meas = np.asarray(self._current_q(), dtype=np.float32)
        q = self._servo_q(q_meas)
        J = self.ee_jacobian(jnp.asarray(q))
        qdot_ik = qdot = self.ik.joint_velocity(
            q=q,
            jacobian=J,
            v_ee=self._hand_twist(q, v_ee, 'base'),
            joint_velocity_limits=self.joint_velocity_limits,
            joint_position_limits=self.joint_position_limits,
            joint_acceleration_limits=self.joint_acceleration_limits,
        )
        if self.raw_output is not None:
            # The setpoint generator limits acceleration and jerk itself: no ramp here.
            self.raw_output.command(qdot_ik, q_meas)
            self._publish_cycle(v_ee, 'base', q_meas, qdot_ik, self.raw_output.velocity)
            return
        # Ramp from the previous command (not the lagging measured velocity) within the joint
        # acceleration limits, scaling the change uniformly to keep its direction. A skipped cycle
        # means the arm was put back on position hold, so the ramp restarts from rest.
        now = time.monotonic()
        dt = now - self.qdot_cmd_time
        if dt > self.ramp_idle_reset:
            self.qdot_cmd = np.zeros(7, dtype=np.float32)
            self._q_step = None
            dt = self.dt
        change = qdot - self.qdot_cmd
        ratio = float(np.max(np.abs(change) / (self.joint_acceleration_limits * dt)))
        qdot = self.qdot_cmd + change / max(ratio, 1.0)
        self.qdot_cmd, self.qdot_cmd_time = qdot, now
        self._publish_cycle(v_ee, 'base', q_meas, qdot_ik, qdot)
        if self._step_sink is not None:
            self._send_step(qdot, q_meas, dt)
            return
        self.limb.set_joint_velocities({name: float(qdot[i]) for i, name in enumerate(self.joint_names)})

    def _send_step(self, qdot, q_meas, dt):
        """raw_id_step (ablation): integrate the ramped velocity once per cycle and send it in
        RAW_POSITION_MODE with a finite-difference qddot, as raw_id did before the setpoint generator."""
        from .joint_setpoint import Setpoint

        if self._q_step is None:
            self._q_step, self._qdot_step = q_meas.copy(), np.zeros(7, dtype=np.float32)
        qddot = (qdot - self._qdot_step) / dt
        q = np.clip(self._q_step + qdot * dt, q_meas - self.max_cmd_lead, q_meas + self.max_cmd_lead)
        self._q_step = np.clip(q, *self.joint_position_limits)
        self._qdot_step = qdot
        self._step_sink(Setpoint(self._q_step.astype(float), qdot.astype(float), qddot.astype(float)))

    def _hold(self):
        """Stop the raw setpoint stream, if any, and put the arm on position hold where it is."""
        if self.raw_output is not None:
            self.raw_output.stop()
        self._q_step = None
        self.limb.exit_control_mode()

    def _go_home(self):
        """Open the gripper (if use_gripper) and move to the untuck pose (as tuck_arms -u) with a slow,
        blocking move."""
        self.q_cmd = None
        self._hold()
        if (
            self.use_gripper
            and self.gripper is not None
            and self.gripper.type() == 'electric'
            and self.gripper.calibrated()
        ):
            self.gripper.command_position(100.0, block=False)  # 100 = open
            self.gripper_target = 100.0
            if getattr(self, '_grip_filtered', None) is not None:  # VR: the trigger filter restarts open
                self._grip_filtered, self._grip_sent = 100.0, None
        self.node.get_logger().info('Moving to the untuck pose...')
        self.limb.set_joint_position_speed(0.3)
        try:
            reached = self.limb.move_to_joint_positions(dict(zip(self.joint_names, self.home)), timeout=15.0)
        finally:
            self.limb.set_joint_position_speed(1.0)
        zero = {name: 0.0 for name in self.joint_names}
        end = time.monotonic() + self.HOME_SETTLE
        while time.monotonic() < end:  # hold in velocity mode, so the next stroke moves at once
            self.limb.set_joint_velocities(zero)
            time.sleep(self.dt)
        self.qdot_cmd, self.qdot_cmd_time = np.zeros(7, dtype=np.float32), time.monotonic()
        if reached:
            self.node.get_logger().info('At the untuck pose.')
        else:
            self.node.get_logger().warning('Did not reach the untuck pose in 15 s: press home again.')

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
            self._spin_pending()
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

    def spin_target(self, target, on_key=None, on_cycle=None):
        """Servo the TCP to a target pose with velocity commands, for any input: KeyTarget
        (keyboard) or VRTarget. target.twist() gives the base-frame twist toward target.goal, or
        None to hold position. The keyboard keeps gripper keys, H (home), SPACE (stop) and ESC;
        other keys go to on_key. on_cycle runs once per control cycle (e.g. the VR gripper) and may
        return a key to act on as if typed (e.g. 'h' for the controller's B button).
        """
        self.limb.set_command_timeout(0.2)
        moving = False
        try:
            while rclpy.ok():
                self._spin_pending()
                injected = on_cycle() if on_cycle is not None else None
                key = self._read_key() or injected
                if key == '\x1b':
                    break
                if key is not None:
                    if key in ('h', ' '):
                        target.reset()
                        moving = False
                    if not self._handle_keyboard_key(key) and on_key is not None:
                        on_key(key)

                q_meas = self._current_q()
                tcp_position, tcp_rotation = tcp_pose(self.ee_transform(q_meas), self.tcp_offset)
                # /teleop/{arm}/tcp stays the measured TCP; the servo closes on _servo_q().
                self.tcp_pub.publish(pose_msg(tcp_position, tcp_rotation, self.node.get_clock().now().to_msg()))
                if self.servo_feedback == 'setpoint':
                    tcp_position, tcp_rotation = tcp_pose(self.ee_transform(self._servo_q(q_meas)), self.tcp_offset)
                twist = target.twist(tcp_position, tcp_rotation)
                if target.goal is not None:
                    self.publish_goal(target.goal)
                if twist is not None:
                    self.apply_twist_velocity(twist)
                    moving = True
                elif moving:
                    self._hold()  # back to position mode holding the current pose
                    moving = False
        finally:
            if moving:
                self._hold()

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

    def spin_vr(self, target, trigger_topic, b_button_topic, home_button_topic=None):
        """Follow a VRTarget; the controller's analog trigger sets the gripper opening (released
        open, pulled closed) and its B button sends the arm home (as H); the keyboard keeps
        gripper/home/ESC. Guided by hand (cuff squeezed), the cuff's buttons drive the gripper."""
        print(f'\n{self.arm.capitalize()} arm VR teleop ({type(self.ik).__name__}, velocity mode)')
        print(
            'Touch the trackpad to move, release to stop. A re-anchors. Trigger closes the gripper. '
            'B, H or the right navigator OK home, ESC quit. By hand: squeeze the cuff to move, its '
            'dash/circle buttons close/open'
        )
        self._trigger = None  # (value 0..1, receive time)
        self.node.create_subscription(
            Float32, trigger_topic, lambda msg: setattr(self, '_trigger', (msg.data, time.monotonic())), 10
        )
        if self.gripper is not None and self.gripper.type() == 'electric' and not self.gripper.calibrated():
            self.node.get_logger().info('Calibrating the gripper...')
            self._calibrate_gripper()
        self._b_pressed = False
        self.node.create_subscription(Empty, b_button_topic, lambda msg: setattr(self, '_b_pressed', True), 10)
        from baxter_core_msgs.msg import DigitalIOState

        if home_button_topic:  # a robot button as B, for every arm's teleop: one press homes them all
            self._home_button = (0, -np.inf)  # last state, when it last went 0

            def on_home_button(msg):
                # Navigator buttons bounce on release: a press counts after 0.2 s released
                prev, released = self._home_button
                if prev and not msg.state:
                    released = time.monotonic()
                elif not prev and msg.state and time.monotonic() - released >= 0.2:
                    self._b_pressed = True
                self._home_button = (msg.state, released)

            self.node.create_subscription(DigitalIOState, home_button_topic, on_home_button, 10)
        # The cuff (zero-g) moves the arm by hand; on release Baxter's position hold would pull it back to
        # where it was held before. Not streaming (no stroke under way): hold where the hand left it.
        self._cuff = self._cuff_seen = False
        self.node.create_subscription(
            DigitalIOState,
            f'/robot/digital_io/{self.arm}_lower_cuff/state',
            lambda msg: setattr(self, '_cuff', bool(msg.state)),
            10,
        )
        self._grip_filtered = self._grip_sent = self._grip_time = None
        # The cuff's buttons drive the gripper, as baxter_examples gripper_cuff_control (kinesthetic
        # demos): dash (upper) closes, circle (lower) opens, on press. A command holds until the next
        # button press or a trigger move.
        self._cuff_button = None  # 'close' or 'open', pressed since the last cycle
        self._cuff_buttons = {}  # topic -> last state, for the press edge

        def on_button(topic, action, state):
            if state and not self._cuff_buttons.get(topic):
                self._cuff_button = action
            self._cuff_buttons[topic] = state

        for button, action in (('upper', 'close'), ('lower', 'open')):
            topic = f'/robot/digital_io/{self.arm}_{button}_button/state'
            self.node.create_subscription(
                DigitalIOState, topic, lambda msg, t=topic, a=action: on_button(t, a, bool(msg.state)), 10
            )

        def on_cycle():
            self._follow_trigger()
            if self._cuff_button is not None:
                action, self._cuff_button = self._cuff_button, None
                self._apply_cuff_button(action)
            if self._cuff_seen and not self._cuff and not target._moving:  # cuff released, arm held
                target.reset()
                if self.raw_output is not None:
                    self.raw_output.stop()
                self.limb.set_joint_positions(self.limb.joint_angles())
            self._cuff_seen = self._cuff
            if self._b_pressed:  # the controller's B button or the home button: home, as the H key
                self._b_pressed = False
                return 'h'
            return None

        self.spin_target(target, on_cycle=on_cycle)

    def _apply_cuff_button(self, action):
        """Open or close the gripper fully from a cuff button, if it is calibrated and ready."""
        if self.gripper is None or not (self.gripper.calibrated() and self.gripper.ready()):
            return
        (self.gripper.close if action == 'close' else self.gripper.open)()
        self.gripper_target = 0.0 if action == 'close' else 100.0

    def _follow_trigger(self, smoothing=0.05, min_step=0.5, min_period=0.02, stale=0.5):
        """Gripper opening from the trigger, continuously: first-order filtered (smoothing, s) and
        sent when it changed by min_step (%) at most every min_period (s); holds when stale."""
        if not self.use_gripper or self._trigger is None or self.gripper is None or self.gripper.type() != 'electric':
            return
        if not self.gripper.calibrated():
            return
        value, received = self._trigger
        now = time.monotonic()
        if now - received > stale:
            return
        goal = 100.0 * (1.0 - float(np.clip(value, 0.0, 1.0)))  # 0 closed .. 100 open
        if self._grip_filtered is None:
            self._grip_filtered, self._grip_sent, self._grip_time = goal, None, now
        else:
            self._grip_filtered += (1.0 - np.exp(-(now - self._grip_time) / smoothing)) * (goal - self._grip_filtered)
            self._grip_time = now
        due = self._grip_sent is None or (
            abs(self._grip_filtered - self._grip_sent[0]) >= min_step and now - self._grip_sent[1] >= min_period
        )
        if due:
            self.gripper.command_position(self._grip_filtered, block=False)
            self._grip_sent = (self._grip_filtered, now)
            self.gripper_target = self._grip_filtered

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
        '--swap-hands',
        choices=['true', 'false'],
        default=None,
        help="override the VR config's swap_hands (true: facing Baxter, each arm driven by the other hand's "
        "controller; false: on Baxter's side, x and y of the axis maps negated)",
    )
    parser.add_argument(
        '--vr-config',
        default=os.path.join(get_package_share_directory('baxter_interface'), 'config', 'vr_axes.yaml'),
        help='VR mapping YAML, a path or a file in the installed config/ (default: vr_axes.yaml; '
        'vr_axes_base.yaml maps translation onto the robot base frame)',
    )
    parser.add_argument('--vr-gain', type=float, default=1.0, help='VR pose servo gain in 1/s')
    parser.add_argument(
        '--vr-trigger-topic', default=None, help='VR trigger topic for the gripper (default: /vive/{arm}/trigger)'
    )
    parser.add_argument(
        '--vr-b-button-topic',
        default=None,
        help='VR B button topic, sends the arm home (default: /vive/{arm}/b_button)',
    )
    parser.add_argument(
        '--no-gripper',
        action='store_true',
        help='leave the gripper to the cuff buttons: the VR trigger does not drive it and going home (B, '
        'H, the home button) does not open it',
    )
    parser.add_argument(
        '--home-button-topic',
        default='/robot/digital_io/right_button_ok/state',
        help="robot button that sends the arm home, as B; every arm's teleop listens, so one press homes "
        "both (default: the right navigator's OK; '' for none)",
    )
    parser.add_argument('--mode', choices=['velocity', 'position'], default='velocity', help='joint control mode')
    parser.add_argument(
        '--output',
        choices=['velocity', 'raw', 'raw_id', 'raw_id_step'],
        default='velocity',
        help="velocity mode's joint command: VELOCITY_MODE (default), 'raw' a jerk-limited setpoint "
        "streamed at 200 Hz in RAW_POSITION_MODE, 'raw_id' raw plus inverse dynamics feedforward "
        '(raw modes bypass collision avoidance); raw_id_step: test-only ablation, the stepped raw_id',
    )
    parser.add_argument(
        '--ik',
        choices=['dls', 'sns'],
        default='dls',
        help="velocity IK: 'dls' damped least squares (default) or 'sns' Rethink's SNS-IK (velocity mode)",
    )
    parser.add_argument(
        '--servo-feedback',
        choices=['measured', 'setpoint'],
        default='measured',
        help='what the pose servo closes on: the measured arm (default) or, with a raw output, its own '
        'setpoint, so the Cartesian loop cannot hunt on joint friction (pick-and-place bag 20261016_212004)',
    )
    parser.add_argument(
        '--joint-accel-scale',
        type=float,
        default=1.0,
        help='scale the joint acceleration limits (4 rad/s^2 shoulder/elbow, 6 wrist); raw outputs overshoot '
        'reversals when the servo asks for more than the limit allows (VR bag 20261016_211106: 39-44%% of cycles)',
    )
    parser.add_argument(
        '--posture-gain',
        type=float,
        default=0.0,
        help='DLS: pull the elbow toward untuck in the null space of the hand task, in 1/s (0 = off); '
        '1/s cost circle tracking in the sim (posture_damping report), 0.5 did not',
    )
    parser.add_argument(
        '--null',
        dest='posture_gain',
        action='store_const',
        const=1.4,
        default=argparse.SUPPRESS,
        help='Enable the DLS null-space untuck pull at 1.4/s (default: off). '
        'If combined with --posture-gain, the last option wins.',
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
        output=args.output,
        ik=args.ik,
        sns_limits=args.sns_limits,
        posture_gain=args.posture_gain,
        joint_accel_scale=args.joint_accel_scale,
        servo_feedback=args.servo_feedback,
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
    teleop.use_gripper = not args.no_gripper
    config = {'args': vars(args), 'tcp_offset': teleop.tcp_offset.tolist(), 'ik': type(teleop.ik).__name__}
    if args.input == 'vr':
        with open(args.vr_config) as f:
            config['vr_config'] = f.read()
    # The controller (libsurvive_ros2 /vive/{hand}/...) driving this arm; swap_hands crosses them.
    hand = args.arm
    swap_hands = None if args.swap_hands is None else args.swap_hands == 'true'
    if args.input == 'vr':
        config['swap_hands'] = (
            yaml.safe_load(config['vr_config']).get('swap_hands', False) if swap_hands is None else swap_hands
        )
        if config['swap_hands']:
            hand = {'left': 'right', 'right': 'left'}[args.arm]
    src = os.path.dirname(os.path.realpath(__file__))
    config['git'] = subprocess.run(
        ['git', '-C', src, 'describe', '--always', '--dirty'], capture_output=True, text=True
    ).stdout.strip()
    teleop.publish_config(config)
    try:
        if args.input == 'vr':
            teleop.spin_vr(
                trigger_topic=args.vr_trigger_topic or f'/vive/{hand}/trigger',
                b_button_topic=args.vr_b_button_topic or f'/vive/{hand}/b_button',
                home_button_topic=args.home_button_topic,
                target=VRTarget(
                    teleop.node,
                    args.vr_topic or f'/vive/{hand}/target',
                    args.vr_config,
                    args.vr_gain,
                    teleop.linear_speed,
                    teleop.angular_speed,
                    swap_hands=swap_hands,
                ),
            )
        else:
            teleop.spin()
    finally:
        if teleop.raw_output is not None:
            teleop.raw_output.close()
        teleop.node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

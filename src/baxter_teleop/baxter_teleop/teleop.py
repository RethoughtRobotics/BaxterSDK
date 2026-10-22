"""Teleop of one Baxter arm: servo its TCP toward a target pose, through differential IK.

Each control cycle a target (inputs/: keyboard, VR, a policy through VRTarget) gives a
base-frame TCP twist toward its goal, the IK (differential_ik.py) turns it into joint velocities,
and an output stage (outputs/) sends them to the arm. `python -m baxter_teleop` runs it (cli.py).
"""

import time

import numpy as np
import rclpy
from rclpy.executors import TimeoutException

import baxter_interface
from baxter_external_devices.getch import getch

from .differential_ik import BoundedLeastSquaresIK, DampedLeastSquaresIK
from .inputs.cuff import Cuff
from .inputs.keyboard import KeyboardTarget, twist_from_key
from .inputs.vr import VRController
from .kinematics import ArmKinematics
from .outputs.raw_position import JointCommandPublisher, start_raw_position_output
from .outputs.replay_stream import ReplayStream
from .outputs.velocity import VelocityOutput
from .safety import CommandValidator
from .settings import DEFAULTS, LIMITS, TCP_OFFSET, joint_names, servo_feedback_for
from .telemetry import Telemetry

ESC = '\x1b'


class Teleop:
    """Servo one Baxter arm's TCP toward a target pose from any input (keyboard, VR, a policy).

    output       'velocity': the IK velocity, ramped, in VELOCITY_MODE. 'raw' / 'raw_id': the IK
                 velocity is the target of a jerk-limited setpoint streamed in RAW_POSITION_MODE,
                 raw_id with inverse dynamics feedforward (outputs/raw_position.py). 'replay': no
                 IK; a plan input's joint trajectory is streamed as replay plays a demo
                 (outputs/replay_stream.py, raw position with inverse dynamics). Raw modes bypass
                 Baxter's collision avoidance.
    Defaults come from config/teleop.yaml (settings.DEFAULTS).

    servo_feedback  what the pose servo and IK close on (None: settings.servo_feedback_for). 'measured': the arm's joint states.
                 'setpoint': the raw output's own setpoint (raw/raw_id only), so the Cartesian loop
                 is purely kinematic and cannot hunt on joint friction; Baxter's joint loop (with
                 inverse dynamics) tracks the setpoint.
    posture_gain (1/s) pulls the spare DOF (elbow swing) toward untuck without moving the hand.
    joint_accel_scale  scales every use of the joint acceleration limits: the IK's braking toward
                 joint limits, the velocity ramp, and the raw setpoint (whose jerk limit follows).
    """

    KEY_SPEEDS = (0.20, 0.8)  # keyboard TCP speeds (m/s, rad/s)
    # VR servo speed caps (m/s, rad/s): above the speeds reached in circle_clockwise_20261001_184937
    # (1.46 m/s, 4.1 rad/s) so they rarely bind; the joint velocity limits still do.
    VR_SPEEDS = (2.0, 5.0)
    # After a position-mode move Baxter ignores velocity commands until ~1.4-1.9 s after it ends
    # (rollouts Clean-SmolVLA*, ACT_Rollout; 20 ms otherwise): wait that out at home.
    HOME_SETTLE = 2.0  # s
    GRIPPER_VELOCITY = 80.0  # % of the electric gripper's max speed (SDK default 50)
    DT = 0.01  # s: the key read timeout each cycle, and the IK's step

    def __init__(
        self,
        arm='right',
        node=None,
        output=DEFAULTS['output'],
        ik='bounded',
        posture_gain=DEFAULTS['posture_gain'],
        joint_accel_scale=1.0,
        servo_feedback=None,
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
        if arm not in ('left', 'right'):
            raise ValueError("arm must be 'left' or 'right'")
        if ik not in ('dls', 'bounded'):
            raise ValueError("ik must be 'dls' or 'bounded'")
        if frame not in ('tool', 'base'):
            raise ValueError("frame must be 'tool' or 'base'")
        if output not in ('velocity', 'raw', 'raw_id', 'replay'):
            raise ValueError("output must be 'velocity', 'raw', 'raw_id' or 'replay'")
        servo_feedback = servo_feedback_for(output, servo_feedback)
        if servo_feedback not in ('measured', 'setpoint'):
            raise ValueError("servo_feedback must be 'measured' or 'setpoint'")
        if servo_feedback == 'setpoint' and output == 'velocity':
            raise ValueError("servo_feedback='setpoint' needs a streamed raw output (--output raw or raw_id)")
        self.arm, self.frame, self.output, self.servo_feedback = arm, frame, output, servo_feedback
        self.dt = self.DT
        self.joint_names = joint_names(arm)
        self.home = baxter_interface.settings.UNTUCK_POSITIONS[arm]

        self.node = node or rclpy.create_node(f'{arm}_arm_ee_teleop')
        # Kept on the global executor, so rclpy.spin_once (e.g. in Limb waits) no longer adds and
        # removes it, which would discard the callbacks still pending; see _spin_pending.
        self.executor = rclpy.get_global_executor()
        self.executor.add_node(self.node)
        self.limb = baxter_interface.Limb(arm, node=self.node)
        self.limb.set_joint_position_speed(1.0)
        self.key_source = getch  # (timeout) -> character or None; key_replay swaps in recorded keys
        self.telemetry = Telemetry(self.node, arm, self.joint_names)
        self.gripper = self._connect_gripper(arm)  # baxter_interface.Gripper, or None without one
        self.gripper_step = float(gripper_step)  # % per [ or ] keypress
        self.drive_gripper = True  # False (--no-gripper): the VR trigger and going home leave it to the cuff buttons
        self._gripper_target = None  # last commanded opening (%), where a key step starts from
        self._vr = None  # inputs.vr.VRController while VR teleop runs

        self.kinematics = ArmKinematics(arm)
        # The TCP offset in the wrist frame: teleop.yaml's tcp.offset, or [0, 0, tcp_offset] if given.
        offset = TCP_OFFSET if tcp_offset is None else [0.0, 0.0, float(tcp_offset)]
        if offset == 'measured':  # the robot's endpoint, which includes the fitted gripper and fingers
            self.kinematics.measure_tcp_offset(self.node, self.limb, self._current_q)
        elif offset != 'urdf':  # 'urdf': ArmKinematics starts at the URDF gripper frame
            offset = np.asarray(offset, dtype=np.float32)
            if offset.shape != (3,):
                raise ValueError(f"tcp offset must be 'measured', 'urdf' or [x, y, z], got {offset.tolist()}")
            self.kinematics.tcp_offset = offset

        self.ik = (BoundedLeastSquaresIK if ik == 'bounded' else DampedLeastSquaresIK)(
            dt=self.dt,
            damping=0.1,
            singular_threshold=0.1,
            max_joint_velocity=6.0,
            posture=self.home,
            posture_gain=posture_gain,
        )
        self.joint_velocity_limits = np.asarray(LIMITS.velocity, dtype=np.float32)
        self.joint_acceleration_limits = float(joint_accel_scale) * np.asarray(LIMITS.acceleration, dtype=np.float32)
        self.joint_position_limits = (
            np.asarray(LIMITS.position_low, dtype=np.float32),
            np.asarray(LIMITS.position_high, dtype=np.float32),
        )
        self.velocity_output = VelocityOutput(self.limb, self.joint_names, self.joint_acceleration_limits, self.dt)
        self.raw_output = None
        if output == 'replay':
            send = JointCommandPublisher(self.node, arm, self.joint_names, self.limb.joint_names())
            self.raw_output = ReplayStream(send, lambda: self.node.get_clock().now().nanoseconds * 1e-9).start()
        elif output != 'velocity':
            self.raw_output = start_raw_position_output(
                self.node,
                arm,
                self.joint_names,
                self.limb.joint_names(),
                self.joint_velocity_limits,
                self.joint_acceleration_limits,
                *self.joint_position_limits,
                inverse_dynamics=output == 'raw_id',
            )

        # A terminal cannot see key release, so a key stays active for a hold window after
        # each character. The first window covers the OS auto-repeat delay (500 ms default).
        self.key_hold_initial = float(key_hold_initial)
        self.key_hold_repeat = float(key_hold_repeat)
        default_linear, default_angular = self.KEY_SPEEDS
        self.linear_speed = default_linear if linear_speed is None else float(linear_speed)
        self.angular_speed = default_angular if angular_speed is None else float(angular_speed)
        self.validator = CommandValidator(
            dt=self.dt,
            max_linear_velocity=max_linear_velocity,
            max_angular_velocity=max_angular_velocity,
            max_linear_delta=max_linear_delta,
            max_angular_delta=max_angular_delta,
            max_delta_jump=max_delta_jump,
        )
        self._cuff = None  # inputs.cuff.Cuff while VR teleop runs
        self._home_requested = False

    def _connect_gripper(self, arm):
        try:
            gripper = baxter_interface.Gripper(arm, node=self.node)
        except Exception as exc:
            self.node.get_logger().warn(f'Gripper control unavailable: {exc}')
            return None
        if gripper.type() != 'electric':
            self.node.get_logger().info(f'Gripper control only supports electric grippers; detected {gripper.type()}.')
            return gripper
        gripper.set_velocity(self.GRIPPER_VELOCITY)
        if gripper.calibrated():
            self._gripper_target = float(gripper.position())
        else:
            self.node.get_logger().info('Gripper is not calibrated; press G before using the gripper keys.')
        return gripper

    @property
    def _gripper_ready(self):
        return self.gripper is not None and self.gripper.type() == 'electric' and self.gripper.calibrated()

    def _command_gripper(self, opening):
        """Command the opening (%: 0 closed, 100 open), if the gripper is ready."""
        if self._gripper_ready:
            self.gripper.command_position(opening, block=False)
            self._gripper_target = float(opening)

    def _step_gripper(self, direction):
        """Open (+1) or close (-1) by one keyboard step from the last command."""
        if not self._gripper_ready:
            self.node.get_logger().warn('The gripper is unavailable or not calibrated (press G).')
            return
        start = self._gripper_target if self._gripper_target is not None else float(self.gripper.position())
        self._command_gripper(float(np.clip(start + direction * self.gripper_step, 0.0, 100.0)))

    def _calibrate_gripper(self):
        if self.gripper is not None and self.gripper.calibrate(block=True):
            self._gripper_target = float(self.gripper.position())

    @property
    def tcp_offset(self):
        return self.kinematics.tcp_offset

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
            self.telemetry.key(key)
        return key

    def _current_q(self):
        """The arm's measured joint angles, in joint order."""
        angles = self.limb.joint_angles()
        return np.array([angles[name] for name in self.joint_names], dtype=np.float32)

    def _servo_q(self, q_meas):
        """Configuration the servo and IK close on: the raw setpoint with servo_feedback='setpoint'
        while it streams, else the measured arm (also for the first cycle of a stroke, which the
        setpoint then starts from)."""
        if self.servo_feedback == 'setpoint' and self.raw_output is not None:
            q_set = self.raw_output.position
            if q_set is not None:
                return q_set.astype(np.float32)
        return q_meas

    def command_twist(self, v_ee):
        """Command joint velocities realizing the base-frame TCP twist v_ee."""
        q_meas = self._current_q()
        q = self._servo_q(q_meas)
        qdot_ik = self.ik.joint_velocity(
            q=q,
            jacobian=self.kinematics.jacobian(q),
            v_ee=self.kinematics.wrist_twist(q, v_ee, 'base'),
            joint_velocity_limits=self.joint_velocity_limits,
            joint_position_limits=self.joint_position_limits,
            joint_acceleration_limits=self.joint_acceleration_limits,
        )
        if self.raw_output is not None:  # the setpoint generator limits acceleration and jerk itself
            self.raw_output.command(qdot_ik, q_meas)
            sent = self.raw_output.velocity
        else:
            sent = self.velocity_output.command(qdot_ik)
        self.telemetry.cycle(v_ee, 'base', q_meas, qdot_ik, sent)

    def _hold(self):
        """Stop the raw setpoint stream, if any, and put the arm on position hold where it is."""
        if self.raw_output is not None:
            self.raw_output.stop()
        self.limb.exit_control_mode()

    def go_home(self):
        """Open the gripper (if enabled) and move to the untuck pose (as tuck_arms -u) with a slow,
        blocking move."""
        self._hold()
        if self.drive_gripper and self._gripper_ready:
            self._command_gripper(100.0)
            if self._vr is not None:  # the trigger filter restarts open
                self._vr.restart_opening(100.0)
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
        self.velocity_output.at_rest()
        if reached:
            self.node.get_logger().info('At the untuck pose.')
        else:
            self.node.get_logger().warning('Did not reach the untuck pose in 15 s: press home again.')

    def _handle_key(self, key):
        """Gripper and home keys, in every input mode; returns whether `key` was one."""
        if key == 'g':
            self._calibrate_gripper()
        elif key == '[':
            self._step_gripper(-1)
        elif key == ']':
            self._step_gripper(+1)
        elif key == 'h':
            self.go_home()
        else:
            return False
        return True

    def spin_target(self, target, on_key=None, on_cycle=None):
        """Servo the TCP to a target pose with velocity commands, for any input: KeyboardTarget,
        VRTarget. target.twist() gives the base-frame twist toward target.goal, or None to hold
        position. The keyboard keeps gripper keys, H (home), SPACE (stop) and ESC; other keys go to
        on_key. on_cycle runs once per control cycle (e.g. the VR gripper) and may return a key to
        act on as if typed (e.g. 'h' for the controller's B button).
        """
        self.limb.set_command_timeout(0.2)
        moving = False
        try:
            while rclpy.ok():
                self._spin_pending()
                injected = on_cycle() if on_cycle is not None else None
                key = self._read_key() or injected
                if key == ESC:
                    break
                if key is not None:
                    if key in ('h', ' '):
                        target.reset()
                        moving = False
                    if not self._handle_key(key) and on_key is not None:
                        on_key(key)

                q_meas = self._current_q()
                tcp_position, tcp_rotation = self.kinematics.tcp(q_meas)
                # /teleop/{arm}/tcp stays the measured TCP; the servo closes on _servo_q().
                self.telemetry.tcp(tcp_position, tcp_rotation)
                if self._cuff is not None and self._cuff.squeezed:
                    # Cuff squeezed: the operator guides the arm in Baxter's zero-g mode. The squeeze
                    # releases the position loop, but the joint controller keeps applying the last
                    # command's torque (in raw_id, the motion's inverse dynamics feedforward) until
                    # the command timeout, which flings the free arm. A position hold at the current
                    # angles replaces that command at once, as at a stroke end; the cuff keeps the
                    # arm in zero-g over it (reports/2026-10-20-cuff-takeover). Then command nothing,
                    # and re-anchor the target so the next stroke or policy step continues from where
                    # the hand leaves the arm instead of pulling it back.
                    if moving:
                        self._hold()
                    target.reset()
                    moving = False
                    continue
                if hasattr(target, 'plan'):  # a timed plan, played as replay plays a demo (no servo, no IK loop)
                    plan = target.plan(q_meas, self.raw_output.position)
                    if target.goal is not None:
                        self.telemetry.goal(target.goal)
                    if plan is not None:
                        self.raw_output.play(*plan)
                        moving = True
                    elif moving:
                        self._hold()
                        moving = False
                    continue
                if self.servo_feedback == 'setpoint':
                    tcp_position, tcp_rotation = self.kinematics.tcp(self._servo_q(q_meas))
                twist = target.twist(tcp_position, tcp_rotation)
                if target.goal is not None:
                    self.telemetry.goal(target.goal)
                if twist is not None:
                    self.command_twist(twist)
                    moving = True
                elif moving:
                    self._hold()  # back to position mode holding the current pose
                    moving = False
        finally:
            if moving:
                self._hold()

    def spin(self):
        """Keyboard teleop: held keys move a goal pose that the TCP is servoed onto."""
        print(f'\n{self.arm.capitalize()} arm keyboard teleop ({type(self.ik).__name__})')
        print(f'speed {self.linear_speed:.2f} m/s, {self.angular_speed:.2f} rad/s')
        print(f'{self.frame} frame, TCP offset in wrist frame {np.round(self.tcp_offset, 3)} m')
        print('W/S X, A/D Y, R/F Z, Q/E yaw, Z/C pitch, X/V roll, SPACE stop, ESC quit')
        print('H go to the untuck pose (home)')
        print('[/ ] gripper delta, G calibrate gripper')
        target = KeyboardTarget(
            self.frame,
            self.linear_speed,
            self.angular_speed,
            hold_initial=self.key_hold_initial,
            hold_repeat=self.key_hold_repeat,
        )

        def on_key(key):
            v = twist_from_key(key, self.linear_speed, self.angular_speed)
            if np.any(v) and self.validator.validate(v * self.dt, logger=self.node.get_logger()):
                target.press(key)

        self.spin_target(target, on_key)

    def spin_vr(self, target, trigger_topic, b_button_topic, home_button=None, takeover_trigger_topic=None):
        """Follow a VRTarget. The controller's analog trigger sets the gripper opening (released
        open, pulled closed) and its B button sends the arm home (as H), as does the robot button
        home_button (a DigitalIO component, e.g. right_button_ok; every arm's teleop listens, so one
        press homes them all). Guided by hand (cuff squeezed), the cuff's buttons drive the gripper.
        With a Takeover target, the operator's trigger (takeover_trigger_topic) drives the gripper
        while they drive the arm, and each cycle says on /teleop/{arm}/takeover whether they do.
        The keyboard keeps gripper/home/ESC."""
        print(f'\n{self.arm.capitalize()} arm VR teleop ({type(self.ik).__name__})')
        print(
            'Touch the trackpad to move, release to stop. A re-anchors. Trigger closes the gripper. '
            'B, H or the right navigator OK home, ESC quit. By hand: squeeze the cuff to move, its '
            'dash/circle buttons close/open'
        )
        controller = self._vr = VRController(self.node, trigger_topic, b_button_topic)
        hand = VRController(self.node, takeover_trigger_topic, None) if takeover_trigger_topic else None
        gripper_source = controller
        if self.gripper is not None and self.gripper.type() == 'electric' and not self.gripper.calibrated():
            self.node.get_logger().info('Calibrating the gripper...')
            self._calibrate_gripper()
        if home_button:  # Baxter's navigator buttons bounce on release: a press counts 0.2 s after one
            self._home_button = baxter_interface.DigitalIO(home_button, self.node, timeout=0, debounce=0.2)
            self._home_button.state_changed.connect(self._on_home_button)
        self._cuff = Cuff(self.node, self.arm)
        cuff_was_squeezed = False

        def on_cycle():
            nonlocal cuff_was_squeezed, gripper_source
            if hand is not None:
                active = target.active
                self.telemetry.takeover(active)
                source = hand if active else controller
                if source is not gripper_source and self._gripper_target is not None:
                    # Filter on from the opening last commanded: the new side's own last command
                    # may equal its trigger now and then never be resent.
                    source.restart_opening(self._gripper_target)
                gripper_source = source
            squeezed = self._cuff.squeezed
            self._drive_gripper(gripper_source, squeezed, released=cuff_was_squeezed and not squeezed)
            # On release of the cuff Baxter's position hold would pull the arm back to where it was
            # held before. Not streaming (no stroke under way): hold where the hand left it.
            if cuff_was_squeezed and not squeezed and not target._moving:
                target.reset()
                if self.raw_output is not None:
                    self.raw_output.stop()
                self.limb.set_joint_positions(self.limb.joint_angles())
            cuff_was_squeezed = squeezed
            home = controller.take_b_press() | self._home_requested
            self._home_requested = False
            return 'h' if home else None

        self.spin_target(target, on_cycle=on_cycle)

    def _drive_gripper(self, source, squeezed, released):
        """One cycle of the gripper: the cuff's buttons open or close it fully (also with --no-gripper),
        and the trigger source drives it unless the cuff is squeezed. Squeezed, the cuff owns it: the
        policy runtime echoes every gripper command back as its trigger, so a trigger filter running
        then would chase its own output and stop a button's motion after one step. On release the
        filter restarts from the cuff's last command."""
        if released and self._gripper_target is not None:
            source.restart_opening(self._gripper_target)
        if self.drive_gripper and self._gripper_ready and not squeezed:
            opening = source.gripper_opening()
            if opening is not None:
                self._command_gripper(opening)
        action = self._cuff.take_button()
        if action is not None and self.gripper is not None and self.gripper.calibrated() and self.gripper.ready():
            (self.gripper.close if action == 'close' else self.gripper.open)()
            self._gripper_target = 0.0 if action == 'close' else 100.0

    def _on_home_button(self, pressed):
        if pressed:
            self._home_requested = True

    def close(self):
        """Stop the raw output's stream thread, if any, and destroy the node."""
        if self.raw_output is not None:
            self.raw_output.close()
        self.node.destroy_node()

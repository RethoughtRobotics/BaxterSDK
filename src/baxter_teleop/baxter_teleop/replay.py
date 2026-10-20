"""Replay a recorded demonstration on the arms, in joint space, by pose or by force.

  python -m baxter_teleop.replay EPISODE [--mode pose|force|both] [--arms left,right] [--out DIR]

EPISODE is a bag folder with /robot/joint_states (a RethoughtAI episode, a record_teleop.sh bag).
Both arms first move slowly to the demo's first pose, then follow the demo at its own timing:

  pose   the demo's joint angles as RAW_POSITION_MODE setpoints at 200 Hz, with their velocities and
         accelerations as Baxter's inverse dynamics feedforward (as raw_id teleop and the SDK's
         position_w_id trajectory playback). Baxter's stiff joint servo tracks the angles.
  force  TORQUE_MODE at 200 Hz: the torques the demo's motion needs, from the arm model (inverse
         dynamics, Pinocchio), plus a computed-torque correction toward the demo angles of bandwidth
         --omega (tau = ID(demo) + M(q) (omega^2 e + 2 omega e_dot)). Baxter adds gravity
         compensation. Lower --omega is more compliant: contact forces come from the scene, not from
         a stiff servo.

Each run logs the measured joints, torques and endpoint wrench; --out writes the comparison with
the demo (diagnostics/replay_comparison.py). Raw position and torque modes bypass Baxter's
collision avoidance: keep the workspace clear.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.executors import TimeoutException
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter

import baxter_interface

from .kinematics import ArmKinematics
from .outputs.raw_position import JointCommandPublisher
from .outputs.setpoint import Setpoint
from .settings import joint_names

RATE = 200.0  # Hz, setpoints per second
DEMO_RATE = 100.0  # Hz, Baxter's /robot/joint_states
TORQUE_LIMITS = np.array([50.0, 50.0, 50.0, 50.0, 15.0, 15.0, 15.0])  # N m, Baxter's joint torque limits
START_TOLERANCE = 0.01  # rad, every joint, before the replay starts


def read_demo(bag):
    """{arm: {'t', 'q', 'effort'}} from /robot/joint_states (by joint name, t from 0 on the robot's clock),
    plus {arm: (t, opening %)} from the gripper states."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag)), rosbag2_py.ConverterOptions('', ''))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    grippers = {f'/robot/end_effector/{arm}_gripper/state': arm for arm in ('left', 'right')}
    reader.set_filter(rosbag2_py.StorageFilter(topics=[t for t in ['/robot/joint_states', *grippers] if t in types]))
    arms = {arm: {'t': [], 'q': [], 'effort': []} for arm in ('left', 'right')}
    gripper = {arm: ([], []) for arm in ('left', 'right')}
    while reader.has_next():
        topic, data, _ = reader.read_next()
        msg = deserialize_message(data, get_message(types[topic]))
        stamp = msg.timestamp if topic in grippers else msg.header.stamp  # both the robot's clock
        stamp = stamp.sec + stamp.nanosec * 1e-9
        if topic in grippers:
            gripper[grippers[topic]][0].append(stamp)
            gripper[grippers[topic]][1].append(msg.position)
            continue
        for arm, d in arms.items():
            names = joint_names(arm)
            if all(n in msg.name for n in names):  # some messages carry only the grippers
                index = [msg.name.index(n) for n in names]
                d['t'].append(stamp)
                d['q'].append([msg.position[i] for i in index])
                d['effort'].append([msg.effort[i] for i in index] if len(msg.effort) == len(msg.name) else [np.nan] * 7)
    t0 = min(d['t'][0] for d in arms.values() if d['t'])
    demo = {arm: {k: np.asarray(v, dtype=float) for k, v in d.items()} for arm, d in arms.items() if d['t']}
    for d in demo.values():
        d['t'] -= t0
    grip = {arm: (np.asarray(t) - t0, np.asarray(p)) for arm, (t, p) in gripper.items() if t}
    return demo, grip


class Trajectory:
    """One arm's demo as a function of time: angles through every recorded sample (cubic spline),
    velocity and acceleration from a Savitzky-Golay fit (90 ms window), for the feedforward."""

    def __init__(self, t, q):
        grid = np.arange(0.0, t[-1], 1.0 / DEMO_RATE)
        uniform = np.stack([np.interp(grid, t, q[:, j]) for j in range(q.shape[1])], axis=1)
        self.duration = float(grid[-1])
        self._q = CubicSpline(t, q, axis=0)
        self._grid = grid
        self._qd = savgol_filter(uniform, 9, 3, deriv=1, delta=1.0 / DEMO_RATE, axis=0)
        self._qdd = savgol_filter(uniform, 9, 3, deriv=2, delta=1.0 / DEMO_RATE, axis=0)

    def __call__(self, t):
        t = float(np.clip(t, 0.0, self.duration))
        qd = np.array([np.interp(t, self._grid, self._qd[:, j]) for j in range(7)])
        qdd = np.array([np.interp(t, self._grid, self._qdd[:, j]) for j in range(7)])
        return np.asarray(self._q(t)), qd, qdd


class Log:
    """What the arms did during a replay, against replay time (s from the replay's start)."""

    def __init__(self, node, arms):
        from baxter_core_msgs.msg import EndpointState
        from sensor_msgs.msg import JointState

        self.start = None
        self.arms = {arm: {'t': [], 'q': [], 'qd': [], 'effort': [], 'wrench_t': [], 'wrench': []} for arm in arms}
        node.create_subscription(JointState, '/robot/joint_states', self._on_joints, 50)
        for arm in arms:
            node.create_subscription(
                EndpointState, f'/robot/limb/{arm}/endpoint_state', lambda m, a=arm: self._on_endpoint(a, m), 50
            )

    def _on_joints(self, msg):
        if self.start is None:
            return
        t = time.monotonic() - self.start
        for arm, d in self.arms.items():
            names = joint_names(arm)
            if all(n in msg.name for n in names):
                index = [msg.name.index(n) for n in names]
                d['t'].append(t)
                d['q'].append([msg.position[i] for i in index])
                d['qd'].append([msg.velocity[i] for i in index] if len(msg.velocity) == len(msg.name) else [np.nan] * 7)
                d['effort'].append([msg.effort[i] for i in index] if len(msg.effort) == len(msg.name) else [np.nan] * 7)

    def _on_endpoint(self, arm, msg):
        if self.start is None:
            return
        f, m = msg.wrench.force, msg.wrench.torque
        self.arms[arm]['wrench_t'].append(time.monotonic() - self.start)
        self.arms[arm]['wrench'].append([f.x, f.y, f.z, m.x, m.y, m.z])

    def arrays(self):
        return {arm: {k: np.asarray(v, dtype=float) for k, v in d.items()} for arm, d in self.arms.items()}


class Replayer:
    """Plays a demo on the arms by pose or by force (module docstring)."""

    def __init__(self, node, demo, gripper, arms, omega=10.0, speed=1.0, lead=0.0, drive_gripper=True, until=None):
        self.node, self.arms, self.omega, self.speed, self.lead = node, arms, float(omega), float(speed), float(lead)
        self.executor = rclpy.get_global_executor()
        self.executor.add_node(node)
        self.limbs = {arm: baxter_interface.Limb(arm, node=node) for arm in arms}
        self.trajectories = {arm: Trajectory(demo[arm]['t'], demo[arm]['q']) for arm in arms}
        self.duration = min(tr.duration for tr in self.trajectories.values())
        if until is not None:
            self.duration = min(self.duration, float(until))
        self.kinematics = {arm: ArmKinematics(arm) for arm in arms}
        self.names = {arm: joint_names(arm) for arm in arms}
        self.raw = {
            arm: JointCommandPublisher(node, arm, self.names[arm], self.limbs[arm].joint_names()) for arm in arms
        }
        self.gripper = {}
        if drive_gripper:
            for arm in arms:  # only a gripper the demo moved
                if arm in gripper and np.ptp(gripper[arm][1]) > 1.0:
                    self.gripper[arm] = (baxter_interface.Gripper(arm, node=node), *gripper[arm])
        self.log = Log(node, arms)

    def _spin(self):
        """Run every callback that is ready now (as the teleop's control loop does)."""
        try:
            while True:
                handler, _, _ = self.executor.wait_for_ready_callbacks(timeout_sec=0.0)
                handler()
        except TimeoutException:
            pass

    def _angles(self, arm):
        a = self.limbs[arm].joint_angles()
        return np.array([a[n] for n in self.names[arm]])

    def _velocities(self, arm):
        v = self.limbs[arm].joint_velocities()
        return np.array([v[n] for n in self.names[arm]])

    def move_to_start(self, timeout=30.0):
        """Both arms to the demo's first pose, slowly and together; True when within START_TOLERANCE."""
        start = {arm: self.trajectories[arm](0.0)[0] for arm in self.arms}
        for limb in self.limbs.values():
            limb.set_joint_position_speed(0.3)
        deadline = time.monotonic() + timeout
        try:
            while time.monotonic() < deadline:
                self._spin()
                errors = [np.max(np.abs(self._angles(arm) - start[arm])) for arm in self.arms]
                if max(errors) < START_TOLERANCE:
                    break
                for arm in self.arms:
                    self.limbs[arm].set_joint_positions(dict(zip(self.names[arm], start[arm].tolist())))
                time.sleep(0.01)
            else:
                return False
            end = time.monotonic() + 1.0  # settle
            while time.monotonic() < end:
                self._spin()
                for arm in self.arms:
                    self.limbs[arm].set_joint_positions(dict(zip(self.names[arm], start[arm].tolist())))
                time.sleep(0.01)
            return True
        finally:
            for limb in self.limbs.values():
                limb.set_joint_position_speed(1.0)

    def _command_force(self, arm, q_d, qd_d, qdd_d):
        q, qd = self._angles(arm), self._velocities(arm)
        k = self.kinematics[arm]
        e, e_dot = q_d - q, qd_d - qd
        tau = k.inverse_dynamics(q_d, qd_d, qdd_d) + k.mass_matrix(q) @ (self.omega**2 * e + 2.0 * self.omega * e_dot)
        tau = np.clip(tau, -TORQUE_LIMITS, TORQUE_LIMITS)
        self.limbs[arm].set_joint_torques(dict(zip(self.names[arm], tau.tolist())))

    def run(self, mode):
        """Play the demo once by `mode` ('pose' or 'force') from its first pose; returns the log arrays."""
        if not self.move_to_start():
            raise RuntimeError('the arms did not reach the demo start pose in 30 s')
        for limb in self.limbs.values():
            limb.set_command_timeout(0.2)
        self.log.start = time.monotonic()
        period, next_tick, sent = 1.0 / RATE, time.monotonic(), {arm: None for arm in self.gripper}
        try:
            while True:
                self._spin()
                t = (time.monotonic() - self.log.start) * self.speed
                if t > self.duration:
                    break
                for arm in self.arms:
                    q_d, qd_d, qdd_d = self.trajectories[arm](t + self.lead)
                    if mode == 'pose':
                        self.raw[arm](Setpoint(q_d, qd_d * self.speed, qdd_d * self.speed**2))
                    else:
                        self._command_force(arm, q_d, qd_d * self.speed, qdd_d * self.speed**2)
                for arm, (gripper, gt, gp) in self.gripper.items():
                    opening = float(np.interp(t, gt, gp))
                    if sent[arm] is None or abs(opening - sent[arm]) >= 1.0:
                        gripper.command_position(opening, block=False)
                        sent[arm] = opening
                next_tick += period
                delay = next_tick - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                elif delay < -period:
                    next_tick = time.monotonic()
        finally:
            for arm in self.arms:  # hold where the arm is
                self.limbs[arm].set_joint_positions(dict(zip(self.names[arm], self._angles(arm).tolist())))
            self.log.start = None
        result = self.log.arrays()
        self.log = Log(self.node, self.arms)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m baxter_teleop.replay', description=__doc__.split('\n\n')[0])
    parser.add_argument('episode', help='bag folder with /robot/joint_states')
    parser.add_argument(
        '--mode', choices=['pose', 'force', 'both'], default='both', help='how to replay (default: both)'
    )
    parser.add_argument('--arms', default=None, help='left, right or left,right (default: every arm the demo moved)')
    parser.add_argument('--omega', type=float, default=10.0, help='force mode tracking bandwidth, rad/s (default: 10)')
    parser.add_argument('--speed', type=float, default=1.0, help='playback speed factor (default: 1, the demo timing)')
    parser.add_argument(
        '--lead', type=float, default=0.0, help='setpoints ahead of the demo by this many s (default: 0)'
    )
    parser.add_argument('--no-gripper', action='store_true', help='leave the grippers alone')
    parser.add_argument('--until', type=float, default=None, help="replay only the demo's first SECONDS")
    parser.add_argument('--out', default=None, help='write the comparison with the demo into this folder')
    args, ros_args = parser.parse_known_args(sys.argv[1:] if argv is None else argv)

    demo, gripper = read_demo(args.episode)
    moved = [arm for arm, d in demo.items() if np.max(np.ptp(d['q'], axis=0)) > np.radians(5)]
    arms = args.arms.split(',') if args.arms else moved
    missing = [arm for arm in arms if arm not in demo]
    if missing or not arms:
        parser.error(f'the demo has no joint states for {missing or "any arm"}')
    modes = ['pose', 'force'] if args.mode == 'both' else [args.mode]
    duration = min(d['t'][-1] for arm, d in demo.items() if arm in arms)
    print(
        f'Demo {Path(args.episode).name}: {duration:.1f} s, arms {", ".join(arms)}; replaying by {" then ".join(modes)}'
    )

    rclpy.init(args=[sys.argv[0], *ros_args])
    node = rclpy.create_node('demo_replay')
    runs = {}
    try:
        replayer = Replayer(
            node, demo, gripper, arms, args.omega, args.speed, args.lead, not args.no_gripper, args.until
        )
        for mode in modes:
            print(f'== {mode}: moving to the demo start pose, then {replayer.duration / args.speed:.1f} s of replay')
            runs[mode] = replayer.run(mode)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    if args.out:
        from .diagnostics.replay_comparison import write_report

        settings = {
            'episode': str(Path(args.episode).resolve()),
            **{k: v for k, v in vars(args).items() if k != 'episode'},
        }
        print(write_report(args.out, demo, runs, arms, settings))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""End-to-end regression test of the teleop on the MuJoCo sim, against a stored baseline.

  python -m baxter_teleop.diagnostics.sim_regression [--only circle,vr,keys] [--headless]
                                                      [--write-baseline] [--domain 43]

Each scenario starts a fresh sim (the sim's tuck_arms stops short of untuck by up to 0.12 rad,
depending on where the arm was, so runs that share a sim start from different poses), untucks the
right arm, and drives the real entry point, then measures what the arm did:

  circle  the scripted trajectory (one 15 cm circle, the teleop's default output and feedback):
          TCP tracking error to the servo goal (rms, max) and the elbow roll (e0) range
  vr      `python -m baxter_teleop --input vr` with its defaults, driven by a fake libsurvive_ros2
          controller (this module): a 10 cm stroke, a release, the trigger, then B. Measures the
          TCP stroke, the drift while released, the gripper opening the trigger set (0.3 pulled:
          70 %), and the distance from the start pose after B (home)
  keys    `python -m baxter_teleop` (keyboard) with keys typed into its terminal: W for 1 s, then
          A for 1 s. Measures the TCP's x and y travel

The teleop runs in a pseudo-terminal, so its key reads (getch) and ESC to quit are the real ones.
Results are compared to src/baxter_teleop/test/sim_baseline.yaml (value and tolerance per metric);
--write-baseline stores this run's values instead. Exit status 1 if any metric is out of tolerance.

The MuJoCo viewer opens when a display is available, so the runs can be watched; --headless
hides it. Runs on ROS_DOMAIN_ID 43 by default, apart from a real robot on 0.
"""

import argparse
import os
import pty
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import yaml

ARM = 'right'
HERE = Path(__file__).resolve()
SOURCE_BASELINE = 'src/baxter_teleop/test/sim_baseline.yaml'


def workspace():
    """The workspace root: BAXTER_SDK, else found from the sourced install (scripts/env.sh)."""
    if os.environ.get('BAXTER_SDK'):
        return Path(os.environ['BAXTER_SDK'])
    for parent in HERE.parents:
        if (parent / 'src' / 'baxter_mujoco').is_dir():
            return parent
    raise RuntimeError('workspace not found: set BAXTER_SDK to the baxter_sdk checkout')


class Probe:
    """Records the arm's measured TCP, servo goals and joint states, and plays a fake VR controller."""

    def __init__(self):
        import rclpy
        from baxter_core_msgs.msg import EndEffectorState
        from geometry_msgs.msg import PoseStamped
        from rclpy.executors import SingleThreadedExecutor
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Empty, Float32

        self._PoseStamped, self._Empty, self._Float32 = PoseStamped, Empty, Float32
        rclpy.init()
        self.node = rclpy.create_node('sim_regression')
        self.tcp, self.goals, self.e0 = [], [], []
        self.gripper = None  # the gripper's measured opening (%)
        self.node.create_subscription(
            EndEffectorState,
            f'/robot/end_effector/{ARM}_gripper/state',
            lambda m: setattr(self, 'gripper', m.position),
            10,
        )
        self.node.create_subscription(PoseStamped, f'/teleop/{ARM}/tcp', self._on_tcp, 10)
        self.node.create_subscription(PoseStamped, f'/teleop/{ARM}/goal', self._on_goal, 10)
        self.node.create_subscription(JointState, '/robot/joint_states', self._on_joints, 10)
        # vr_axes_raw_id.yaml has swap_hands: true (operator facing Baxter): the right arm follows the left hand
        hand = 'left'
        self.target = self.node.create_publisher(PoseStamped, f'/vive/{hand}/target', 10)
        self.trigger = self.node.create_publisher(Float32, f'/vive/{hand}/trigger', 10)
        self.b_button = self.node.create_publisher(Empty, f'/vive/{hand}/b_button', 10)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.node)
        threading.Thread(target=self._executor.spin, daemon=True).start()

    def _on_tcp(self, msg):
        p = msg.pose.position
        self.tcp.append((time.monotonic(), np.array([p.x, p.y, p.z])))

    def _on_goal(self, msg):
        p = msg.pose.position
        self.goals.append((time.monotonic(), np.array([p.x, p.y, p.z])))

    def _on_joints(self, msg):
        if f'{ARM}_e0' in msg.name:
            self.e0.append(msg.position[msg.name.index(f'{ARM}_e0')])

    def clear(self):
        self.tcp, self.goals, self.e0 = [], [], []

    def latest_tcp(self):
        return self.tcp[-1][1] if self.tcp else None

    def wait_for_tcp(self, timeout):
        """Wait until the teleop publishes its TCP (it is up), return it."""
        deadline = time.monotonic() + timeout
        while not self.tcp:
            if time.monotonic() > deadline:
                raise RuntimeError(f'no /teleop/{ARM}/tcp in {timeout:.0f} s: the teleop did not start')
            time.sleep(0.1)
        return self.latest_tcp()

    def play_vr(self, seconds, x_of_t):
        """Publish the fake controller's target (anchor frame) and a 0.3 trigger at 200 Hz."""
        start = time.monotonic()
        while (t := time.monotonic() - start) < seconds:
            msg = self._PoseStamped()
            msg.header.frame_id = 'anchor'
            msg.pose.position.x, msg.pose.position.z = float(x_of_t(t)), 1.0
            msg.pose.orientation.w = 1.0
            self.target.publish(msg)
            self.trigger.publish(self._Float32(data=0.3))
            time.sleep(0.005)

    def press_b(self):
        self.b_button.publish(self._Empty())

    def close(self):
        import rclpy

        self._executor.shutdown()
        self.node.destroy_node()
        rclpy.shutdown()


class Sim:
    """A fresh MuJoCo sim on the current ROS domain, untucked, for one scenario."""

    def __init__(self, ws, viewer):
        args = ['./launch_sim.sh', '--ros', '--empty'] + ([] if viewer else ['--no-viewer'])
        self.log = open(Path(os.environ.get('XDG_RUNTIME_DIR', '/tmp')) / 'sim_regression_sim.log', 'w')
        self.proc = subprocess.Popen(
            args, cwd=ws / 'src' / 'baxter_mujoco', stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True
        )

    def wait_ready(self, probe, timeout=60):
        probe.clear()
        deadline = time.monotonic() + timeout
        while not probe.e0:
            if self.proc.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError(f'the sim did not come up (log: {self.log.name})')
            time.sleep(0.2)
        subprocess.run(['ros2', 'run', 'baxter_tools', 'tuck_arms', '-u'], capture_output=True, timeout=60)

    def stop(self):
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGINT)
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
        self.log.close()


class Terminal:
    """A command in a pseudo-terminal, so its getch key reads work and keys can be typed into it."""

    def __init__(self, command, log_name):
        self._master, slave = pty.openpty()
        self.log = Path(os.environ.get('XDG_RUNTIME_DIR', '/tmp')) / log_name
        self._out = open(self.log, 'wb')
        self.proc = subprocess.Popen(command, stdin=slave, stdout=slave, stderr=slave, start_new_session=True)
        os.close(slave)
        threading.Thread(target=self._drain, daemon=True).start()

    def _drain(self):
        while True:
            try:
                data = os.read(self._master, 4096)
            except OSError:
                return
            if not data:
                return
            self._out.write(data)
            self._out.flush()

    def type(self, text):
        os.write(self._master, text.encode())

    def hold_key(self, key, seconds, rate=20.0):
        """Type key repeatedly, as a held key's auto-repeat does, slower than the control loop (~75 Hz)
        reads keys, so none queue up and stretch the press."""
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.type(key)
            time.sleep(1.0 / rate)

    def quit(self, timeout=20):
        """ESC, as the operator quits; then signals if it does not exit."""
        if self.proc.poll() is None:
            self.type('\x1b')
            try:
                self.proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGINT)
                self.proc.wait(timeout=10)
        self._out.close()


def teleop_command(*args):
    return [sys.executable, '-m', 'baxter_teleop', '--arm', ARM, *args]


def scenario_circle(probe):
    """The scripted circle: TCP error to the servo goal while it runs, and the e0 range."""
    probe.clear()
    run = subprocess.run(
        [sys.executable, '-m', 'baxter_teleop.diagnostics.scripted_trajectory', '--arm', ARM, '--protocol', 'circle',
         '--loops', '1', '--gain', '3', '--no-feedforward'],
        capture_output=True, text=True, timeout=180,
    )  # fmt: skip
    if run.returncode != 0:
        raise RuntimeError(f'scripted_trajectory failed ({run.returncode}): {run.stderr[-500:]}')
    tcp_t = np.array([t for t, _ in probe.tcp])
    tcp_p = np.array([p for _, p in probe.tcp])
    errors = []
    for t, goal in probe.goals:  # each goal against the TCP measured right after it
        i = np.searchsorted(tcp_t, t)
        if i < len(tcp_p):
            errors.append(np.linalg.norm(goal - tcp_p[i]))
    errors = np.asarray(errors)
    return {
        'goal_error_rms_mm': 1000.0 * float(np.sqrt(np.mean(errors**2))),
        'goal_error_max_mm': 1000.0 * float(errors.max()),
        'e0_range_deg': float(np.degrees(np.ptp(probe.e0))),
    }


def scenario_vr(probe):
    """VR teleop with its defaults: a 10 cm stroke, a release, then B (home)."""
    probe.clear()
    teleop = Terminal(teleop_command('--input', 'vr'), 'sim_regression_vr.log')
    try:
        start = probe.wait_for_tcp(90)
        probe.play_vr(1.0, lambda t: 1e-6 * (t * 1000 % 2))  # changing pose: clutched, anchored here
        anchored = probe.latest_tcp()
        probe.play_vr(2.0, lambda t: 0.1 * t / 2.0)  # 10 cm along the anchor x
        stroked = probe.latest_tcp()
        probe.play_vr(2.0, lambda t: 0.1)  # frozen pose: released, the arm holds
        held = probe.latest_tcp()
        goals = len(probe.goals)
        opening = probe.gripper  # trigger 0.3 the whole time: 70 % open
        probe.press_b()
        time.sleep(15)  # the home move (0.3 speed) and its settle
        home = probe.latest_tcp()
    finally:
        teleop.quit()
    return {
        'stroke_x_mm': 1000.0 * float(stroked[0] - anchored[0]),
        'hold_drift_mm': 1000.0 * float(np.linalg.norm(held - stroked)),
        'goals_during_stroke': float(goals),
        'home_from_start_mm': 1000.0 * float(np.linalg.norm(home - start)),
        'gripper_opening_pct': float('nan') if opening is None else float(opening),
    }


def scenario_keys(probe):
    """Keyboard teleop with typed keys: W (forward, +x) for 1 s, then A (left, +y) for 1 s."""
    probe.clear()
    teleop = Terminal(teleop_command(), 'sim_regression_keys.log')
    try:
        probe.wait_for_tcp(90)
        time.sleep(3.0)  # past startup (the TCP measurement, the first cycles), so keys are read as they arrive
        start = probe.latest_tcp()
        teleop.hold_key('w', 1.0)
        time.sleep(2.5)  # released: the TCP settles onto the goal
        forward = probe.latest_tcp()
        teleop.hold_key('a', 1.0)
        time.sleep(2.5)
        left = probe.latest_tcp()
    finally:
        teleop.quit()
    return {
        'w_travel_x_mm': 1000.0 * float(forward[0] - start[0]),
        'a_travel_y_mm': 1000.0 * float(left[1] - forward[1]),
        'w_cross_axis_mm': 1000.0 * float(np.linalg.norm((forward - start)[1:])),
    }


SCENARIOS = {'circle': scenario_circle, 'vr': scenario_vr, 'keys': scenario_keys}


def compare(results, baseline):
    """Rows (scenario, metric, value, expected, tolerance, ok) and whether all passed."""
    rows, ok_all = [], True
    for scenario, metrics in results.items():
        for metric, value in metrics.items():
            ref = baseline.get(scenario, {}).get(metric)
            if ref is None:
                rows.append((scenario, metric, value, None, None, None))
                continue
            ok = abs(value - ref['value']) <= ref['tolerance']
            ok_all &= ok
            rows.append((scenario, metric, value, ref['value'], ref['tolerance'], ok))
    return rows, ok_all


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--only', default=','.join(SCENARIOS), help='comma-separated scenarios (default: all)')
    parser.add_argument('--headless', action='store_true', help='no MuJoCo viewer')
    parser.add_argument('--write-baseline', action='store_true', help="store this run's values as the baseline")
    parser.add_argument('--domain', default='43', help='ROS_DOMAIN_ID for the sim and teleop (default: 43)')
    args = parser.parse_args()
    names = [n for n in args.only.split(',') if n]
    unknown = set(names) - set(SCENARIOS)
    if unknown:
        parser.error(f'unknown scenarios: {sorted(unknown)}')
    os.environ['ROS_DOMAIN_ID'] = args.domain
    ws = workspace()
    baseline_path = ws / SOURCE_BASELINE
    viewer = not args.headless and bool(os.environ.get('DISPLAY'))

    # One Zenoh router for the whole run: each sim would otherwise start its own and stop it on exit,
    # cutting the probe off between scenarios.
    router = None
    if subprocess.run(['pgrep', '-f', 'rmw_zenohd'], capture_output=True).returncode != 0:
        router = subprocess.Popen(
            ['ros2', 'run', 'rmw_zenoh_cpp', 'rmw_zenohd'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
        )  # fmt: skip
    probe = Probe()
    time.sleep(3)
    if probe.e0:
        probe.close()
        if router is not None:
            os.killpg(router.pid, signal.SIGINT)
        sys.exit(f'Something already publishes /robot/joint_states on ROS_DOMAIN_ID={args.domain}: stop it first.')
    results = {}
    try:
        for name in names:
            print(f'== {name}: starting a fresh sim{" (viewer)" if viewer else ""}', flush=True)
            sim = Sim(ws, viewer)
            try:
                sim.wait_ready(probe)
                results[name] = SCENARIOS[name](probe)
            finally:
                sim.stop()
            time.sleep(2)  # let the router forget the old sim before the next one
    finally:
        probe.close()
        if router is not None:
            os.killpg(router.pid, signal.SIGINT)

    if args.write_baseline:
        for scenario, metrics in results.items():
            for metric, value in metrics.items():
                print(f'{scenario:8} {metric:22} {value:9.2f}')
        old = yaml.safe_load(baseline_path.read_text()) if baseline_path.exists() else {}
        for scenario, metrics in results.items():
            for metric, value in metrics.items():
                tolerance = old.get(scenario, {}).get(metric, {}).get('tolerance', max(2.0, 0.1 * abs(value)))
                old.setdefault(scenario, {})[metric] = {'value': round(value, 2), 'tolerance': tolerance}
        header = (
            ''.join(line for line in baseline_path.read_text().splitlines(keepends=True) if line.startswith('#'))
            if baseline_path.exists()
            else '# Expected sim_regression results: value and allowed deviation per metric (same units).\n'
            '# Written by python -m baxter_teleop.diagnostics.sim_regression --write-baseline; tolerances\n'
            '# are kept across rewrites, so tighten them here by hand.\n'
        )
        baseline_path.write_text(header + yaml.safe_dump(old, sort_keys=True))
        print(f'Baseline written: {baseline_path}')
        return
    baseline = yaml.safe_load(baseline_path.read_text()) if baseline_path.exists() else {}
    rows, ok_all = compare(results, baseline)
    print(f'\n{"scenario":8} {"metric":22} {"value":>9} {"expected":>9} {"tol":>6}  result')
    for scenario, metric, value, expected, tolerance, ok in rows:
        if expected is None:
            print(f'{scenario:8} {metric:22} {value:9.2f} {"-":>9} {"-":>6}  no baseline')
        else:
            print(f'{scenario:8} {metric:22} {value:9.2f} {expected:9.2f} {tolerance:6.2f}  {"PASS" if ok else "FAIL"}')
    sys.exit(0 if ok_all else 1)


if __name__ == '__main__':
    main()

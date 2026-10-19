#!/usr/bin/env python3

"""Drive the teleop through a scripted TCP trajectory, for comparing controllers on equal input.

The trajectory is a target like VRTarget and KeyboardTarget: spin_target servos the TCP onto its goal
(servo_twist, with the path velocity as feedforward), so this exercises the same control code as
VR and the keyboard. It starts from the current TCP (run tuck_arms -u first) and holds the start
orientation. Phases (offsets in the robot base frame, 0.1 m/s, 1 s pauses):

  square x3     20 cm square in x-y, three times: does the configuration return with the hand?
  down and up   20 cm down and back
  elbow limit   35 cm forward and 30 cm out, 9 cm past where the e1 (elbow) limit stops the arm, and back
  circle x2     10 cm radius circle in x-y, twice

--protocol circle runs only planar circles instead (x-y, centred --radius to the arm's outer side so
they start at the start TCP), --loops times at --circle-speed: a repeatable stimulus for comparing
joint output stages (--output) on the robot, e.g. via run_circle_test.sh.

  python -m baxter_teleop.diagnostics.scripted_trajectory --ik dls|bounded [--arm left]
  python -m baxter_teleop.diagnostics.scripted_trajectory --arm right --protocol circle --output raw_id \
      --gain 3 --no-feedforward
"""

import argparse
import sys
import time

import numpy as np
import rclpy
from scipy.spatial.transform import Rotation

from ..servo import servo_twist
from ..teleop import Teleop
from .iso9283 import iso_protocol

DT = 0.01
SPEED = 0.1  # m/s
PAUSE = 1.0  # s
SQUARE = [(0.2, 0, 0), (0.2, 0.2, 0), (0, 0.2, 0), (0, 0, 0)]


def circle_protocol(radius, speed, loops, side):
    """Planar x-y circles starting and ending at the start TCP, centred `radius` to the arm's outer
    side (side = +1 left arm, -1 right arm, robot y), counterclockwise seen from above. Dense offsets
    (n, 3) at DT, 1 s pauses before and after, and phases [(name, t_start, t_end)]."""
    pause = np.zeros((int(PAUSE / DT), 3))
    n = int(round(loops * 2 * np.pi * radius / (speed * DT)))
    angle = -side * np.pi / 2 + np.linspace(0, loops * 2 * np.pi, n + 1)[1:]
    circle = np.c_[radius * np.cos(angle), side * radius + radius * np.sin(angle), np.zeros(n)]
    t0 = len(pause) * DT
    return np.vstack([pause, circle, pause]), [(f'circle x{loops} r{radius:g} at {speed:g} m/s', t0, t0 + n * DT)]


def protocol():
    """Dense path offsets (n, 3) at DT, and phases [(name, t_start, t_end)]."""
    points, phases = [np.zeros(3)], []

    def line_to(target):
        start, target = points[-1], np.asarray(target, float)
        n = max(1, int(np.ceil(np.linalg.norm(target - start) / (SPEED * DT))))
        points.extend(start + (target - start) * k / n for k in range(1, n + 1))

    def pause():
        points.extend([points[-1]] * int(PAUSE / DT))

    def phase(name, build):
        start = len(points) * DT
        build()
        phases.append((name, start, len(points) * DT))
        pause()

    pause()
    phase('square x3', lambda: [line_to(c) for _ in range(3) for c in SQUARE])
    phase('down and up', lambda: [line_to(c) for c in ((0, 0, -0.2), (0, 0, 0))])
    # Diagonal, away from the body: straight forward the upper elbow meets the torso before e1 does.
    phase('elbow limit', lambda: [line_to(c) for c in ((0.35, 0.3, 0), (0, 0, 0))])

    def circle():
        n = int(2 * 2 * np.pi * 0.1 / (SPEED * DT))
        angle = np.pi + np.linspace(0, 4 * np.pi, n + 1)[1:]
        points.extend(np.c_[0.1 + 0.1 * np.cos(angle), 0.1 * np.sin(angle), np.zeros(n)])

    phase('circle x2', circle)
    return np.asarray(points), phases


class SafetyMonitor:
    """Ends a run that has gone wrong, for unattended benchmark sweeps: the TCP more than
    max_error from the goal for error_time, or the endpoint force above max_force for force_time
    (contact). Measured from /robot/limb/{arm}/endpoint_state (Baxter's own TCP and wrench estimate),
    independent of what the servo closes on. update() returns the reason once tripped."""

    def __init__(self, max_error=0.10, error_time=0.3, max_force=60.0, force_time=0.1):
        self.max_error, self.error_time = max_error, error_time
        self.max_force, self.force_time = max_force, force_time
        self._error_since = self._force_since = None
        self.reason = None

    def update(self, now, goal_position, tcp_position, force):
        """Feed one sample; returns the trip reason (str) or None."""
        if self.reason is None:
            error = np.inf if goal_position is None else float(np.linalg.norm(np.subtract(goal_position, tcp_position)))
            if goal_position is not None and error > self.max_error:
                self._error_since = now if self._error_since is None else self._error_since
                if now - self._error_since >= self.error_time:
                    self.reason = f'tracking error {error * 1e3:.0f} mm > {self.max_error * 1e3:.0f} mm'
            else:
                self._error_since = None
            force = float(np.linalg.norm(force))
            if force > self.max_force:
                self._force_since = now if self._force_since is None else self._force_since
                if now - self._force_since >= self.force_time:
                    self.reason = f'endpoint force {force:.0f} N > {self.max_force:.0f} N (contact?)'
            else:
                self._force_since = None
        return self.reason


class TrajectoryTarget:
    """Goal = start TCP + path offset at the elapsed time, orientation held; done after the end."""

    def __init__(self, offsets, gain=5.0, settle=2.0, feedforward=True, max_linear=4 * SPEED):
        """feedforward: add the path velocity to the servo (tracks without standing error); without
        it the servo is P-only like VR teleop, so the TCP trails the goal by ~1/gain."""
        self.offsets, self.gain, self.settle = offsets, gain, settle
        self.feedforward, self.max_linear = feedforward, max_linear
        self.start = self.origin = self.goal = None
        self.done = False

    def reset(self):
        pass

    def twist(self, tcp_position, tcp_rotation):
        now = time.monotonic()
        tcp_rotation = Rotation.from_matrix(tcp_rotation)
        if self.start is None:
            self.start, self.origin = now, (np.asarray(tcp_position, float).copy(), tcp_rotation)
        i = int((now - self.start) / DT)
        if i >= len(self.offsets) + self.settle / DT:
            self.done, self.goal = True, None
            return None
        i = min(i, len(self.offsets) - 1)
        nxt = min(i + 1, len(self.offsets) - 1)
        self.goal = (self.origin[0] + self.offsets[i], self.origin[1])
        feedforward = np.r_[(self.offsets[nxt] - self.offsets[i]) / DT, np.zeros(3)] if self.feedforward else None
        return servo_twist(
            self.goal, tcp_position, tcp_rotation, self.gain, self.max_linear, 1.0, feedforward=feedforward
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ik', choices=['dls', 'bounded'], default='bounded')
    parser.add_argument('--arm', choices=['left', 'right'], default='left')
    parser.add_argument('--posture-gain', type=float, default=0.0, help='DLS null-space pull toward untuck, 1/s')
    parser.add_argument('--output', choices=['velocity', 'raw', 'raw_id'], default='velocity', help='as the teleop')
    parser.add_argument('--servo-feedback', choices=['measured', 'setpoint'], default='measured', help='as the teleop')
    parser.add_argument('--joint-accel-scale', type=float, default=1.0, help='as the teleop')
    parser.add_argument('--gain', type=float, default=5.0, help='servo gain, 1/s (VR: --vr-gain x linear_scale)')
    parser.add_argument('--no-feedforward', dest='feedforward', action='store_false', help='P-only servo, as VR')
    parser.add_argument(
        '--feedforward',
        dest='feedforward',
        action='store_true',
        default=argparse.SUPPRESS,
        help='path velocity feedforward (the default); after --no-feedforward, the last one wins',
    )
    parser.add_argument('--protocol', choices=['full', 'circle', 'iso'], default='full')
    parser.add_argument('--speed', type=float, default=0.15, help='iso protocol: path speed, m/s')
    parser.add_argument('--accel', type=float, default=0.5, help='iso protocol: path acceleration, m/s^2')
    parser.add_argument(
        '--safety',
        action='store_true',
        help='abort on tracking error > 100 mm beyond the expected P-servo lag, or endpoint force > 60 N (exit 3)',
    )
    parser.add_argument('--radius', type=float, default=0.15, help='circle protocol: radius, m')
    parser.add_argument('--circle-speed', type=float, default=0.15, help='circle protocol: TCP speed, m/s')
    parser.add_argument('--loops', type=int, default=3, help='circle protocol: number of circles')
    args, ros_args = parser.parse_known_args()
    rclpy.init(args=['scripted_trajectory', *ros_args])
    teleop = Teleop(
        arm=args.arm,
        output=args.output,
        ik=args.ik,
        frame='base',
        posture_gain=args.posture_gain,
        servo_feedback=args.servo_feedback,
        joint_accel_scale=args.joint_accel_scale,
    )
    events = []
    if args.protocol == 'iso':
        offsets, phases, events = iso_protocol(args.speed, args.accel, side=1 if args.arm == 'left' else -1, dt=DT)
        speed = args.speed
    elif args.protocol == 'circle':
        side = 1 if args.arm == 'left' else -1
        offsets, phases = circle_protocol(args.radius, args.circle_speed, args.loops, side)
        speed = args.circle_speed
    else:
        offsets, phases = protocol()
        speed = SPEED
    teleop.telemetry.config({
        'trajectory_test': {**vars(args), 'speed': speed, 'phases': phases, 'events': events},
        'args': {
            'arm': args.arm,
            'input': 'trajectory',
            'ik': args.ik,
            'posture_gain': args.posture_gain,
            'output': args.output,
            'servo_gain': args.gain,
            'feedforward': args.feedforward,
            'servo_feedback': args.servo_feedback,
            'joint_accel_scale': args.joint_accel_scale,
        },
    })
    target = TrajectoryTarget(offsets, gain=args.gain, feedforward=args.feedforward, max_linear=4 * speed)
    # Without feedforward a P servo trails a goal moving at v by v/K by design: allow that on top.
    expected_lag = 0.0 if args.feedforward else speed / args.gain
    monitor = SafetyMonitor(max_error=0.10 + expected_lag) if args.safety else None
    if monitor is not None:
        from baxter_core_msgs.msg import EndpointState

        endpoint = {}
        teleop.node.create_subscription(
            EndpointState, f'/robot/limb/{args.arm}/endpoint_state', lambda m: endpoint.update(msg=m), 10
        )

    def key_source(timeout):
        if target.done:
            return '\x1b'
        if monitor is not None and 'msg' in endpoint:
            p, f = endpoint['msg'].pose.position, endpoint['msg'].wrench.force
            goal = None if target.goal is None else target.goal[0]
            if monitor.update(time.monotonic(), goal, (p.x, p.y, p.z), (f.x, f.y, f.z)):
                return '\x1b'
        time.sleep(timeout)
        return None

    teleop.key_source = key_source
    print(f'{args.ik}: {len(offsets) * DT:.0f} s trajectory, phases ' + ', '.join(p[0] for p in phases))
    try:
        teleop.spin_target(target)
    finally:
        teleop.close()
        rclpy.shutdown()
    if monitor is not None and monitor.reason:
        print(f'ABORTED: {monitor.reason}')
        sys.exit(3)


if __name__ == '__main__':
    main()

"""Command line of the teleop node: `python -m baxter_teleop [options]` (scripts/run_teleop.sh runs it)."""

import argparse
import os
import subprocess
import sys

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory

from .inputs.vr import VRTarget
from .settings import DEFAULTS, servo_feedback_for
from .teleop import Teleop


def config_file(name):
    """A path as given, else a file in the installed config/."""
    if os.path.exists(name):
        return name
    return os.path.join(get_package_share_directory('baxter_teleop'), 'config', name)


def parser():
    p = argparse.ArgumentParser(prog='python -m baxter_teleop', description='Baxter arm teleop')
    p.add_argument('--arm', choices=['left', 'right'], default='right', help='arm to control (default: right)')
    p.add_argument('--input', choices=['keyboard', 'vr'], default='keyboard', help='command source')
    p.add_argument(
        '--vr-topic', default=None, help='VR target topic (default: /vive/{arm}/target from libsurvive_ros2)'
    )
    p.add_argument(
        '--swap-hands',
        choices=['true', 'false'],
        default=None,
        help="override the VR config's swap_hands (true: facing Baxter, each arm driven by the other hand's "
        "controller; false: on Baxter's side, x and y of the axis maps negated)",
    )
    p.add_argument(
        '--vr-config',
        default=DEFAULTS['vr_config'],
        help='VR mapping YAML, a path or a file in the installed config/ (default: %(default)s)',
    )
    p.add_argument(
        '--vr-gain', type=float, default=DEFAULTS['vr_gain'], help='VR pose servo gain in 1/s (default: %(default)s)'
    )
    p.add_argument(
        '--vr-trigger-topic', default=None, help='VR trigger topic for the gripper (default: /vive/{arm}/trigger)'
    )
    p.add_argument(
        '--vr-b-button-topic',
        default=None,
        help='VR B button topic, sends the arm home (default: /vive/{arm}/b_button)',
    )
    p.add_argument(
        '--no-gripper',
        action='store_true',
        help='leave the gripper to the cuff buttons: the VR trigger does not drive it and going home (B, '
        'H, the home button) does not open it',
    )
    p.add_argument(
        '--home-button',
        default='right_button_ok',
        help="robot button (DigitalIO component) that sends the arm home, as B; every arm's teleop listens, "
        "so one press homes both (default: the right navigator's OK; '' for none)",
    )
    p.add_argument(
        '--output',
        choices=['velocity', 'raw', 'raw_id'],
        default=DEFAULTS['output'],
        help="joint command (default: %(default)s): VELOCITY_MODE, 'raw' a jerk-limited setpoint streamed at 200 Hz in "
        "RAW_POSITION_MODE, 'raw_id' raw plus inverse dynamics feedforward (raw modes bypass collision avoidance)",
    )
    p.add_argument(
        '--ik',
        choices=['dls', 'bounded'],
        default='bounded',
        help="IK: 'bounded' (default) least squares with joint limits as hard bounds, so a joint at its "
        "limit no longer stops the hand; 'dls' scales the whole motion to a stop at a limit",
    )
    p.add_argument(
        '--servo-feedback',
        choices=['measured', 'setpoint'],
        default=None,
        help='what the pose servo closes on: the measured arm or, with a raw output, its own setpoint, so '
        'the Cartesian loop cannot hunt on joint friction (pick-and-place bag 20261016_212004); default: '
        f'{DEFAULTS["servo_feedback"]} with a raw output, measured with velocity',
    )
    p.add_argument(
        '--joint-accel-scale',
        type=float,
        default=1.0,
        help='scale the joint acceleration limits (4 rad/s^2 shoulder/elbow, 6 wrist); raw outputs overshoot '
        'reversals when the servo asks for more than the limit allows (VR bag 20261016_211106: 39-44%% of cycles)',
    )
    p.add_argument(
        '--posture-gain',
        type=float,
        default=DEFAULTS['posture_gain'],
        help='pull the elbow toward untuck in the null space of the hand task, in 1/s (0 = off; default: '
        '%(default)s); 1/s cost circle tracking in the sim (posture_damping report), 0.5 did not',
    )
    p.add_argument(
        '--null',
        dest='posture_gain',
        action='store_const',
        const=1.4,
        default=argparse.SUPPRESS,
        help='the null-space untuck pull at 1.4/s; with --posture-gain, the last option wins',
    )
    p.add_argument(
        '--frame', choices=['tool', 'base'], default='base', help='frame keys are expressed in (default: base)'
    )
    p.add_argument(
        '--tcp-offset',
        type=float,
        default=None,
        help="TCP as [0, 0, Z] in the wrist frame, metres (default: teleop.yaml's tcp.offset)",
    )
    p.add_argument(
        '--linear-speed', type=float, default=None, help='key speed / VR cap in m/s (default: keys 0.20, VR 2.0)'
    )
    p.add_argument(
        '--angular-speed', type=float, default=None, help='key speed / VR cap in rad/s (default: keys 0.8, VR 5.0)'
    )
    p.add_argument('--key-hold-initial', type=float, default=0.55, help='seconds a first keypress stays active')
    p.add_argument('--max-linear-velocity', type=float, default=1.40, help='max linear velocity in m/s')
    p.add_argument('--max-angular-velocity', type=float, default=14.00, help='max angular velocity in rad/s')
    p.add_argument('--max-linear-delta', type=float, default=0.020, help='max linear delta per cycle in meters')
    p.add_argument('--max-angular-delta', type=float, default=0.200, help='max angular delta per cycle in radians')
    p.add_argument('--max-delta-jump', type=float, default=0.300, help='max Euclidean jump between consecutive deltas')
    p.add_argument('--gripper-step', type=float, default=2.0, help='gripper position delta per keypress in percent')
    return p


def main(argv=None):
    args, ros_args = parser().parse_known_args(sys.argv[1:] if argv is None else argv)
    args.servo_feedback = servo_feedback_for(args.output, args.servo_feedback)
    if args.input == 'vr':
        args.vr_config = config_file(args.vr_config)
        vr_linear, vr_angular = Teleop.VR_SPEEDS
        args.linear_speed = vr_linear if args.linear_speed is None else args.linear_speed
        args.angular_speed = vr_angular if args.angular_speed is None else args.angular_speed

    rclpy.init(args=[sys.argv[0], *ros_args])
    teleop = Teleop(
        arm=args.arm,
        output=args.output,
        ik=args.ik,
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
    teleop.gripper.enabled = not args.no_gripper
    config = {'args': vars(args), 'tcp_offset': teleop.tcp_offset.tolist(), 'ik': type(teleop.ik).__name__}
    # The controller (libsurvive_ros2 /vive/{hand}/...) driving this arm; swap_hands crosses them.
    hand = args.arm
    swap_hands = None if args.swap_hands is None else args.swap_hands == 'true'
    if args.input == 'vr':
        with open(args.vr_config) as f:
            config['vr_config'] = f.read()
        config['swap_hands'] = (
            yaml.safe_load(config['vr_config']).get('swap_hands', False) if swap_hands is None else swap_hands
        )
        if config['swap_hands']:
            hand = {'left': 'right', 'right': 'left'}[args.arm]
    source = os.path.dirname(os.path.realpath(__file__))
    config['git'] = subprocess.run(
        ['git', '-C', source, 'describe', '--always', '--dirty'], capture_output=True, text=True
    ).stdout.strip()
    teleop.telemetry.config(config)
    try:
        if args.input == 'vr':
            teleop.spin_vr(
                trigger_topic=args.vr_trigger_topic or f'/vive/{hand}/trigger',
                b_button_topic=args.vr_b_button_topic or f'/vive/{hand}/b_button',
                home_button=args.home_button,
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
        teleop.close()
        rclpy.shutdown()

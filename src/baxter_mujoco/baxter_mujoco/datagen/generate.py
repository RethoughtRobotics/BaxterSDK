"""Record scripted demonstrations of a task in the sim.

Needs the sim with the task's scene, the teleop in policy mode and the recorder, on one ROS domain:

    scripts/run_sim_teleop.sh --cup --headless --left --policy   # sim with cameras, RViz, teleop
    uv run recorder record --config baxter_sim.yaml --session NAME   # RethoughtAI
    uv run --group ros python -m baxter_mujoco.datagen.generate --task cup --episodes 55 --seed 0

Every episode starts and ends at home, under its own random lighting (scenes.sample_lighting: lights
moved, scaled, tinted, shadows on or off) unless --lab-lighting. A failed episode is discarded and a
new layout tried, so the session ends with exactly --episodes saved, from the seed's layouts.
"""

import argparse
import os
import threading
import time

import numpy as np
import rclpy
from baxter_core_msgs.msg import JointCommand
from geometry_msgs.msg import PoseArray
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from baxter_mujoco import scenes
from baxter_mujoco.datagen.cup import CupTask
from baxter_mujoco.datagen.motion import RATE, PolicyArm, wait_for_subscriber
from baxter_mujoco.datagen.navigator import Navigator

TASKS = {task.name: task for task in (CupTask(),)}
OBJECTS = ('cup', 'container')  # /sim/objects order
SETTLE = 1.0  # s after placing the layout
JOINTS = ('s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2')
PARK_TOLERANCE = 0.01  # rad, every joint


def park(node, arm, q, timeout=20.0):
    """Move an arm the teleop does not drive to joints q (JOINTS order) and leave it there: the sim holds
    a position command until the next one."""
    names = [f'{arm}_{j}' for j in JOINTS]
    pub = node.create_publisher(JointCommand, f'/robot/limb/{arm}/joint_command', 10)
    state = {}
    sub = node.create_subscription(
        JointState, '/robot/joint_states', lambda m: state.update(zip(m.name, m.position)), 10
    )
    wait_for_subscriber(pub)
    command = JointCommand(mode=JointCommand.POSITION_MODE, command=[float(v) for v in q], names=names)
    end = time.monotonic() + timeout
    try:
        while not (state and max(abs(state[n] - v) for n, v in zip(names, q)) < PARK_TOLERANCE):
            if time.monotonic() > end:
                raise RuntimeError(f'{arm} arm did not reach its parked pose in {timeout} s')
            pub.publish(command)
            time.sleep(1 / RATE)
    finally:
        node.destroy_subscription(sub)
        node.destroy_publisher(pub)


class Generator:
    def __init__(self, node, task, arm='left'):
        self.node, self.task, self.arm, self.navigator = node, task, PolicyArm(node, arm), Navigator(node)
        self._layout = node.create_publisher(Float64MultiArray, '/sim/layout', 10)
        self._lighting = node.create_publisher(Float64MultiArray, '/sim/lighting', 10)
        self._objects = None
        node.create_subscription(PoseArray, '/sim/objects', lambda m: setattr(self, '_objects', m), 10)

    def require_sim(self, timeout=10.0):
        """Refuse to run without the sim: this drives /policy/*/target, which a real robot follows."""
        end = time.monotonic() + timeout
        while self._objects is None:
            if time.monotonic() > end:
                domain = os.environ.get('ROS_DOMAIN_ID', '0 (unset)')
                raise SystemExit(f'no /sim/objects on ROS_DOMAIN_ID={domain} in {timeout} s: start the sim scene there')
            time.sleep(0.05)

    def objects(self):
        return {name: [p.position.x, p.position.y, p.position.z] for name, p in zip(OBJECTS, self._objects.poses)}

    def episode(self, layout, lighting):
        """One recorded demonstration from home to home; returns whether it succeeded."""
        self.arm.go_home()
        for pub, values in ((self._layout, layout), (self._lighting, lighting)):
            wait_for_subscriber(pub)
            pub.publish(Float64MultiArray(data=[float(v) for v in values]))
        time.sleep(SETTLE)
        home = self.arm.tcp()
        self.navigator.press('record')
        self.arm.hold(home, 0.5)  # engage on the current pose: the target maps 1:1 onto the TCP
        self.arm.follow(self.task.plan(layout, home), home)
        self.navigator.press('record')
        time.sleep(0.5)
        ok = self.task.success(layout, self.objects())
        self.navigator.press('save' if ok else 'discard')
        return ok

    def run(self, episodes, seed, randomize_lighting=True):
        self.require_sim()
        for arm, q in self.task.parked.items():
            park(self.node, arm, q)
        rng = np.random.default_rng(seed)
        light_rng = np.random.default_rng([seed, 1])  # its own stream: a seed's layouts do not depend on it
        saved = attempts = 0
        while saved < episodes:
            layout = self.task.sample(rng)
            lighting = scenes.sample_lighting(light_rng) if randomize_lighting else scenes.lab_lighting()
            attempts += 1
            ok = self.episode(layout, lighting)
            saved += ok
            print(
                f'attempt {attempts}: {"saved" if ok else "discarded"} ({saved}/{episodes}), layout '
                f'{np.round(layout, 3).tolist()}',
                flush=True,
            )
        self.arm.go_home()


def main():
    parser = argparse.ArgumentParser(description='Record scripted demonstrations of a task in the sim')
    parser.add_argument('--task', choices=sorted(TASKS), required=True)
    parser.add_argument('--episodes', type=int, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--arm', default='left')
    parser.add_argument(
        '--lab-lighting', action='store_true', help="the lab's lighting in every episode, not randomised"
    )
    args, ros_args = parser.parse_known_args()
    rclpy.init(args=['datagen', *ros_args])
    node = rclpy.create_node('datagen')
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    Generator(node, TASKS[args.task], args.arm).run(args.episodes, args.seed, not args.lab_lighting)
    rclpy.shutdown()


if __name__ == '__main__':
    main()

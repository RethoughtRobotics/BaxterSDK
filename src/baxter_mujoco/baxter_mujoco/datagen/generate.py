"""Record scripted demonstrations of a task in the sim.

Needs the sim with the task's scene, the teleop in policy mode and the recorder, on one ROS domain:

    scripts/run_sim_teleop.sh --cup --left --policy           # sim (headless), RViz, teleop
    uv run recorder record --config baxter_sim.yaml --session NAME   # RethoughtAI
    uv run --group ros python -m baxter_mujoco.datagen.generate --task cup --episodes 55 --seed 0

Every episode starts and ends at home. A failed episode is discarded and a new layout tried, so the
session ends with exactly --episodes saved, from the seed's layouts.
"""

import argparse
import os
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Float64MultiArray

from baxter_mujoco.datagen.cup import CupTask
from baxter_mujoco.datagen.motion import PolicyArm, wait_for_subscriber
from baxter_mujoco.datagen.navigator import Navigator

TASKS = {task.name: task for task in (CupTask(),)}
OBJECTS = ('cup', 'container')  # /sim/objects order
SETTLE = 1.0  # s after placing the layout


class Generator:
    def __init__(self, node, task, arm='left'):
        self.task, self.arm, self.navigator = task, PolicyArm(node, arm), Navigator(node)
        self._layout = node.create_publisher(Float64MultiArray, '/sim/layout', 10)
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

    def episode(self, layout):
        """One recorded demonstration from home to home; returns whether it succeeded."""
        self.arm.go_home()
        wait_for_subscriber(self._layout)
        self._layout.publish(Float64MultiArray(data=[float(v) for v in layout]))
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

    def run(self, episodes, seed):
        self.require_sim()
        rng = np.random.default_rng(seed)
        saved = attempts = 0
        while saved < episodes:
            layout = self.task.sample(rng)
            attempts += 1
            ok = self.episode(layout)
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
    args, ros_args = parser.parse_known_args()
    rclpy.init(args=['datagen', *ros_args])
    node = rclpy.create_node('datagen')
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    Generator(node, TASKS[args.task], args.arm).run(args.episodes, args.seed)
    rclpy.shutdown()


if __name__ == '__main__':
    main()

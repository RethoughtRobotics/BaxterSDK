"""Run a policy closed loop on seeded layouts and score every attempt.

Needs the sim with the task's scene, the teleop in policy mode and the policy runtime, on one ROS domain:

    scripts/run_sim_teleop.sh --cup --cameras --left --policy     # sim, cameras, RViz, teleop
    scripts/run_inference --NAME STEP                             # RethoughtAI: not started
    uv run --group ros python -m baxter_mujoco.datagen.evaluate --task cup --seeds 1000-1049 --out eval.jsonl

A seed's attempt has the layout and lighting the generator gives that seed's first episode
(generate.conditions), the parked arms parked and the arm and gripper at home. The policy is then
engaged through the runtime's engage service (as its scripts/start) for at most --timeout s, stopped at
success or a stall, released and sent home. One JSON line per attempt (outcome.py); seeds already in
--out are skipped, so a rerun resumes. With --bags, a bag per attempt of BAG_TOPICS (and the policy's
cameras with --bag-images), from just before the engage to the release, its chunks zstd compressed
(the cameras' 2 GB of an attempt to about a ninth). The expert's chunk at every frame of a bag:
python -m baxter_mujoco.datagen.expert.
"""

import argparse
import json
import signal
import subprocess
import threading
import time
from pathlib import Path

import rclpy
from baxter_core_msgs.msg import EndEffectorState
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import RobotTrajectory
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Bool, Empty
from std_srvs.srv import SetBool

from baxter_mujoco import scenes
from baxter_mujoco.datagen.generate import TASKS, Scene, conditions, park
from baxter_mujoco.datagen.motion import HOME_TIME, wait_for_subscriber
from baxter_mujoco.datagen.outcome import Attempt

TIMEOUT = 40.0  # s per attempt
SAMPLE = 25.0  # Hz, the datasets' rate
BAG_TOPICS = (
    '/robot/joint_states',
    '/teleop/{arm}/tcp',
    '/teleop/{arm}/goal',
    '/robot/end_effector/{arm}_gripper/state',
    '/policy/{arm}/target',
    '/policy/{arm}/trigger',
    '/policy/{arm}/chunk',
    '/policy/{arm}/engaged',
    '/sim/objects',
)
BAG_IMAGES = ('/cameras/{arm}_hand_camera/image', '/cameras/third_eye/color/image_raw')


def seed_list(text):
    """'1000-1049', '500,503,510-519' as a list of ints."""
    seeds = []
    for part in text.split(','):
        lo, _, hi = part.partition('-')
        seeds += range(int(lo), int(hi or lo) + 1)
    return seeds


class Evaluator:
    def __init__(self, node, task, arm='left', home_topic='/vive/right/b_button', cup_color='blue'):
        self.node, self.task, self.arm, self.scene = node, task, arm, Scene(node, cup_color)
        self.tcp = self.grip = self.engaged = None
        self.chunks = 0
        node.create_subscription(PoseStamped, f'/teleop/{arm}/tcp', lambda m: setattr(self, 'tcp', m), 10)
        node.create_subscription(
            EndEffectorState, f'/robot/end_effector/{arm}_gripper/state', lambda m: setattr(self, 'grip', m), 10
        )
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        node.create_subscription(Bool, f'/policy/{arm}/engaged', lambda m: setattr(self, 'engaged', m.data), latched)
        node.create_subscription(RobotTrajectory, f'/policy/{arm}/chunk', self._on_chunk, 10)
        self._engage = node.create_client(SetBool, f'/policy/{arm}/engage')
        self._home = node.create_publisher(Empty, home_topic, 10)

    def _on_chunk(self, msg):
        self.chunks += 1

    def wait(self, condition, timeout, what):
        end = time.monotonic() + timeout
        while not condition():
            if time.monotonic() > end:
                raise RuntimeError(f'{what} after {timeout:.0f} s')
            time.sleep(0.05)

    def require_policy(self, timeout):
        """The sim, the teleop and a policy server publishing chunks."""
        self.scene.require()
        self.wait(
            lambda: self.tcp is not None and self.grip is not None, 30, 'no TCP or gripper state: is the teleop up?'
        )
        self.wait(self._engage.service_is_ready, timeout, f'no {self._engage.srv_name}: is the policy runtime up?')
        self.wait(lambda: self.chunks > 0, timeout, f'no chunks on /policy/{self.arm}/chunk: is the policy server up?')

    def set_engaged(self, on, timeout=15.0):
        future = self._engage.call_async(SetBool.Request(data=on))
        self.wait(future.done, 10, f'{self._engage.srv_name} did not answer')
        self.wait(lambda: self.engaged is on, timeout, f'the runtime did not {"engage" if on else "release"}')

    def go_home(self):
        """The teleop's home (untuck, gripper open); the runtime clears its chunks on it too."""
        wait_for_subscriber(self._home)
        self._home.publish(Empty())
        time.sleep(HOME_TIME)

    def sample(self):
        p, grip = self.tcp.pose.position, self.grip.position
        poses = self.scene.poses()
        return [p.x, p.y, p.z], grip, *poses['cup'], *poses['container']

    def attempt(self, seed, layout, lighting, timeout, bag=None, bag_topics=()):
        if self.engaged:
            self.set_engaged(False)
        self.go_home()
        park_failed = []
        for arm, q in self.task.parked.items():
            try:
                park(self.node, arm, q)
            except RuntimeError as e:
                # an object the last attempt left against the arm, or the policy arm not yet home: the new
                # layout clears the one, home again the other, then park again; if the arm still does not
                # get there, the attempt runs from where it is and its row says so
                print(f'seed {seed}: {e}, retrying with the scene reset and home again', flush=True)
                self.scene.set(layout, lighting)
                self.go_home()
                try:
                    park(self.node, arm, q, timeout=40.0)
                except RuntimeError as e2:
                    print(f'seed {seed}: {e2}, attempting anyway', flush=True)
                    park_failed.append(arm)
        self.scene.set(layout, lighting)
        recorder = None
        if bag:
            recorder = subprocess.Popen(
                ['ros2', 'bag', 'record', '-o', str(bag), '--storage-preset-profile', 'zstd_fast', *bag_topics],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            time.sleep(3.0)  # its subscriptions discovered: zenoh drops what is sent before
        try:
            self.set_engaged(True)
            score, start, stop = Attempt(self.task, layout), time.monotonic(), None
            while (now := time.monotonic()) - start < timeout:
                score.add(now - start, *self.sample())
                if stop := score.done():
                    break
                time.sleep(max(0.0, start + len(score.t) / SAMPLE - time.monotonic()))
            duration = time.monotonic() - start
            self.set_engaged(False)
        finally:
            if recorder:
                recorder.send_signal(signal.SIGINT)
                try:
                    recorder.wait(30)
                except subprocess.TimeoutExpired:
                    recorder.kill()
        row = {'seed': seed, **score.outcome(stalled=stop == 'stalled'), 'duration': round(duration, 2)}
        if park_failed:
            row['park_failed'] = park_failed
        return {**row, 'layout': [round(float(v), 4) for v in layout], 'bag': str(bag) if bag else None}

    def run(self, seeds, out, timeout=TIMEOUT, randomize_lighting=True, bags=None, bag_images=False, ready=900.0):
        out = Path(out)
        done = set()
        if out.exists():
            done = {json.loads(line)['seed'] for line in out.read_text().splitlines() if line.strip()}
        todo = [s for s in seeds if s not in done]
        print(f'{len(todo)} attempts ({len(seeds) - len(todo)} already in {out})', flush=True)
        if not todo:
            return
        self.require_policy(ready)
        topics = [t.format(arm=self.arm) for t in BAG_TOPICS + (BAG_IMAGES if bag_images else ())]
        out.parent.mkdir(parents=True, exist_ok=True)
        for seed in todo:
            layout, lighting = next(conditions(self.task, seed, randomize_lighting))
            bag = Path(bags) / f'seed_{seed:04d}' if bags else None
            row = self.attempt(seed, layout, lighting, timeout, bag, topics)
            with out.open('a') as f:
                f.write(json.dumps(row) + '\n')
            print(
                f'seed {seed}: {"success in " + str(row["time"]) + " s" if row["success"] else row["failure"]}, '
                f'phase {row["phase"]}',
                flush=True,
            )
        self.go_home()


def main():
    parser = argparse.ArgumentParser(description='Run a policy closed loop on seeded layouts and score it')
    parser.add_argument('--task', choices=sorted(TASKS), required=True)
    parser.add_argument('--seeds', type=seed_list, required=True, help='e.g. 1000-1049 or 500,503,510-519')
    parser.add_argument('--out', required=True, help='JSON lines, one per attempt (appended)')
    parser.add_argument('--arm', default='left')
    parser.add_argument('--timeout', type=float, default=TIMEOUT, help='s per attempt')
    parser.add_argument('--bags', help='a bag per attempt in this directory')
    parser.add_argument('--bag-images', action='store_true', help="with --bags, also the policy's cameras")
    parser.add_argument('--ready-timeout', type=float, default=900.0, help='s to wait for the policy server')
    parser.add_argument(
        '--lab-lighting', action='store_true', help="the lab's lighting in every attempt, not the seed's"
    )
    parser.add_argument('--cup-color', choices=sorted(scenes.CUP_COLORS), default='blue')
    args, ros_args = parser.parse_known_args()
    rclpy.init(args=['evaluate', *ros_args])
    node = rclpy.create_node('evaluate')
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    Evaluator(node, TASKS[args.task], args.arm, cup_color=args.cup_color).run(
        args.seeds, args.out, args.timeout, not args.lab_lighting, args.bags, args.bag_images, args.ready_timeout
    )
    rclpy.shutdown()


if __name__ == '__main__':
    main()

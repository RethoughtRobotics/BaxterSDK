"""Drive an arm as a policy does: a stream of TCP targets and a gripper trigger.

The teleop in policy mode (run_teleop.sh --policy) follows /policy/{arm}/target from the pose the arm
had when the stream started, and reads a target unchanged for 0.2 s as released (it holds and
re-anchors). So a stream starts on the current TCP, and a sub-micron alternation keeps it engaged
while it holds still.
"""

import time

import numpy as np
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation, Slerp
from std_msgs.msg import Empty, Float32

SPEED, SLOW, TURN = 0.18, 0.06, 0.8  # m/s in transit, m/s near objects, rad/s
MIN_MOVE = 0.6  # s, shortest move
RATE = 100.0  # Hz, the runtime's target rate
HOME_TIME = 7.0  # s, the teleop's move to the untuck pose


def wait_for_subscriber(pub, timeout=30.0):
    """Block until `pub` has a subscriber: over zenoh a message sent before discovery is dropped."""
    end = time.monotonic() + timeout
    while pub.get_subscription_count() == 0:
        if time.monotonic() > end:
            raise RuntimeError(f'nothing subscribes to {pub.topic_name} after {timeout} s')
        time.sleep(0.05)


def min_jerk(s):
    """Minimum-jerk time scaling on [0, 1]: zero velocity and acceleration at both ends."""
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def move_duration(start, goal, slow=False):
    """Seconds for a move between (position, Rotation) poses at the transit or slow speed."""
    distance = np.linalg.norm(goal[0] - start[0])
    angle = (start[1].inv() * goal[1]).magnitude()
    return max(distance / (SLOW if slow else SPEED), angle / TURN, MIN_MOVE)


class PolicyArm:
    """One arm's policy interface: targets, trigger, home button, and its measured TCP."""

    def __init__(self, node, arm='left', home_topic='/vive/right/b_button'):
        self.node, self.grip, self._sign = node, 0.0, 1.0
        self._target = node.create_publisher(PoseStamped, f'/policy/{arm}/target', 10)
        self._trigger = node.create_publisher(Float32, f'/policy/{arm}/trigger', 10)
        self._home = node.create_publisher(Empty, home_topic, 10)
        self._tcp = None
        node.create_subscription(PoseStamped, f'/teleop/{arm}/tcp', lambda m: setattr(self, '_tcp', m), 10)

    def tcp(self, timeout=30.0):
        """The measured TCP (position, Rotation), waiting for the first message."""
        end = time.monotonic() + timeout
        while self._tcp is None:
            if time.monotonic() > end:
                raise RuntimeError(f'no TCP in {timeout} s: is the teleop up?')
            time.sleep(0.05)
        p, o = self._tcp.pose.position, self._tcp.pose.orientation
        return np.array([p.x, p.y, p.z]), Rotation.from_quat([o.x, o.y, o.z, o.w])

    def send(self, pose):
        self._sign = -self._sign
        msg = PoseStamped()
        msg.header.stamp, msg.header.frame_id = self.node.get_clock().now().to_msg(), 'base'
        p, o = msg.pose.position, msg.pose.orientation
        p.x, p.y, p.z = (pose[0] + np.array([self._sign * 1e-7, 0.0, 0.0])).tolist()
        o.x, o.y, o.z, o.w = pose[1].as_quat().tolist()
        self._target.publish(msg)
        self._trigger.publish(Float32(data=float(self.grip)))

    def hold(self, pose, seconds):
        for _ in range(max(1, int(seconds * RATE))):
            self.send(pose)
            time.sleep(1 / RATE)

    def move(self, start, goal, slow=False):
        """Minimum-jerk line and slerp from start to goal; returns goal."""
        slerp = Slerp([0, 1], Rotation.concatenate([start[1], goal[1]]))
        steps = int(move_duration(start, goal, slow) * RATE)
        for k in range(1, steps + 1):
            s = min_jerk(k / steps)
            self.send((start[0] + s * (goal[0] - start[0]), slerp(s)))
            time.sleep(1 / RATE)
        return goal

    def follow(self, waypoints, start):
        """Run a task's waypoints from `start`."""
        pose = start
        for w in waypoints:
            goal = (w.position, w.rotation)
            pose = self.move(pose, goal, w.slow)
            self.grip = w.grip
            if w.hold:
                self.hold(pose, w.hold)
        return pose

    def go_home(self):
        """The teleop's home: untuck pose, gripper open."""
        self.grip = 0.0
        wait_for_subscriber(self._home)
        self._home.publish(Empty())
        time.sleep(HOME_TIME)

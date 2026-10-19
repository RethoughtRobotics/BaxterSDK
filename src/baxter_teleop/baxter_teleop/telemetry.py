"""What the teleop asked for and computed, every cycle, on /teleop/{arm}/*, for record_teleop.sh.

tcp_twist  TwistStamped   the TCP twist handed to the IK, in base or tool ({arm}_gripper) axes
ik         JointState     joints used (position), IK joint velocities (velocity), velocities
                          actually sent after the output stage's limits (effort)
goal       PoseStamped    the pose the TCP is servoed toward, base frame
tcp        PoseStamped    the measured TCP (kinematics.tcp_pose), base frame
key        String         each key read, as typed
config     String         the session's settings as JSON, latched
"""

import json

import numpy as np
from geometry_msgs.msg import PoseStamped, TwistStamped
from rclpy.qos import DurabilityPolicy, QoSProfile
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
from std_msgs.msg import String


def pose_msg(position, rotation, stamp, frame_id='base'):
    """PoseStamped from a position and a scipy Rotation or a rotation matrix."""
    if not isinstance(rotation, Rotation):
        rotation = Rotation.from_matrix(rotation)
    msg = PoseStamped()
    msg.header.stamp, msg.header.frame_id = stamp, frame_id
    p, o = msg.pose.position, msg.pose.orientation
    p.x, p.y, p.z = (float(x) for x in position)
    o.x, o.y, o.z, o.w = (float(x) for x in rotation.as_quat())
    return msg


class Telemetry:
    """Publishers for one arm's teleop telemetry (topics in the module docstring)."""

    def __init__(self, node, arm, joint_names):
        self._node, self._arm, self._joint_names = node, arm, joint_names
        ns = f'/teleop/{arm}/'
        self._tcp_twist = node.create_publisher(TwistStamped, ns + 'tcp_twist', 10)
        self._ik = node.create_publisher(JointState, ns + 'ik', 10)
        self._goal = node.create_publisher(PoseStamped, ns + 'goal', 10)
        self._tcp = node.create_publisher(PoseStamped, ns + 'tcp', 10)
        self._key = node.create_publisher(String, ns + 'key', 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._config = node.create_publisher(String, ns + 'config', latched)

    def _now(self):
        return self._node.get_clock().now().to_msg()

    def cycle(self, twist, frame, q, qdot_ik, qdot_sent):
        """One control cycle: the TCP twist (in `frame`, base or tool) and the joints it gave."""
        stamp = self._now()
        msg = TwistStamped()
        msg.header.stamp, msg.header.frame_id = stamp, 'base' if frame == 'base' else f'{self._arm}_gripper'
        t = msg.twist
        t.linear.x, t.linear.y, t.linear.z, t.angular.x, t.angular.y, t.angular.z = (float(x) for x in twist)
        self._tcp_twist.publish(msg)
        joints = JointState(
            name=self._joint_names,
            position=np.asarray(q, float).tolist(),
            velocity=np.asarray(qdot_ik, float).tolist(),
            effort=np.asarray(qdot_sent, float).tolist(),
        )
        joints.header.stamp = stamp
        self._ik.publish(joints)

    def goal(self, goal):
        """The servo goal (position, scipy Rotation) in the base frame."""
        self._goal.publish(pose_msg(goal[0], goal[1], self._now()))

    def tcp(self, position, rotation):
        self._tcp.publish(pose_msg(position, rotation, self._now()))

    def key(self, key):
        self._key.publish(String(data=key))

    def config(self, config):
        self._config.publish(String(data=json.dumps(config, indent=1, default=str)))

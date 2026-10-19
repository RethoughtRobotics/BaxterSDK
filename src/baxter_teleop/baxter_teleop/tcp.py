"""The TCP that teleop servos and that policies observe and act on, defined once.

TCP = frax forward kinematics of the arm chain to {arm}_hand, plus a fixed offset in the hand
frame (measured from the robot's own endpoint, so it includes the fitted gripper and fingers).
Its orientation is the hand's. Published live as /teleop/{arm}/tcp and backfilled into older
recordings by backfill_tcp.py from their joint states, so both carry the same quantity.
"""

import numpy as np
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation


def tcp_pose(hand_transform, tcp_offset):
    """(position (3,), rotation matrix (3, 3)) of the TCP in the base frame."""
    T = np.asarray(hand_transform, dtype=np.float64)
    return T[:3, 3] + T[:3, :3] @ np.asarray(tcp_offset, dtype=np.float64), T[:3, :3]


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

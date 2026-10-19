"""Baxter arm kinematics: the URDF, the arm chain, and the TCP.

The TCP is the point the teleop servos and policies observe and act on, defined once here: frax
forward kinematics of the arm chain to {arm}_hand, plus a fixed offset in the hand frame (measured
from the robot's own endpoint, so it includes the fitted gripper and fingers). Its orientation is
the hand's. Published live as /teleop/{arm}/tcp.
"""

import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation

try:
    from ament_index_python.packages import get_package_share_directory
except ImportError:  # pragma: no cover
    get_package_share_directory = None


def resolve_baxter_urdf():
    """Resolve the Baxter URDF path from env, ROS package index, or workspace."""
    env_path = os.environ.get('BAXTER_FRAX_URDF')
    if env_path:
        if os.path.exists(env_path):
            return env_path
        raise RuntimeError(f'BAXTER_FRAX_URDF is set but file does not exist: {env_path}')

    if get_package_share_directory is not None:
        try:
            # Prefer installed package data when available.
            share_dir = Path(get_package_share_directory('baxter_description'))
            pkg_urdf = share_dir / 'urdf' / 'baxter.urdf'
            if pkg_urdf.exists():
                return str(pkg_urdf)
        except Exception:
            pass

    file_path = Path(__file__).resolve()
    for parent in file_path.parents:
        candidate = parent / 'src' / 'baxter_description' / 'urdf' / 'baxter.urdf'
        if candidate.exists():
            return str(candidate)

    raise RuntimeError(
        'Could not find Baxter URDF. Set BAXTER_FRAX_URDF=/abs/path/to/baxter.urdf '
        '(or install/source baxter_description so it can be discovered via ROS package index).'
    )


def extract_arm_chain_urdf(full_urdf, arm, joint_names):
    """Extract an arm-only serial-chain URDF from full Baxter URDF."""
    chain = ['torso_t0', f'{arm}_torso_arm_mount', *joint_names, f'{arm}_hand']

    root = ET.parse(full_urdf).getroot()
    joints_by_name = {j.attrib['name']: j for j in root.findall('joint')}
    required = [name for name in chain if name not in joints_by_name]
    if required:
        raise RuntimeError(f'Baxter URDF missing expected joints for {arm} arm: {required}')

    keep_joints = set(chain)
    keep_links = {'base'}
    for jname in chain:
        j = joints_by_name[jname]
        # Keep only links that belong to the selected kinematic chain.
        keep_links.add(j.find('parent').attrib['link'])
        keep_links.add(j.find('child').attrib['link'])

    out_root = ET.Element('robot', root.attrib)
    for link in root.findall('link'):
        if link.attrib['name'] in keep_links:
            out_root.append(link)
    for joint in root.findall('joint'):
        if joint.attrib['name'] in keep_joints:
            out_root.append(joint)

    out_path = os.path.join(tempfile.gettempdir(), f'baxter_{arm}_frax_chain.urdf')
    ET.ElementTree(out_root).write(out_path, encoding='utf-8', xml_declaration=True)
    return out_path


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

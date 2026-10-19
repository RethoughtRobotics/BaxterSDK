"""Baxter arm kinematics: the URDF, the arm chain, and the TCP.

The TCP is the point the teleop servos and policies observe and act on, defined once here: frax
forward kinematics of the arm chain to {arm}_hand, plus a fixed offset in the hand frame (measured
from the robot's own endpoint, so it includes the fitted gripper and fingers). Its orientation is
the hand's. Published live as /teleop/{arm}/tcp.
"""

import os
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import rclpy

from .settings import joint_names

try:
    import frax
    import jax
    import jax.numpy as jnp
except ImportError as exc:  # pragma: no cover
    frax = jax = jnp = None
    FRAX_IMPORT_ERROR = exc
else:
    FRAX_IMPORT_ERROR = None

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


def require_frax():
    """Raise a clear error when frax and jax are missing (they are pip-only: requirements.txt)."""
    if frax is None:
        raise RuntimeError(
            'frax + jax are required: .venv-ros/bin/pip install -r src/baxter_teleop/requirements.txt'
        ) from FRAX_IMPORT_ERROR


class ArmKinematics:
    """Forward kinematics, Jacobian and TCP of one Baxter arm (frax, compiled with jax.jit).

    Joint vectors are in settings.JOINTS order. The TCP offset starts at the URDF {arm}_gripper frame;
    measure_tcp_offset replaces it with the offset of the robot's own endpoint.
    """

    URDF_TCP_OFFSET = (0.0, 0.0, 0.025)  # the URDF {arm}_gripper frame, in the hand frame (m)

    def __init__(self, arm, warm_up_at):
        require_frax()
        self.joint_names = joint_names(arm)
        manipulator = getattr(getattr(frax, 'core', object()), 'manipulator', None)
        if not manipulator or not hasattr(manipulator, 'Manipulator'):
            raise RuntimeError('This frax version does not expose core.manipulator.Manipulator')
        # A single arm model, so the Jacobian is 6 x 7.
        urdf = extract_arm_chain_urdf(resolve_baxter_urdf(), arm, self.joint_names)
        self.robot = manipulator.Manipulator(urdf, joint_ordering=self.joint_names)
        self.jacobian = jax.jit(self.robot.ee_jacobian)
        self.hand_transform = jax.jit(self.robot.ee_transform)
        # Compile now, so the first command does not stall the control loop.
        q = jnp.asarray(warm_up_at, dtype=jnp.float32)
        self.jacobian(q).block_until_ready()
        self.hand_transform(q).block_until_ready()
        self.tcp_offset = np.asarray(self.URDF_TCP_OFFSET, dtype=np.float32)

    def tcp(self, q):
        """(position (3,), rotation matrix (3, 3)) of the TCP in the base frame at joints q."""
        return tcp_pose(self.hand_transform(jnp.asarray(q)), self.tcp_offset)

    def hand_twist(self, q, twist, frame):
        """Convert a TCP twist into the base-frame twist of the hand link used by the Jacobian.

        In tool frame the twist is expressed in the TCP axes; in base frame it is
        expressed in the robot base axes. Either way rotation is about the TCP.
        """
        R = np.asarray(self.hand_transform(jnp.asarray(q)), dtype=np.float32)[:3, :3]
        v = np.asarray(twist[:3], dtype=np.float32)
        w = np.asarray(twist[3:], dtype=np.float32)
        if frame == 'tool':
            v = R @ v
            w = R @ w
        # v_tcp = v_hand + w x r, with r the hand-to-TCP vector in base frame.
        v_hand = v - np.cross(w, R @ self.tcp_offset)
        return np.concatenate([v_hand, w]).astype(np.float32)

    def measure_tcp_offset(self, node, limb, read_q, num_samples=10, sample_period=0.02, timeout=3.0):
        """Set the TCP offset to the robot endpoint expressed in the hand frame, averaged over samples.

        The endpoint includes the configured gripper and fingers. Without one in `timeout` seconds
        the offset stays at the URDF gripper frame. read_q() gives the current joints.
        """
        samples = []
        deadline = time.monotonic() + timeout
        next_sample = time.monotonic()
        while len(samples) < num_samples and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.005)
            if time.monotonic() < next_sample:
                continue
            pose = limb.endpoint_pose()
            if pose and 'position' in pose:
                T = np.asarray(self.hand_transform(jnp.asarray(read_q())), dtype=np.float32)
                p = np.asarray(pose['position'], dtype=np.float32)
                samples.append(T[:3, :3].T @ (p - T[:3, 3]))
                next_sample = time.monotonic() + sample_period
        if not samples:
            node.get_logger().warn('No endpoint state received; using URDF gripper frame as TCP.')
            return
        samples = np.asarray(samples)
        spread_mm = float(np.max(np.linalg.norm(samples - samples.mean(axis=0), axis=1))) * 1000.0
        node.get_logger().info(f'TCP offset from {len(samples)} samples, max deviation {spread_mm:.1f} mm')
        self.tcp_offset = samples.mean(axis=0).astype(np.float32)

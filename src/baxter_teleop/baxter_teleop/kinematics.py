"""Baxter arm kinematics: the URDF, one arm's model, and the TCP (Pinocchio).

The TCP is the point the teleop servos and policies observe and act on, defined once here: the
forward kinematics of the arm to its {arm}_wrist frame (the w2 joint), plus a fixed offset in that
frame, measured from the robot's own endpoint, so it includes the fitted gripper and fingers. Its
orientation is the wrist's (the hand and gripper frames below it are only translated along z).
Published live as /teleop/{arm}/tcp; the recorded tcp_offset (in /teleop/{arm}/config) is in the
wrist frame.

  Carpentier et al., "The Pinocchio C++ library", IEEE/SICE SII 2019. ROS package: pinocchio.
"""

import os
import time
from pathlib import Path

import numpy as np
import pinocchio as pin
import rclpy

from .settings import joint_names

try:
    from ament_index_python.packages import get_package_share_directory
except ImportError:  # pragma: no cover
    get_package_share_directory = None


def resolve_baxter_urdf():
    """The Baxter URDF: BAXTER_URDF, else baxter_description's (installed, then in the source tree)."""
    env_path = os.environ.get('BAXTER_URDF')
    if env_path:
        if os.path.exists(env_path):
            return env_path
        raise RuntimeError(f'BAXTER_URDF is set but file does not exist: {env_path}')
    if get_package_share_directory is not None:
        try:
            pkg_urdf = Path(get_package_share_directory('baxter_description')) / 'urdf' / 'baxter.urdf'
            if pkg_urdf.exists():
                return str(pkg_urdf)
        except Exception:
            pass
    for parent in Path(__file__).resolve().parents:
        candidate = parent / 'src' / 'baxter_description' / 'urdf' / 'baxter.urdf'
        if candidate.exists():
            return str(candidate)
    raise RuntimeError('Could not find the Baxter URDF: source the workspace (baxter_description) or set BAXTER_URDF.')


def tcp_pose(wrist_transform, tcp_offset):
    """(position (3,), rotation matrix (3, 3)) of the TCP in the base frame."""
    T = np.asarray(wrist_transform, dtype=np.float64)
    return T[:3, 3] + T[:3, :3] @ np.asarray(tcp_offset, dtype=np.float64), T[:3, :3]


class ArmKinematics:
    """Forward kinematics, Jacobian and TCP of one Baxter arm.

    The full Baxter URDF, with every joint but this arm's seven held at zero; the Jacobian keeps this
    arm's seven columns, so it is 6 x 7. Joint vectors are in settings.JOINTS order. (Pinocchio's
    reduced model would be the usual way, but buildReducedModel rejects Baxter's URDF, where a link
    and a fixed joint share each name such as left_hand.) The TCP offset starts at the
    URDF {arm}_gripper frame; measure_tcp_offset replaces it with the robot's own endpoint.
    """

    def __init__(self, arm):
        self.joint_names = joint_names(arm)
        self.model = pin.buildModelFromUrdf(resolve_baxter_urdf())
        self._data = self.model.createData()
        joints = [self.model.joints[self.model.getJointId(name)] for name in self.joint_names]
        if any(j.nq != 1 or j.nv != 1 for j in joints):
            raise RuntimeError(f'{arm} arm joints are not all single-axis in the URDF')
        self._iq = np.array([j.idx_q for j in joints])
        self._iv = np.array([j.idx_v for j in joints])
        self._q = pin.neutral(self.model)  # the full configuration, the other joints at zero
        self._wrist = self.model.getFrameId(f'{arm}_wrist', pin.FrameType.BODY)
        gripper = self.model.getFrameId(f'{arm}_gripper', pin.FrameType.BODY)
        pin.framesForwardKinematics(self.model, self._data, self._q)
        to_gripper = self._data.oMf[self._wrist].actInv(self._data.oMf[gripper])
        self.urdf_tcp_offset = np.asarray(to_gripper.translation, dtype=np.float32)  # used without an endpoint
        self.tcp_offset = self.urdf_tcp_offset.copy()

    def _full(self, q):
        self._q[self._iq] = np.asarray(q, dtype=np.float64)
        return self._q

    def wrist_transform(self, q):
        """4 x 4 transform of the wrist frame in the base frame at joints q."""
        pin.framesForwardKinematics(self.model, self._data, self._full(q))
        return self._data.oMf[self._wrist].homogeneous

    def jacobian(self, q):
        """6 x 7 Jacobian of the wrist frame's origin: linear then angular velocity, in base axes."""
        J = pin.computeFrameJacobian(self.model, self._data, self._full(q), self._wrist, pin.LOCAL_WORLD_ALIGNED)
        return J[:, self._iv]

    def inverse_dynamics(self, q, qd, qdd):
        """Joint torques (7,) for this motion, without gravity: M(q) qdd + C(q, qd) qd. Gravity is left
        out because Baxter compensates it itself in torque mode (as the sim does)."""
        v, a = np.zeros(self.model.nv), np.zeros(self.model.nv)
        v[self._iv], a[self._iv] = qd, qdd
        q_full = self._full(q)
        tau = pin.rnea(self.model, self._data, q_full, v, a) - pin.computeGeneralizedGravity(
            self.model, self._data, q_full
        )
        return tau[self._iv]

    def mass_matrix(self, q):
        """7 x 7 joint-space inertia M(q) of the arm."""
        M = pin.crba(self.model, self._data, self._full(q))[np.ix_(self._iv, self._iv)]
        return np.triu(M) + np.triu(M, 1).T  # crba fills the upper triangle

    def tcp(self, q):
        """(position (3,), rotation matrix (3, 3)) of the TCP in the base frame at joints q."""
        return tcp_pose(self.wrist_transform(q), self.tcp_offset)

    def wrist_twist(self, q, twist, frame):
        """Convert a TCP twist into the base-frame twist of the wrist frame the Jacobian is for.

        In tool frame the twist is expressed in the TCP axes; in base frame it is
        expressed in the robot base axes. Either way rotation is about the TCP.
        """
        R = np.asarray(self.wrist_transform(q), dtype=np.float32)[:3, :3]
        v = np.asarray(twist[:3], dtype=np.float32)
        w = np.asarray(twist[3:], dtype=np.float32)
        if frame == 'tool':
            v = R @ v
            w = R @ w
        # v_tcp = v_wrist + w x r, with r the wrist-to-TCP vector in base frame.
        v_wrist = v - np.cross(w, R @ self.tcp_offset)
        return np.concatenate([v_wrist, w]).astype(np.float32)

    def measure_tcp_offset(self, node, limb, read_q, num_samples=10, sample_period=0.02, timeout=3.0):
        """Set the TCP offset to the robot endpoint expressed in the wrist frame, averaged over samples.

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
                T = np.asarray(self.wrist_transform(read_q()), dtype=np.float32)
                p = np.asarray(pose['position'], dtype=np.float32)
                samples.append(T[:3, :3].T @ (p - T[:3, 3]))
                next_sample = time.monotonic() + sample_period
        if not samples:
            node.get_logger().warn('No endpoint state received; using the URDF gripper frame as TCP.')
            return
        samples = np.asarray(samples)
        spread_mm = float(np.max(np.linalg.norm(samples - samples.mean(axis=0), axis=1))) * 1000.0
        node.get_logger().info(f'TCP offset from {len(samples)} samples, max deviation {spread_mm:.1f} mm')
        self.tcp_offset = samples.mean(axis=0).astype(np.float32)

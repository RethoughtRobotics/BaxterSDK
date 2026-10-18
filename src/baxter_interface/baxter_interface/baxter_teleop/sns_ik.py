"""Rethink's SNS-IK as an alternative to DiffIKSolver (velocity mode)."""

import os
import xml.etree.ElementTree as ET

import numpy as np
import yaml
from ament_index_python.packages import get_package_share_directory


def default_joint_limits():
    """Rethink's Baxter limits, as loaded by sns_ik_examples/launch/test_baxter.launch."""
    return os.path.join(get_package_share_directory('sns_ik_py'), 'config', 'example_baxter_joint_limits.yaml')


class SNSIKSolver:
    """Joint velocities from sns_ik::SNS_IK::CartToJntVel (Flacco, De Luca, Khatib).

    Rethink's settings: URDF position limits with the velocity and acceleration limits of a
    Baxter YAML, the standard SNS solver and library defaults. SNS computes its own KDL Jacobian
    and enforces joint position, velocity and acceleration limits by saturating joints in the
    nullspace. With `posture`, its nullspace bias task pulls the joints toward that posture at
    `posture_rate` (1/s) in the nullspace of the hand task, so the redundant elbow does not drift
    (pseudoinverse control is not repeatable: the joints need not return when the hand does).
    Without it the elbow is left free, as in DiffIKSolver.
    """

    def __init__(self, urdf_path, arm, joint_names, dt, joint_limits=None, posture=None, posture_rate=1.0):
        from sns_ik_py import SnsIk

        with open(urdf_path) as f:
            urdf_xml = f.read()
        with open(joint_limits or default_joint_limits()) as f:
            limits = yaml.safe_load(f)['joint_limits']
        # rosparam would load the has_*_limits flags too, but SNS_IK only reads the numeric max_/min_ keys.
        limits = {
            joint: {key: float(value) for key, value in entries.items() if key.startswith(('max_', 'min_'))}
            for joint, entries in limits.items()
        }
        # Rethink's chain base -> {arm}_hand. The runtime's twists are about the frax end effector,
        # the {arm}_wrist origin (frax drops the fixed {arm}_hand joint), so each twist is moved to
        # the {arm}_hand origin: same axes, offset by the fixed joint's origin.
        hand_joint = ET.fromstring(urdf_xml).find(f"joint[@name='{arm}_hand']/origin")
        if hand_joint is None or any(float(a) for a in hand_joint.get('rpy', '0 0 0').split()):
            raise RuntimeError(f'Expected a fixed {arm}_hand joint with no rotation in the Baxter URDF')
        self.wrist_to_hand = np.array(hand_joint.get('xyz').split(), dtype=np.float64)
        self.ik = SnsIk(urdf_xml, 'base', f'{arm}_hand', limits, dt)
        if self.ik.joint_names() != list(joint_names):
            raise RuntimeError(f'SNS-IK chain joints {self.ik.joint_names()} do not match {list(joint_names)}')
        self.q_low, self.q_high = (np.asarray(limit) for limit in self.ik.position_limits())
        # SNS asks for nullspace velocity gain * (q_bias - q) / dt, gain in [0, 1].
        self.posture = [] if posture is None else [float(x) for x in posture]
        self.ik.set_nullspace_gain(min(1.0, posture_rate * dt))

    def joint_velocity(
        self,
        q,
        jacobian,
        v_ee,
        joint_velocity_limits=None,
        joint_position_limits=None,
        joint_acceleration_limits=None,
    ):
        """Same call as DiffIKSolver.joint_velocity; jacobian and limits are unused (SNS has its own)."""
        if not np.any(v_ee):
            # sns_ik_lib cannot scale an all-zero task ("Infinite loop on SNS") and returns large
            # arbitrary joint velocities, so a zero twist holds still.
            return np.zeros(len(q), dtype=np.float32)
        # The real arm can rest slightly past a URDF limit, where SNS's braking bound
        # sqrt(2 a (q_max - q)) is undefined and the solve fails; treat it as at the limit.
        q = np.clip(np.asarray(q, dtype=np.float64), self.q_low, self.q_high).tolist()
        v, w = np.asarray(v_ee[:3], dtype=np.float64), np.asarray(v_ee[3:], dtype=np.float64)
        r = np.asarray(self.ik.tip_rotation(q)) @ self.wrist_to_hand  # wrist -> hand in base axes
        # When SNS cannot execute the task it returns zero joint velocities, so the arm holds.
        qdot = self.ik.cart_to_jnt_vel(q, np.concatenate([v + np.cross(w, r), w]).tolist(), self.posture)
        return np.asarray(qdot, dtype=np.float32)

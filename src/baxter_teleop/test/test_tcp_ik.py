"""tcp_ik: joints along a TCP path reach every pose and move continuously from where they start."""

import numpy as np
from scipy.spatial.transform import Rotation

from baxter_teleop.kinematics import ArmKinematics
from baxter_teleop.settings import LIMITS
from baxter_teleop.tcp_ik import joints_along, solve

HOME = np.array([-0.1124, -1.0097, -1.1317, 1.9033, 0.5854, 1.0585, -0.7055])


def motion(n=200):
    """A smooth 4 s joint motion from home, every joint swinging by up to 0.3 rad."""
    t = np.linspace(0.0, 4.0, n)[:, None]
    return HOME + 0.3 * np.sin(2 * np.pi * t / 4.0 * np.arange(1, 8) / 4.0)


def tcp_path(kinematics, q):
    poses = [kinematics.tcp(row) for row in q]
    return np.array([p for p, _ in poses]), Rotation.from_matrix([R for _, R in poses])


def test_reaches_a_pose_from_nearby():
    kinematics = ArmKinematics('left')
    target = HOME + 0.05
    p, R = kinematics.tcp(target)
    _, (error, angle) = solve(kinematics, HOME, p, Rotation.from_matrix(R))
    assert error < 1e-4 and angle < 1e-3


def test_follows_a_tcp_path_continuously():
    kinematics = ArmKinematics('left')
    q = motion()
    positions, rotations = tcp_path(kinematics, q)
    solved, errors = joints_along(kinematics, q[0], positions, rotations, LIMITS.position_low, LIMITS.position_high)
    assert errors.max() < 1e-4  # every pose reached within 0.1 mm
    assert np.abs(np.diff(solved, axis=0)).max() < 0.05  # no jumps: at most 0.05 rad per 20 ms sample
    reached = tcp_path(kinematics, solved)
    assert (reached[1] * rotations.inv()).magnitude().max() < 1e-3

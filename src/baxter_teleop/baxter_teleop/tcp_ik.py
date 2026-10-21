"""Joint angles along a TCP path, moving the joints as little as possible from one sample to the next.

  Nakamura, "Advanced Robotics: Redundancy and Optimization", Addison-Wesley 1991 (minimum-norm solutions).
  Buss, "Introduction to inverse kinematics with Jacobian transpose, pseudoinverse and damped least
  squares methods", 2004.

Baxter's arm has 7 joints and a TCP pose fixes 6 numbers, so every pose has a one-parameter family of
joint solutions (the elbow can swing about the shoulder-wrist line). A replay of joint angles never
meets that choice; a replay of TCP poses (a demo's or a policy's) must make it. Here each sample starts
from the previous sample's solution and takes damped least-squares steps, the minimum-norm joint change
that removes the pose error, so the free joint stays where the previous sample left it, as it does when
the arm is guided by hand. Nothing pulls it toward a preferred posture.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def solve(kinematics, q, position, rotation, low=None, high=None, iterations=30, damping=1e-3, tolerance=(1e-5, 1e-4)):
    """Joint angles reaching (position (3,), Rotation) from q, and the final error (m, rad)."""
    q = np.asarray(q, dtype=float).copy()
    for _ in range(iterations):
        p, R = kinematics.tcp(q)
        dp = np.asarray(position, dtype=float) - p
        dr = (rotation * Rotation.from_matrix(R).inv()).as_rotvec()
        if np.linalg.norm(dp) < tolerance[0] and np.linalg.norm(dr) < tolerance[1]:
            break
        J = np.asarray(kinematics.jacobian(q), dtype=float)
        twist = np.asarray(kinematics.wrist_twist(q, np.concatenate([dp, dr]), 'base'), dtype=float)
        q += J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), twist)
        if low is not None:
            q = np.clip(q, low, high)
    p, R = kinematics.tcp(q)
    return q, (np.linalg.norm(np.asarray(position) - p), (rotation * Rotation.from_matrix(R).inv()).magnitude())


def joints_along(kinematics, q_start, positions, rotations, low=None, high=None):
    """(N, 7) joint angles along a TCP path (positions (N, 3), Rotations), each sample solved from the
    previous one and the first from q_start; and each sample's remaining position error (N,) in m."""
    q, out, errors = np.asarray(q_start, dtype=float), [], []
    for k in range(len(positions)):
        q, (error, _) = solve(kinematics, q, positions[k], rotations[k], low, high)
        out.append(q)
        errors.append(error)
    return np.array(out), np.array(errors)

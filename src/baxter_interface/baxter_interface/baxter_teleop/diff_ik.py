"""Differential IK utilities for Baxter teleop."""

import numpy as np


class DiffIKSolver:
    """Damped least-squares differential IK with a nullspace posture attractor.

    Baxter has 7 joints for a 6-DOF end-effector task, leaving one redundant
    degree of freedom (mostly elbow swing: e0 with s0/w0). Like the Franka
    cartesian impedance controller, the redundancy is resolved by pulling the
    joints toward a rest posture inside the Jacobian nullspace, which does not
    disturb the end-effector motion.
    """

    def __init__(
        self,
        dt=0.01,
        damping=0.05,
        max_joint_step=0.02,
        max_joint_velocity=1.5,
        null_gain=0.0,
        max_null_velocity=0.5,
    ):
        self.dt = float(dt)
        self.damping = float(damping)
        self.max_joint_step = float(max_joint_step)
        self.max_joint_velocity = float(max_joint_velocity)
        self.null_gain = float(null_gain)  # 1/s, rate of pull toward the rest posture
        self.max_null_velocity = float(max_null_velocity)  # rad/s cap on the nullspace term

    def _pinv_and_nullspace(self, J):
        # Damped least squares remains stable near singular Jacobians.
        JJT = J @ J.T + (self.damping**2) * np.eye(J.shape[0], dtype=np.float32)
        J_pinv = J.T @ np.linalg.inv(JJT)
        N = np.eye(J.shape[1], dtype=np.float32) - J_pinv @ J
        return J_pinv, N

    def _nullspace_velocity(self, N, q, q_rest):
        if q_rest is None or self.null_gain <= 0.0:
            return np.zeros(N.shape[0], dtype=np.float32)
        qdot_null = N @ (self.null_gain * (np.asarray(q_rest, dtype=np.float32) - q))
        norm = np.linalg.norm(qdot_null)
        if norm > self.max_null_velocity:
            qdot_null *= self.max_null_velocity / norm
        return qdot_null

    def joint_velocity(self, q, jacobian, v_ee, q_rest=None, joint_velocity_limits=None):
        """Joint velocities (rad/s) realizing EE twist v_ee plus the nullspace posture term.

        The whole vector is scaled uniformly so every joint stays within its
        velocity limit, which preserves the end-effector direction of motion.
        """
        J = np.asarray(jacobian, dtype=np.float32)
        q_np = np.asarray(q, dtype=np.float32)
        J_pinv, N = self._pinv_and_nullspace(J)

        qdot = J_pinv @ np.asarray(v_ee, dtype=np.float32) + self._nullspace_velocity(N, q_np, q_rest)

        limits = (
            np.full_like(qdot, self.max_joint_velocity)
            if joint_velocity_limits is None
            else np.asarray(joint_velocity_limits, dtype=np.float32)
        )
        ratio = float(np.max(np.abs(qdot) / limits))
        if ratio > 1.0:
            qdot /= ratio
        return qdot

    def step_delta(self, q, jacobian, delta_cartesian, q_rest=None):
        """Compute next joint vector from a cartesian delta input.

        Args:
            q: current joint configuration (n,)
            jacobian: end-effector Jacobian (6,n)
            delta_cartesian: desired EE delta [dx,dy,dz,droll,dpitch,dyaw] (6,)
            q_rest: optional rest posture for the nullspace attractor (n,)
        """
        # Keep all math in float32 to match JAX tensors used in the runtime.
        J = np.asarray(jacobian, dtype=np.float32)
        q_np = np.asarray(q, dtype=np.float32)
        J_pinv, N = self._pinv_and_nullspace(J)

        dq = J_pinv @ np.asarray(delta_cartesian, dtype=np.float32)
        dq += self._nullspace_velocity(N, q_np, q_rest) * self.dt

        # Enforce bounded joint increments per control cycle.
        vel_step_limit = self.max_joint_velocity * self.dt
        step_limit = min(self.max_joint_step, vel_step_limit)
        step_norm = np.linalg.norm(dq)
        if step_norm > step_limit:
            dq *= step_limit / max(step_norm, 1e-9)

        return q_np + dq

    def step(self, q, jacobian, v_ee):
        """Compatibility wrapper: solve from EE twist using configured dt."""
        return self.step_delta(q=q, jacobian=jacobian, delta_cartesian=np.asarray(v_ee, dtype=np.float32) * self.dt)

"""Differential IK utilities for Baxter teleop."""

import numpy as np


class DiffIKSolver:
    """Damped least-squares differential IK.

    Baxter has 7 joints for a 6-DOF end-effector task, leaving one redundant degree of
    freedom (mostly elbow swing: e0 with s0/w0). It is left free (minimum-norm, no posture term).
    Holding a joint instead (e.g. e0) makes the arm 6-DOF and singular in parts of its workspace:
    straight down from a pose near untuck it reached 0.083 of 0.2 m/s, against 0.186 free.
    """

    def __init__(
        self,
        dt=0.01,
        damping=0.05,
        max_joint_step=0.02,
        max_joint_velocity=1.5,
    ):
        self.dt = float(dt)
        self.damping = float(damping)
        self.max_joint_step = float(max_joint_step)
        self.max_joint_velocity = float(max_joint_velocity)

    def _pinv(self, J):
        # Damped least squares remains stable near singular Jacobians.
        JJT = J @ J.T + (self.damping**2) * np.eye(J.shape[0], dtype=np.float32)
        return J.T @ np.linalg.inv(JJT)

    def joint_velocity(
        self,
        q,
        jacobian,
        v_ee,
        joint_velocity_limits=None,
        joint_position_limits=None,
        joint_acceleration_limits=None,
    ):
        """Joint velocities (rad/s) realizing EE twist v_ee.

        With joint_position_limits (low, high), a joint heading into a limit may move at most as
        fast as it can still brake from in time (sqrt(2 a d) with joint_acceleration_limits, else
        one control step). The whole vector is scaled down to respect that, so at a limit the
        hand stops in that direction (a wall) instead of the motion being pushed onto the other
        joints, which reconfigures the arm unpredictably (e.g. 20 deg after one reach to the e1
        limit and back). The vector is then scaled uniformly so every joint stays within its
        velocity limit. Both scalings preserve the end-effector direction of motion.
        """
        J = np.asarray(jacobian, dtype=np.float32)
        q_np = np.asarray(q, dtype=np.float32)
        v = np.asarray(v_ee, dtype=np.float32)
        n = J.shape[1]
        low_v = np.full(n, -np.inf, dtype=np.float32)
        high_v = np.full(n, np.inf, dtype=np.float32)
        if joint_position_limits is not None:
            to_low = np.maximum(q_np - np.asarray(joint_position_limits[0], dtype=np.float32), 0.0)
            to_high = np.maximum(np.asarray(joint_position_limits[1], dtype=np.float32) - q_np, 0.0)
            if joint_acceleration_limits is None:
                low_v, high_v = -to_low / self.dt, to_high / self.dt
            else:
                a = np.asarray(joint_acceleration_limits, dtype=np.float32)
                low_v, high_v = -np.sqrt(2.0 * a * to_low), np.sqrt(2.0 * a * to_high)

        qdot = self._pinv(J) @ v
        with np.errstate(divide='ignore', invalid='ignore'):
            allowed = np.where(qdot > 0, high_v / qdot, np.where(qdot < 0, low_v / qdot, np.inf))
        qdot *= float(np.clip(np.min(allowed), 0.0, 1.0))

        limits = (
            np.full_like(qdot, self.max_joint_velocity)
            if joint_velocity_limits is None
            else np.asarray(joint_velocity_limits, dtype=np.float32)
        )
        ratio = float(np.max(np.abs(qdot) / limits))
        if ratio > 1.0:
            qdot /= ratio
        return qdot

    def step_delta(self, q, jacobian, delta_cartesian):
        """Compute next joint vector from a cartesian delta input.

        Args:
            q: current joint configuration (n,)
            jacobian: end-effector Jacobian (6,n)
            delta_cartesian: desired EE delta [dx,dy,dz,droll,dpitch,dyaw] (6,)
        """
        # Keep all math in float32 to match JAX tensors used in the runtime.
        J = np.asarray(jacobian, dtype=np.float32)
        q_np = np.asarray(q, dtype=np.float32)
        dq = self._pinv(J) @ np.asarray(delta_cartesian, dtype=np.float32)

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

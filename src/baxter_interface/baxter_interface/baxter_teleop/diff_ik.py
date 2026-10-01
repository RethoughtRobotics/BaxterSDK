"""Differential IK utilities for Baxter teleop."""

import numpy as np


class DiffIKSolver:
    """Damped least-squares differential IK stepper."""

    def __init__(self, dt=0.01, damping=0.05, max_joint_step=0.02, max_joint_velocity=1.5):
        self.dt = float(dt)
        self.damping = float(damping)
        self.max_joint_step = float(max_joint_step)
        self.max_joint_velocity = float(max_joint_velocity)

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
        delta_np = np.asarray(delta_cartesian, dtype=np.float32)

        # Damped least squares remains stable near singular Jacobians.
        JJT = J @ J.T + (self.damping**2) * np.eye(6, dtype=np.float32)
        dq = J.T @ np.linalg.solve(JJT, delta_np)

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

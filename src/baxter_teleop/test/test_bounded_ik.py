"""BoundedDiffIKSolver: equals DLS when unconstrained, never exceeds its bounds, slides at a limit."""

import numpy as np
import pytest

from baxter_teleop.diff_ik import BoundedDiffIKSolver, DiffIKSolver

VEL = np.array([1.2, 1.2, 1.2, 1.2, 3.2, 3.2, 3.2])
ACC = np.array([4.0, 4.0, 4.0, 4.0, 6.0, 6.0, 6.0])
LOW = np.array([-1.70168, -2.147, -3.05418, 0.35, -3.059, -1.5708, -3.059])
HIGH = np.array([1.70168, 1.047, 3.05418, 2.618, 3.059, 2.094, 3.059])
Q0 = np.array([0.08, -1.0, 1.19, 1.94, -0.67, 1.03, 0.5])


def jacobian(seed):
    """A well-conditioned random 6x7 Jacobian (smallest singular value above the damping threshold)."""
    rng = np.random.default_rng(seed)
    U, _, Vt = np.linalg.svd(rng.normal(size=(6, 7)))
    return U @ np.hstack([np.diag(rng.uniform(0.3, 1.0, 6)), np.zeros((6, 1))]) @ Vt


def solve(solver, J, v, q=Q0):
    return solver.joint_velocity(q, J, v, VEL, (LOW, HIGH), ACC)


@pytest.mark.parametrize('seed', range(10))
def test_unconstrained_solution_equals_dls(seed):
    J, v = jacobian(seed), np.random.default_rng(seed).normal(size=6) * 0.05
    np.testing.assert_allclose(solve(BoundedDiffIKSolver(), J, v), solve(DiffIKSolver(), J, v), atol=1e-3)


@pytest.mark.parametrize('seed', range(20))
def test_never_exceeds_velocity_or_braking_bounds(seed):
    rng = np.random.default_rng(seed)
    q = LOW + rng.uniform(0, 1, 7) * (HIGH - LOW)
    J, v = jacobian(seed), rng.normal(size=6) * 2.0  # often infeasible
    qdot = solve(BoundedDiffIKSolver(), J, v, q)
    assert np.all(np.abs(qdot) <= VEL + 1e-6)
    assert np.all(qdot <= np.sqrt(2 * ACC * (HIGH - q)) + 1e-6)
    assert np.all(qdot >= -np.sqrt(2 * ACC * (q - LOW)) - 1e-6)


def test_at_a_joint_limit_the_hand_slides_instead_of_stopping():
    J = jacobian(3)
    q = Q0.copy()
    q[3] = LOW[3]  # e1 at its lower limit
    # a hand motion that the DLS solution would make with e1 moving further down
    v = J @ np.where(np.arange(7) == 3, -0.5, 0.1)
    wall, bounded = solve(DiffIKSolver(), J, v, q), solve(BoundedDiffIKSolver(), J, v, q)
    assert np.allclose(wall, 0, atol=1e-6)  # the wall stops every joint
    assert bounded[3] >= -1e-9  # e1 does not go past its limit
    err_wall, err_bounded = np.linalg.norm(J @ wall - v), np.linalg.norm(J @ bounded - v)
    assert err_bounded < 0.5 * err_wall  # the other joints achieve most of the hand motion


def test_posture_pull_stays_in_the_null_space():
    J, v = jacobian(5), np.zeros(6)
    posture = Q0 + 0.3
    qdot = BoundedDiffIKSolver(posture=posture, posture_gain=1.0).joint_velocity(Q0, J, v, VEL, (LOW, HIGH), ACC)
    assert np.linalg.norm(qdot) > 1e-3  # it moves toward the posture...
    np.testing.assert_allclose(J @ qdot, 0, atol=1e-3)  # ...without moving the hand

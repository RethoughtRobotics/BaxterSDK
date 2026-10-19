"""JointSetpointGenerator: limits, continuity, convergence, position and lead backstops."""

import numpy as np
import pytest

from baxter_teleop.outputs.setpoint import JointLimits, JointSetpointGenerator

DT = 0.005
# Baxter teleop limits (Teleop): velocity 0.8 x URDF, acceleration as the old ramp,
# jerk 10 x acceleration.
VEL = np.array([1.2, 1.2, 1.2, 1.2, 3.2, 3.2, 3.2])
ACC = np.array([4.0, 4.0, 4.0, 4.0, 6.0, 6.0, 6.0])
LOW = np.array([-1.70168, -2.147, -3.05418, 0.35, -3.059, -1.5708, -3.059])
HIGH = np.array([1.70168, 1.047, 3.05418, 2.618, 3.059, 2.094, 3.059])
Q0 = np.array([0.08, -1.0, 1.19, 1.94, -0.67, 1.03, 0.5])  # right untuck
TOL = 1e-6


def make(jerk_factor=10.0):
    limits = JointLimits(VEL, ACC, ACC * jerk_factor, LOW, HIGH)
    gen = JointSetpointGenerator(limits, DT)
    gen.reset(Q0)
    return gen


def run(gen, steps, target=None):
    if target is not None:
        gen.set_target_velocity(target)
    return [gen.step() for _ in range(steps)]


def test_reset_starts_at_rest_at_the_given_position():
    gen = make()
    np.testing.assert_allclose(gen.state.position, Q0)
    assert not gen.state.velocity.any() and not gen.state.acceleration.any()


def test_step_before_reset_raises():
    gen = JointSetpointGenerator(JointLimits(VEL, ACC, ACC * 10, LOW, HIGH), DT)
    with pytest.raises(RuntimeError):
        gen.step()


def test_converges_to_target_velocity_and_holds_it():
    gen = make()
    target = np.array([0.5, -0.3, 0.2, 0.0, 1.0, -1.0, 0.0])
    states = run(gen, 200, target)  # 1 s: well past the ~0.3 s transition
    np.testing.assert_allclose(states[-1].velocity, target, atol=1e-9)
    np.testing.assert_allclose(states[-1].acceleration, 0.0, atol=1e-9)
    # Constant velocity afterwards: the position keeps integrating it.
    later = run(gen, 20)
    np.testing.assert_allclose(later[-1].position - states[-1].position, target * 20 * DT, atol=1e-9)


@pytest.mark.parametrize('seed', range(5))
def test_random_targets_respect_velocity_acceleration_and_jerk_limits(seed):
    """Targets jump every ~17 ms like the teleop's IK output; the setpoint must stay smooth."""
    rng = np.random.default_rng(seed)
    gen = make()
    states = []
    for _ in range(60):
        states += run(gen, rng.integers(2, 6), rng.uniform(-1.5, 1.5, 7) * VEL)
    v = np.array([s.velocity for s in states])
    a = np.array([s.acceleration for s in states])
    q = np.array([s.position for s in states])
    assert np.all(np.abs(v) <= VEL + TOL)
    assert np.all(np.abs(a) <= ACC + TOL)
    assert np.all(np.abs(np.diff(a, axis=0)) / DT <= ACC * 10 * (1 + 1e-6) + TOL)
    # Position is the integral of velocity: no jumps (the staircase this replaces).
    assert np.all(np.abs(np.diff(q, axis=0)) <= VEL * DT + TOL)


def test_target_velocity_is_clipped_to_the_limits():
    gen = make()
    gen.reset((LOW + HIGH) / 2)  # mid-range, so 0.7 s at full speed stays clear of the position limits
    states = run(gen, 140, 10 * VEL)
    np.testing.assert_allclose(states[-1].velocity, VEL, atol=1e-9)


def test_stops_at_a_position_limit():
    gen = make()
    gen.reset(np.where(np.arange(7) == 3, LOW[3] + 0.01, Q0))  # e1 just above its lower limit
    target = np.zeros(7)
    target[3] = -1.0
    states = run(gen, 100, target)
    assert states[-1].position[3] == pytest.approx(LOW[3])
    assert states[-1].velocity[3] == 0.0
    assert all(s.position[3] >= LOW[3] for s in states)


def test_hold_within_clamps_a_setpoint_running_away_from_the_arm():
    gen = make()
    run(gen, 200, np.full(7, 0.5))  # setpoint ~0.4 rad ahead of an arm that did not move
    assert gen.hold_within(Q0, max_lead=0.2)
    np.testing.assert_allclose(gen.state.position, Q0 + 0.2)
    assert not gen.state.velocity.any()
    assert not gen.hold_within(Q0, max_lead=0.2)  # already inside: nothing to do


def test_limits_validation():
    with pytest.raises(ValueError):
        JointLimits(VEL, ACC, -ACC, LOW, HIGH)
    with pytest.raises(ValueError):
        JointLimits(VEL, ACC, ACC, HIGH, LOW)
    with pytest.raises(ValueError):
        JointLimits(VEL[:6], ACC, ACC, LOW, HIGH)

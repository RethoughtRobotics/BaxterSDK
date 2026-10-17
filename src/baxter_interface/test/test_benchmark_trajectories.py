"""ISO 9283-style trajectories: profiles respect speed/accel limits, shapes and events are right."""

import numpy as np
import pytest

from baxter_interface.baxter_teleop.benchmark_trajectories import (
    DT,
    circle_test,
    iso_protocol,
    linear_test,
    pose_test,
    trapezoid,
)


def derivatives(x):
    v = np.diff(np.vstack([np.zeros(x.shape[1]), x]), axis=0) / DT
    a = np.diff(v, axis=0) / DT
    return v, a


@pytest.mark.parametrize('distance,speed', [(0.3, 0.15), (0.3, 0.4), (0.01, 0.15)])
def test_trapezoid_reaches_the_distance_within_limits(distance, speed):
    s = trapezoid(distance, speed, 0.5)
    v = np.diff(np.r_[0.0, s]) / DT
    assert s[-1] == distance and np.all(np.diff(s) >= -1e-12)
    assert v.max() <= speed + 1e-9
    assert np.abs(np.diff(v)).max() / DT <= 0.5 * 1.05 + 1e-6  # accel limit (5% for sampling)


def test_trapezoid_cruises_at_speed_when_long_enough():
    v = np.diff(np.r_[0.0, trapezoid(0.3, 0.15, 0.5)]) / DT
    assert np.isclose(np.median(v[len(v) // 3 : 2 * len(v) // 3]), 0.15)


@pytest.mark.parametrize('speed', [0.05, 0.15, 0.40])
def test_iso_protocol_is_planar_continuous_and_limited(speed):
    offsets, phases, events = iso_protocol(speed=speed)
    assert [p[0] for p in phases] == ['pose', 'linear', 'circle']
    np.testing.assert_allclose(offsets[0], 0) and np.testing.assert_allclose(offsets[-1], 0, atol=1e-9)
    assert np.abs(offsets[:, 2]).max() == 0.0  # planar
    v, a = derivatives(offsets)
    assert np.linalg.norm(v, axis=1).max() <= speed * 1.01 + 1e-9
    # Per-axis acceleration within the limit (circle: centripetal v^2/r adds at speed)
    assert np.abs(a).max() <= max(0.5, speed**2 / 0.12) * 1.1 + 1e-6


def test_pose_test_visits_the_square_and_holds():
    offsets, phases, events = pose_test(0.15, 0.5, side=-1, half=0.1, cycles=2, dwell=1.5)
    assert [e['target'] for e in events] == ['P1', 'P2', 'P3', 'P4', 'P5'] * 2
    for e in events:
        i0, i1 = int(round(e['t_arrive'] / DT)), int(round(e['t_leave'] / DT))
        np.testing.assert_allclose(offsets[i0:i1], np.tile(e['point'], (i1 - i0, 1)))
        assert np.isclose(e['t_leave'] - e['t_arrive'], 1.5)
    pts = np.array([e['point'] for e in events[:5]])
    np.testing.assert_allclose(pts[0], [0, -0.1, 0])  # centre toward the right arm's outer side
    assert np.allclose(np.abs(pts[1:] - pts[0])[:, :2], 0.1)


def test_linear_test_reverses_at_rest_at_both_ends():
    offsets, _, events = linear_test(0.15, 0.5, side=-1, length=0.3, reps=3)
    reversals = [e for e in events if e['kind'] == 'reversal']
    assert len(reversals) == 6
    v, _ = derivatives(offsets)
    for e in reversals:
        i = int(round(e['t'] / DT))
        np.testing.assert_allclose(offsets[i], e['point'], atol=1e-12)
        assert np.linalg.norm(v[i]) < 0.5 * DT * 1.01  # at rest (one accel step)
    assert np.isclose(offsets[:, 1].min(), -0.3)


def test_circle_test_has_the_radius_and_starts_at_rest():
    offsets, _, _ = circle_test(0.15, 0.5, side=-1, radius=0.12, loops=2)
    np.testing.assert_allclose(np.linalg.norm(offsets[1:] - [0, -0.12, 0], axis=1), 0.12, atol=1e-9)
    v, _ = derivatives(offsets)
    assert np.linalg.norm(v[0]) < 0.5 * DT * 1.01
    np.testing.assert_allclose(offsets[-1], 0, atol=1e-9)

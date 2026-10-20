"""JointErrorModel: a FIR from commanded velocity to tracking error, recovered from data it generated."""

import numpy as np

from baxter_teleop.joint_error_model import JointErrorModel, lagged

DT = 0.01


def first_order_lag(tau, taps):
    """Impulse response of a joint whose angle follows the command through a first-order lag tau:
    its tracking error is the command's velocity filtered by exp(-t / tau), summing to tau."""
    t = DT * np.arange(taps)
    h = np.exp(-t / tau)
    return tau * h / h.sum()


def segment(h, c, n=2000, seed=0, noise=0.0):
    """(v, e) for one joint driven by a random smooth commanded velocity."""
    rng = np.random.default_rng(seed)
    v = np.convolve(rng.normal(0.0, 1.0, n), np.ones(25) / 25, mode='same')
    e = c + np.convolve(v, h)[:n] + rng.normal(0.0, noise, n)
    return v[:, None], e[:, None]


def test_lagged_rows_hold_the_recent_past():
    rows = lagged(np.arange(5.0), 3)
    np.testing.assert_array_equal(rows, [[2, 1, 0], [3, 2, 1], [4, 3, 2]])


def test_recovers_a_first_order_lag():
    h = first_order_lag(0.05, 30)
    model = JointErrorModel(taps=30, dt=DT, ridge=0.0).fit([segment(h, 0.002)])
    np.testing.assert_allclose(model.h[0], h, atol=1e-9)
    np.testing.assert_allclose(model.c[0], 0.002, atol=1e-9)
    np.testing.assert_allclose(model.lag()[0], 0.05, atol=1e-9)


def test_explains_held_out_motion_and_not_noise():
    h = first_order_lag(0.05, 30)
    model = JointErrorModel(taps=30, dt=DT).fit([segment(h, 0.0, seed=0, noise=1e-4)])
    rms, left, explained = model.explained([segment(h, 0.0, seed=1, noise=1e-4)])[0]
    assert explained > 0.9
    assert left < 0.3 * rms


def test_a_wrong_offset_counts_against_the_model():
    """Held-out data with a different constant error: explained must drop, not hide it in the variance."""
    h = first_order_lag(0.05, 30)
    model = JointErrorModel(taps=30, dt=DT, ridge=0.0).fit([segment(h, 0.0)])
    rms, left, explained = model.explained([segment(h, 0.01, seed=1)])[0]
    assert left > 0.9 * 0.01
    assert explained < 0.5


def test_rows_never_cross_segments():
    """Two segments whose concatenation would fake a velocity history: the fit must not see it."""
    h = first_order_lag(0.05, 30)
    a, b = segment(h, 0.0, n=500, seed=0), segment(h, 0.0, n=500, seed=1)
    model = JointErrorModel(taps=30, dt=DT, ridge=0.0)
    X_rows, _ = model._rows([a, b], 0)
    assert len(X_rows) == 2 * (500 - 29)


def test_predict_pads_the_start_and_round_trips(tmp_path):
    h = first_order_lag(0.03, 10)
    model = JointErrorModel(taps=10, dt=DT, ridge=0.0).fit([segment(h, 0.0)])
    v, e = segment(h, 0.0, n=200, seed=3)
    predicted = model.predict(v)
    assert np.isnan(predicted[:9]).all()
    np.testing.assert_allclose(predicted[9:], e[9:], atol=1e-9)
    model.save(tmp_path / 'model.npz')
    np.testing.assert_allclose(JointErrorModel.load(tmp_path / 'model.npz').predict(v)[9:], e[9:], atol=1e-9)

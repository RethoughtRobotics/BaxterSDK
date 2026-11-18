"""The command line: what --steer reaches, and the combinations it refuses."""

import os
import types

import numpy as np
import pytest

from baxter_teleop import cli

CONFIG = os.path.join(os.path.dirname(__file__), '..', 'config')


class FakeTeleop:
    VR_SPEEDS = (0.5, 1.0)
    spun = None

    def __init__(self, **kwargs):
        self.node = types.SimpleNamespace(get_logger=lambda: types.SimpleNamespace(info=print))
        self.tcp_offset = np.zeros(3)
        self.ik = self.kinematics = None
        self.telemetry = types.SimpleNamespace(config=lambda config: None)
        self.linear_speed, self.angular_speed = kwargs['linear_speed'], kwargs['angular_speed']

    def spin_vr(self, **kwargs):
        FakeTeleop.spun = kwargs

    def close(self):
        pass


@pytest.fixture
def run(monkeypatch):
    monkeypatch.setattr(cli, 'Teleop', FakeTeleop)
    monkeypatch.setattr(cli, 'VRTarget', lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, 'PlanTarget', lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, 'config_file', lambda name: os.path.join(CONFIG, os.path.basename(name)))
    monkeypatch.setattr(cli.rclpy, 'init', lambda args: None)
    monkeypatch.setattr(cli.rclpy, 'shutdown', lambda: None)

    def main(*argv):
        FakeTeleop.spun = None
        cli.main(['--arm', 'left', '--vr-config', 'vr_axes_policy.yaml', *argv])
        return FakeTeleop.spun

    return main


TARGET = ('--input', 'vr', '--vr-topic', '/policy/left/target')


@pytest.mark.parametrize('chain', [TARGET, ('--input', 'plan')])
def test_steer_reaches_the_spin_of_both_policy_chains(run, chain):
    assert run(*chain, '--steer')['steer'] is True
    assert run(*chain)['steer'] is False


@pytest.mark.parametrize(
    'argv',
    [
        ('--input', 'vr', '--steer'),  # a person's own controller, no policy to steer
        ('--input', 'trajectory', '--vr-topic', '/policy/left/target', '--steer'),
        (*TARGET, '--output', 'velocity', '--steer'),  # no streamed setpoint to add to
    ],
)
def test_steer_is_refused_without_a_policy_stream(run, argv):
    with pytest.raises(SystemExit):
        run(*argv)

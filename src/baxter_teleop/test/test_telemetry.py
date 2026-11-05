"""Teleop telemetry: a correction's flag and offset, on the nudge's topics or the kinesthetic ones."""

import types

import numpy as np
import pytest

from baxter_teleop.telemetry import Telemetry


class Node:
    def __init__(self):
        self.sent = {}

    def create_publisher(self, _type, topic, _qos):
        return types.SimpleNamespace(publish=lambda msg: self.sent.setdefault(topic, []).append(msg))

    def get_clock(self):
        return types.SimpleNamespace(now=lambda: types.SimpleNamespace(to_msg=lambda: None))


@pytest.mark.parametrize(
    ('call', 'flag', 'offset'),
    [
        ('nudge', '/teleop/left/nudging', '/teleop/left/correction_offset'),
        ('kinesthetic', '/teleop/left/kinesthetic/active', '/teleop/left/kinesthetic/offset'),
    ],
)
def test_a_correction_goes_out_on_its_own_topics(call, flag, offset):
    node = Node()
    getattr(Telemetry(node, 'left', []), call)(True, np.arange(6.0))
    assert [m.data for m in node.sent[flag]] == [True]
    (msg,) = node.sent[offset]
    t = msg.twist
    assert [t.linear.x, t.linear.y, t.linear.z, t.angular.x, t.angular.y, t.angular.z] == list(range(6))
    assert msg.header.frame_id == 'base'
    assert set(node.sent) == {flag, offset}  # nothing on the other correction's topics

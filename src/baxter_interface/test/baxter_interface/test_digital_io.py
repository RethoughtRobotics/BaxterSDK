"""Unit tests for the baxter_interface.digital_io module."""

import unittest
from unittest.mock import MagicMock, patch

from baxter_core_msgs.msg import DigitalIOState

from baxter_interface import digital_io
from baxter_interface.digital_io import DigitalIO


def _state(pressed, is_input_only=True):
    msg = DigitalIOState()
    msg.state = DigitalIOState.PRESSED if pressed else DigitalIOState.UNPRESSED
    msg.is_input_only = is_input_only
    return msg


def _make_node():
    """Returns (mock_node, trigger_callback) where trigger_callback(msg) fires _on_io_state."""
    node = MagicMock()
    captured = {}

    def capture_subscription(msg_type, topic, callback, qos, callback_group=None):
        captured['callback'] = callback
        return MagicMock()

    node.create_subscription.side_effect = capture_subscription
    return node, lambda msg: captured['callback'](msg)


class Recorder:
    """A bound method to connect: Signal holds plain functions weakly, so a lambda would vanish."""

    def __init__(self):
        self.calls = []

    def on_change(self, state):
        self.calls.append(state)


class TestDigitalIO(unittest.TestCase):
    def test_timeout_zero_does_not_wait(self):
        node, _ = _make_node()
        with patch.object(digital_io.baxter_dataflow, 'wait_for') as wait_for:
            io = DigitalIO('left_lower_cuff', node, timeout=0)
        wait_for.assert_not_called()
        self.assertIsNone(io.state)

    def test_default_waits_for_the_first_state(self):
        node, _ = _make_node()
        with patch.object(digital_io.baxter_dataflow, 'wait_for') as wait_for:
            DigitalIO('left_lower_cuff', node)
        self.assertEqual(wait_for.call_args.kwargs['timeout'], 2.0)

    def test_state_changed_fires_on_each_edge(self):
        node, trigger = _make_node()
        io = DigitalIO('left_upper_button', node, timeout=0)
        rec = Recorder()
        io.state_changed.connect(rec.on_change)
        for pressed in (False, True, True, False):
            trigger(_state(pressed))
        self.assertEqual(rec.calls, [True, False])
        self.assertFalse(io.state)

    def test_debounce_drops_a_press_soon_after_a_release(self):
        node, trigger = _make_node()
        clock = [0.0]
        with patch.object(digital_io.time, 'monotonic', lambda: clock[0]):
            io = DigitalIO('right_button_ok', node, timeout=0, debounce=0.2)
            rec = Recorder()
            io.state_changed.connect(rec.on_change)
            trigger(_state(False))
            trigger(_state(True))  # first press: counts
            clock[0] = 1.0
            trigger(_state(False))  # released at 1.0
            clock[0] = 1.05
            trigger(_state(True))  # bounce: dropped
            clock[0] = 1.1
            trigger(_state(False))
            clock[0] = 1.5
            trigger(_state(True))  # 0.4 s after the last release: counts
        self.assertEqual(rec.calls, [True, False, False, True])

    def test_output_publisher_created_on_first_command(self):
        node, trigger = _make_node()
        io = DigitalIO('left_inner_light', node, timeout=0)
        node.create_publisher.assert_not_called()
        trigger(_state(False, is_input_only=False))
        io.set_output(True, timeout=0)
        node.create_publisher.assert_called_once()
        self.assertTrue(node.create_publisher.return_value.publish.call_args.args[0].value)

    def test_input_only_refuses_output(self):
        node, trigger = _make_node()
        io = DigitalIO('left_lower_cuff', node, timeout=0)
        trigger(_state(False, is_input_only=True))
        with self.assertRaises(IOError):
            io.set_output(True, timeout=0)


if __name__ == '__main__':
    unittest.main()

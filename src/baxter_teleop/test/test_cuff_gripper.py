"""The cuff's gripper buttons under a policy: the policy runtime echoes every gripper command back as
its trigger, and a button must still open or close the gripper fully."""

import time
import types

from baxter_teleop.inputs.vr import VRController
from baxter_teleop.teleop import Teleop


class Gripper:
    """An electric gripper that records its last commanded opening (%)."""

    def __init__(self):
        self.opening = 100.0

    def type(self):
        return 'electric'

    def calibrated(self):
        return True

    def ready(self):
        return True

    def command_position(self, opening, block=False):
        self.opening = float(opening)

    def close(self):
        self.opening = 0.0

    def open(self):
        self.opening = 100.0


def teleop():
    t = object.__new__(Teleop)
    t.gripper, t.drive_gripper, t._gripper_target = Gripper(), True, 100.0
    t._cuff = types.SimpleNamespace(pressed=None)
    t._cuff.take_button = lambda: (t._cuff.pressed, setattr(t._cuff, 'pressed', None))[0]
    return t


def run(t, source, squeezed, cycles=50, was_squeezed=None):
    """`cycles` teleop cycles at 100 Hz, the runtime echoing the gripper's last command as the trigger."""
    was = squeezed if was_squeezed is None else was_squeezed
    for _ in range(cycles):
        source.trigger = (1.0 - t.gripper.opening / 100.0, time.monotonic())
        t._drive_gripper(source, squeezed, released=was and not squeezed)
        was = squeezed
        time.sleep(0.01)


def source():
    s = object.__new__(VRController)
    s.smoothing, s.min_step, s.min_period, s.stale = 0.05, 0.5, 0.02, 0.5
    s._filtered = s._filtered_at = s._sent = None
    s.trigger = None
    return s


def test_a_cuff_button_closes_and_opens_the_gripper_fully_against_the_echo():
    t, s = teleop(), source()
    run(t, s, squeezed=False, cycles=5)  # the policy drives, the gripper open
    t._cuff.pressed = 'close'
    run(t, s, squeezed=True, was_squeezed=False)
    assert t.gripper.opening == 0.0
    t._cuff.pressed = 'open'
    run(t, s, squeezed=True)
    assert t.gripper.opening == 100.0


def test_on_release_the_trigger_takes_over_from_the_cuffs_last_command():
    t, s = teleop(), source()
    run(t, s, squeezed=False, cycles=5)
    t._cuff.pressed = 'close'
    run(t, s, squeezed=True, was_squeezed=False)
    run(t, s, squeezed=False, was_squeezed=True)
    assert t.gripper.opening < 1.0

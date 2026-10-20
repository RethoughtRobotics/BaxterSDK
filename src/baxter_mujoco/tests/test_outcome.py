"""Closed-loop scoring (datagen.outcome) on scripted series of the TCP, gripper and cup, without ROS."""

import numpy as np
import pytest

from baxter_mujoco import scenes
from baxter_mujoco.datagen.cup import CupTask
from baxter_mujoco.datagen.outcome import STALL, Attempt

RATE = 25.0
CUP, BOX = np.array([0.7, 0.2, scenes.TABLE_TOP]), np.array([0.7, -0.1, scenes.TABLE_TOP])
LAYOUT = [*CUP[:2], *BOX[:2], 0.0]
GRASP = np.array([0.0, 0.028, 0.043])  # TCP from the cup's base at the wall grasp
HOME = np.array([0.57, 0.22, 0.10])
UP = np.array([0.0, 0.0, 0.12])
IDENTITY = [0.0, 0.0, 0.0, 1.0]


def score(keys, timeout=40.0):
    """Run an Attempt over keyframes (seconds to get there, TCP, gripper opening, cup held) linearly
    interpolated; a cup let go falls to the container's floor or the table below it. Returns
    (outcome, the reason it stopped)."""
    attempt = Attempt(CupTask(), LAYOUT)
    tcp, grip, cup, t = HOME.copy(), 100.0, CUP.copy(), 0.0
    keys = [*keys, (timeout, None, None, None)]
    for seconds, goal, opening, attached in keys:
        start = tcp
        for k in range(1, int(seconds * RATE) + 1):
            if goal is not None:
                tcp = start + (goal - start) * k / int(seconds * RATE)
                grip = opening
            if attached:
                cup = tcp - GRASP
            elif cup[2] > scenes.TABLE_TOP + 0.006:
                inside = CupTask().success(LAYOUT, {'cup': [*cup[:2], 0.0]})
                cup = np.array([cup[0], cup[1], scenes.TABLE_TOP + (0.005 if inside else 0.0)])
            attempt.add(t, tcp, grip, cup, IDENTITY, BOX, IDENTITY)
            t += 1 / RATE
            if stop := attempt.done():
                return attempt.outcome(stalled=stop == 'stalled'), stop
            if t >= timeout:
                return attempt.outcome(), 'timeout'
    return attempt.outcome(), 'timeout'


def grasp_and_lift():
    at = CUP + GRASP
    return [(2.0, at + UP, 100.0, False), (1.5, at, 100.0, False), (1.0, at, 10.0, False), (1.5, at + UP, 10.0, True)]


def test_success_after_placing_in_the_container():
    over = BOX + GRASP + UP
    lowered = BOX + GRASP + [0, 0, 0.005]
    out, stop = score([
        *grasp_and_lift(),
        (2.0, over, 10.0, True),
        (1.5, lowered, 10.0, True),
        (0.5, lowered, 100.0, False),
        (2.0, over, 100.0, False),
    ])
    assert stop == 'success'
    assert out['success'] and out['phase'] == 'release' and out['failure'] is None
    assert out['time'] == pytest.approx(9.5, abs=0.1)  # the gripper opened at 9.5 s, the cup at rest


def test_missed_grasp():
    at = CUP + GRASP + [0.0, 0.015, 0.0]  # just outside the wall: closes on nothing
    out, _ = score(
        [
            (2.0, at + UP, 100.0, False),
            (1.5, at, 100.0, False),
            (1.0, at, 0.0, False),
            (1.5, at + UP, 0.0, False),
            (3.0, HOME, 0.0, False),
            (3.0, at + UP, 0.0, False),
        ]
        * 3
    )
    assert not out['success'] and out['failure'] == 'missed grasp' and out['phase'] == 'reach'


def wander(a, grip):
    """Keep moving (no stall) between a and home, the gripper as it is, to the timeout."""
    return [(3.0, HOME, grip, False), (3.0, a, grip, False)] * 6


def test_dropped_in_transit():
    halfway = (CUP + BOX) / 2 + GRASP + UP
    out, stop = score([
        *grasp_and_lift(),
        (1.0, halfway, 10.0, True),
        (0.5, halfway, 10.0, False),
        *wander(halfway, 10.0),
    ])
    assert stop == 'timeout' and out['failure'] == 'dropped in transit' and out['phase'] == 'grasp'


def test_released_outside_before_the_container():
    beside = BOX + GRASP + UP + [0.2, 0.0, 0.0]
    out, _ = score([*grasp_and_lift(), (2.0, beside, 10.0, True), (0.5, beside, 100.0, False), *wander(beside, 100.0)])
    assert out['failure'] == 'released outside' and out['phase'] == 'grasp'


def test_released_after_the_transfer_but_outside():
    over, beside = BOX + GRASP + UP, BOX + GRASP + UP + [0.2, 0.0, 0.0]
    out, _ = score([
        *grasp_and_lift(),
        (2.0, over, 10.0, True),
        (1.0, beside, 10.0, True),
        (0.5, beside, 100.0, False),
        *wander(beside, 100.0),
    ])
    assert out['failure'] == 'released outside' and out['phase'] == 'release'


def test_held_over_the_container_until_the_timeout():
    over = BOX + GRASP + UP
    out, stop = score(
        [*grasp_and_lift(), (2.0, over, 10.0, True)]
        + [(1.0, over + [0.03, 0, 0], 10.0, True), (1.0, over, 10.0, True)] * 20
    )
    assert stop == 'timeout' and out['failure'] == 'timeout' and out['phase'] == 'transfer'


def test_stalled_stops_the_attempt():
    out, stop = score([(2.0, CUP + GRASP + UP, 100.0, False), (30.0, CUP + GRASP + UP, 100.0, False)])
    assert stop == 'stalled' and out['failure'] == 'stalled' and out['phase'] is None


def test_stall_needs_the_whole_window():
    attempt = Attempt(CupTask(), LAYOUT)
    for k in range(int((STALL - 0.5) * RATE)):
        attempt.add(k / RATE, HOME, 100.0, CUP, IDENTITY, BOX, IDENTITY)
    assert attempt.done() is None
    for k in range(int((STALL - 0.5) * RATE), int((STALL + 0.5) * RATE)):
        attempt.add(k / RATE, HOME, 100.0, CUP, IDENTITY, BOX, IDENTITY)
    assert attempt.done() == 'stalled'


def test_timeout_while_moving_away_from_the_cup():
    a, b = HOME, HOME + [0.0, -0.2, 0.0]
    out, stop = score([(2.0, b, 100.0, False), (2.0, a, 100.0, False)] * 12)
    assert stop == 'timeout' and out['failure'] == 'timeout' and out['phase'] is None


def test_container_pushed_away_is_scored_where_it_is():
    attempt = Attempt(CupTask(), LAYOUT)
    moved = BOX + [0.0, -0.2, 0.0]
    for k in range(int(RATE)):
        attempt.add(k / RATE, HOME, 100.0, BOX + [0, 0, 0.005], IDENTITY, moved, IDENTITY)
    assert attempt.done() is None

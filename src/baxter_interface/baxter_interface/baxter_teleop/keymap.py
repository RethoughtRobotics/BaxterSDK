"""Keyboard mapping utilities for EE teleop."""

import numpy as np


def twist_from_key(key, linear_speed, angular_speed):
    """Convert a keyboard key to a 6D EE twist command."""
    v = np.zeros(6, dtype=np.float32)
    k = key.lower()

    # This map is intentionally explicit because the key layout is part of the
    # user interface contract for manual teleoperation.
    if k == 'w':
        v[0] = linear_speed
    elif k == 's':
        v[0] = -linear_speed
    elif k == 'a':
        v[1] = linear_speed
    elif k == 'd':
        v[1] = -linear_speed
    elif k == 'r':
        v[2] = linear_speed
    elif k == 'f':
        v[2] = -linear_speed
    elif k == 'x':
        v[3] = angular_speed
    elif k == 'v':
        v[3] = -angular_speed
    elif k == 'z':
        v[4] = angular_speed
    elif k == 'c':
        v[4] = -angular_speed
    elif k == 'q':
        v[5] = angular_speed
    elif k == 'e':
        v[5] = -angular_speed
    return v

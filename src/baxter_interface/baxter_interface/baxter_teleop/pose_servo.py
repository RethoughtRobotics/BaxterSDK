"""The one control law from a goal pose to a TCP twist, shared by every input (VR, keyboard, policy)."""

import numpy as np


def servo_twist(goal, tcp_position, tcp_rotation, gain, max_linear, max_angular, scale=None, feedforward=None):
    """Base-frame TCP twist [v, w] that closes the gap to goal = (position, scipy Rotation).

    twist = feedforward + gain * error, with the error scaled per TCP axis by `scale` (6,), then
    each part capped to max_linear (m/s) / max_angular (rad/s) keeping its direction. Because
    the error is measured every cycle, lag, blocked motion and IK error are pulled back instead
    of accumulating; the feedforward carries a moving goal's own velocity, so tracking it needs
    no standing error.
    """
    v = gain * (goal[0] - tcp_position)
    w = gain * (goal[1] * tcp_rotation.inv()).as_rotvec()
    if scale is not None:  # per TCP axis
        R = tcp_rotation.as_matrix()
        v = R @ (scale[:3] * (R.T @ v))
        w = R @ (scale[3:] * (R.T @ w))
    if feedforward is not None:
        v, w = v + feedforward[:3], w + feedforward[3:]
    v *= min(1.0, max_linear / max(np.linalg.norm(v), 1e-9))
    w *= min(1.0, max_angular / max(np.linalg.norm(w), 1e-9))
    return np.concatenate([v, w]).astype(np.float32)

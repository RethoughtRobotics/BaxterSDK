"""Score a closed-loop cup attempt from what the sim reports: the TCP, the gripper and the objects.

Fed one sample at a time (Attempt.add), as the evaluator watches the sim, or a recorded series. Nothing
here talks to ROS. Per attempt:

  success   the cup at rest in the container (CupTask.success), the gripper open
  phase     the furthest reached: reach (the TCP at the cup), grasp (the cup held off the table),
            transfer (held over the container), release (let go after the transfer)
  failure   missed grasp (the gripper closed but never held the cup), dropped in transit (the cup
            left the closed gripper), released outside (the gripper let go, the cup is not in), stalled
            (STALL s without progress), timeout

The cup is held once it is lifted (LIFT) next to the TCP (HOLD) with the gripper not open, and stays
held while it is next to the TCP and the gripper is not open, so lowering it into the container
does not lose it. A hold lost with the gripper opening (within RELEASE s) is a release, else a drop.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from baxter_mujoco import scenes

PHASES = ('reach', 'grasp', 'transfer', 'release')
FAILURES = ('missed grasp', 'dropped in transit', 'released outside', 'stalled', 'timeout')
REACH = 0.05  # m, TCP to the cup's centre: at the cup (0.028 at the wall grasp)
HOLD = 0.06  # m, TCP to the cup's centre while held
LIFT = 0.02  # m, the cup's base above where it stood
OPEN = 50.0  # gripper opening (0 closed .. 100 open) above which it is open
CLOSED = 30.0  # below which it has closed (on the cup's wall about 10)
RELEASE = 0.5  # s after a lost hold within which the gripper opening makes it a release
STALL = 5.0  # s without progress
STILL = 0.01  # m: TCP and cup moved less than this
GRIP_STILL = 5.0  # gripper opening changed less than this
REST = 0.5  # s the cup stays still in the container before it counts


def cup_centre(position, quat):
    """The cup's centre from its body pose (origin at its base)."""
    return np.asarray(position) + Rotation.from_quat(quat).apply([0.0, 0.0, scenes.CUP_HEIGHT / 2])


def _yaw(quat):
    return float(Rotation.from_quat(quat).as_euler('xyz')[2])


class Attempt:
    """One attempt's samples and per-sample state; `layout` is the task's ([cup x y, container x y yaw])."""

    def __init__(self, task, layout):
        self.task, self.layout = task, list(layout)
        self.t, self.tcp, self.grip, self.cup = [], [], [], []
        self.near, self.held, self.over = [], [], []
        self.rest_z = None
        self.success_time = None

    def add(self, t, tcp, grip, cup, cup_quat, container, container_quat):
        """A sample at t (s from engage): TCP position, gripper opening, cup and container poses."""
        tcp, cup = np.asarray(tcp, float), np.asarray(cup, float)
        if self.rest_z is None:
            self.rest_z = cup[2]
        distance = np.linalg.norm(tcp - cup_centre(cup, cup_quat))
        not_open = grip < OPEN
        lifted = cup[2] > self.rest_z + LIFT
        held = distance < HOLD and not_open and (lifted or bool(self.held and self.held[-1]))
        box = [*self.layout[:2], container[0], container[1], _yaw(container_quat)]
        self.t.append(float(t))
        self.tcp.append(tcp)
        self.grip.append(float(grip))
        self.cup.append(cup)
        self.near.append(distance < REACH)
        self.held.append(held)
        self.over.append(self.task.success(box, {'cup': cup}))
        if self.success_time is None:
            self.success_time = self._rested()

    def _rested(self):
        """The time the cup came to rest in the container, once it has stayed there REST s; else None."""
        t = np.array(self.t)
        window = np.flatnonzero(t >= t[-1] - REST)
        if t[-1] - t[0] < REST or window[0] == 0:
            return None
        idx = range(window[0] - 1, len(t))  # the sample before the window spans it fully
        if not all(self.over[i] and not self.held[i] and self.grip[i] > OPEN for i in idx):
            return None
        cup = np.array(self.cup)[list(idx)]
        if np.max(np.linalg.norm(cup - cup[-1], axis=1)) > STILL:
            return None
        return t[window[0] - 1]

    def stalled(self):
        """Nothing moved for the last STALL s: the TCP, the cup and the gripper."""
        t = np.array(self.t)
        if not len(t) or t[-1] - t[0] < STALL:
            return False
        idx = np.flatnonzero(t >= t[-1] - STALL)
        idx = np.r_[idx[0] - 1, idx] if idx[0] > 0 else idx
        tcp, cup, grip = np.array(self.tcp)[idx], np.array(self.cup)[idx], np.array(self.grip)[idx]
        return bool(
            np.max(np.linalg.norm(tcp - tcp[-1], axis=1)) < STILL
            and np.max(np.linalg.norm(cup - cup[-1], axis=1)) < STILL
            and np.ptp(grip) < GRIP_STILL
        )

    def done(self):
        """'success', 'stalled' or None (go on)."""
        if self.success_time is not None:
            return 'success'
        return 'stalled' if self.stalled() else None

    def losses(self):
        """(time, 'release' or 'drop') of each hold lost."""
        t, grip, out = np.array(self.t), np.array(self.grip), []
        for i in range(1, len(t)):
            if self.held[i - 1] and not self.held[i]:
                after = (t >= t[i]) & (t <= t[i] + RELEASE)
                out.append((t[i], 'release' if np.any(grip[after] > OPEN) else 'drop'))
        return out

    def outcome(self, stalled=False):
        """{success, time, phase, failure}; `stalled`: the attempt was stopped for a stall."""
        t, held = np.array(self.t), np.array(self.held, bool)
        transfer = np.flatnonzero(held & np.array(self.over, bool))
        losses = self.losses()
        phase = None
        if any(self.near):
            phase = 'reach'
        if held.any():
            phase = 'grasp'
        if len(transfer):
            phase = 'transfer'
            if any(k == 'release' and when > t[transfer[0]] for when, k in losses):
                phase = 'release'
        success = self.success_time is not None
        if success:
            return {'success': True, 'time': round(self.success_time - t[0], 2), 'phase': 'release', 'failure': None}
        if held.any():
            if held[-1] or not losses:
                failure = 'stalled' if stalled else 'timeout'
            else:
                failure = 'dropped in transit' if losses[-1][1] == 'drop' else 'released outside'
        elif self.grip and min(self.grip) < CLOSED:
            failure = 'missed grasp'
        else:
            failure = 'stalled' if stalled else 'timeout'
        return {'success': False, 'time': None, 'phase': phase, 'failure': failure}

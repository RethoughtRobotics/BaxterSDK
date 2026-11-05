"""Steer a playing plan by pushing on the held cuff: the arm stays under position control and the
policy keeps moving it, instead of the zero-g a squeeze gives on its own.

Baxter's joints are series elastic, so the torque the spring carries is measured after the gearbox:
    I qdd = (actual_effort - gravity_model_effort) + tau_ext
A held arm has qdd = 0, so a hand's torque shows as tau_ext = -(actual - gravity). Two recordings agree
(reports/2026-11-05-cuff-corrections): actual follows the gravity model at r 0.96 to 0.99 on s1, e0 and
e1, so both are the torque that holds the arm up, and in a bag moved by hand alone actual - gravity
falls as the acceleration rises on all seven joints. diagnostics/cuff_probe checks it on the robot. The
arm's own acceleration enters as a virtual mass, not as positive feedback. The torque is filtered at 5 Hz (hand motion stays below 2 Hz; Tuma et al. 2026) and
tared on the 0.2 s before the squeeze, which removes the model's offset at this pose.

Admittance in joint space, with a leak:
    offset' = C dz(tau_ext) - offset / decay,   C = diag(1 / damping)
Joint space, because the TCP mobility J C J^T stays bounded through singularities where inverting J
does not, and the power tau_ext' C tau_ext the hand puts in is never negative. No IK runs: each 200 Hz
setpoint of the replay stream is the plan's plus this offset.

The leak is not optional. A relative policy anchors each plan at the measured TCP, so a plan already
holds the offset of its observation. An offset kept on top of it is counted again by every new plan:
the arm speeds up, and goes on after the hand lets go. The merged plan sees the arm some L ago (the
mean age of the merged chunks, about 0.72 s), which gives x(t) = x(t - L) + offset, so with a steady
input the arm moves at the admittance's velocity when decay = L, and stops when the hand does. With an
absolute policy nothing absorbs the offset: the leak is then a spring back to the plan, stiffness
damping / decay.

The squeeze only selects this mode: suppress_cuff_interaction is published while a plan plays, so
without a plan a squeeze still gives zero-g. Should the suppression not hold, the firmware frees the
arm and it leaves its setpoints; `live` then gives up for the session, and the teleop's zero-g handling
takes over.
"""

import numpy as np
from baxter_core_msgs.msg import SEAJointState
from std_msgs.msg import Empty

from ..settings import STEER, joint_names

SUPPRESS_PERIOD = 0.1  # s: Baxter's suppress topics need more than 5 Hz (Rethink API reference)
TARE_TIME = 0.2  # s


def deadzone(x, width):
    """x moved toward 0 by `width`, 0 inside it: no step in the output at the edge."""
    return np.sign(x) * np.maximum(np.abs(x) - width, 0.0)


class Admittance:
    """The joint offset from the torque stream, without ROS. Effort is actual minus gravity model."""

    def __init__(self, damping, deadband, decay, cutoff, max_rate, max_offset):
        self.damping, self.deadband = np.asarray(damping, float), np.asarray(deadband, float)
        self.decay, self.cutoff = float(decay), float(cutoff)
        self.max_rate, self.max_offset = float(max_rate), float(max_offset)
        self.reset()

    def reset(self):
        self.offset, self.rate = np.zeros(7), np.zeros(7)
        self.effort, self.tare, self._at = None, None, None

    def measure(self, effort, now, held):
        """One torque sample, at time `now`. The tare follows until the cuff is held."""
        effort = np.asarray(effort, float)
        dt = 0.0 if self._at is None else max(now - self._at, 0.0)
        self._at = now
        if self.effort is None:
            self.effort, self.tare = effort, effort
            return
        self.effort = self.effort + (effort - self.effort) * dt / (dt + 1.0 / (2 * np.pi * self.cutoff))
        if not held:
            self.tare = self.tare + (self.effort - self.tare) * dt / (dt + TARE_TIME)

    def step(self, dt, held):
        """(offset, its rate) one period on. Released, the offset only decays."""
        drive = np.zeros(7)
        if held and self.effort is not None:
            drive = np.clip(
                deadzone(self.tare - self.effort, self.deadband) / self.damping, -self.max_rate, self.max_rate
            )
        self.rate = drive - self.offset / self.decay
        offset = np.clip(self.offset + self.rate * dt, -self.max_offset, self.max_offset)
        self.rate, self.offset = (offset - self.offset) / dt, offset
        return self.offset, self.rate


class CuffSteer:
    """What the replay stream adds to each setpoint: called with its period, gives (offset, rate)."""

    def __init__(self, node, arm, cuff, stream):
        self.cuff, self.stream, self.node = cuff, stream, node
        self.admittance = Admittance(**{
            k: STEER[k] for k in ('damping', 'deadband', 'decay', 'cutoff', 'max_rate', 'max_offset')
        })
        self.max_error = float(STEER['max_tracking_error'])
        self.tripped = False
        self._names = joint_names(arm)
        self._now = lambda: node.get_clock().now().nanoseconds * 1e-9
        ns = f'/robot/limb/{arm}/'
        self._suppress = node.create_publisher(Empty, ns + 'suppress_cuff_interaction', 1)
        node.create_subscription(SEAJointState, ns + 'gravity_compensation_torques', self._on_effort, 10)
        node.create_timer(SUPPRESS_PERIOD, self._keep_suppressed)

    @property
    def active(self):
        """A squeeze steers now, rather than giving zero-g: a plan plays and the suppression is sent."""
        return self.stream.active and not self.tripped

    @property
    def held(self):
        return self.active and self.cuff.squeezed

    def live(self, q_measured):
        """`active`, after checking the arm still follows its setpoints: a freed arm does not."""
        position = self.stream.position
        if self.held and position is not None and np.max(np.abs(position - q_measured)) > self.max_error:
            self.tripped = True
            self.node.get_logger().error('Cuff steering off: the arm left its setpoints, cuff suppression did not hold')
        return self.active

    def reset(self):
        self.admittance.offset[:] = 0.0

    def __call__(self, dt):
        return self.admittance.step(dt, self.held)

    def _keep_suppressed(self):
        if self.active:
            self._suppress.publish(Empty())

    def _on_effort(self, msg):
        i = [msg.name.index(name) for name in self._names]
        effort = np.array(msg.actual_effort)[i] - np.array(msg.gravity_model_effort)[i]
        self.admittance.measure(effort, self._now(), self.held)

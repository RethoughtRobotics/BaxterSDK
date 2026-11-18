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
mean age of the merged chunks, about 0.56 s), which gives x(t) = x(t - L) + offset, so with a steady
input the arm moves at the admittance's velocity when decay = L, and stops when the hand does. With an
absolute policy nothing absorbs the offset: the leak is then a spring back to the plan, stiffness
damping / decay.

The squeeze only selects this mode: suppress_cuff_interaction is published while a plan plays, so
without a plan a squeeze still gives zero-g. Should the suppression not hold, the firmware frees the
arm and it leaves its setpoints; `live` then gives up for the session, and the teleop's zero-g handling
takes over.

Baxter's contact safety fires on a hard push (e0 at about 18 N m) and stops the arm following for about
2 s. Steering stands down for that time and the squeeze gives zero-g; it comes back on the next squeeze.

The node's parameters steer_softness_arm (s0 to e1) and steer_softness_wrist divide `damping` live, so a
softness of 2 steers twice as fast for the same push. Sliders on them, with the teleop running:
    ros2 run rqt_reconfigure rqt_reconfigure
Each change is logged, and /teleop/{arm}/steer_state records the admittance: position the total offset,
velocity the admittance's rate, effort the torque of the hand.

Softness moves only the small part of what the hand feels: Baxter's position hold is a spring of about
120 N m/rad on s0 to e1 and 17 to 66 on the wrist, and it cannot be changed in this mode. Following takes
that spring out instead. While held, the setpoint takes on a share `a` of the arm's distance from the plan:
    setpoint = plan + a (measured - plan),   so   stiffness = k (1 - a)
for steer_follow_arm (s0 to e1) and steer_follow_wrist, 0 the plain position hold. It feeds the measured
position back into the command, which is stable while a times the servo's gain stays below 1: the share
stops short of 1, the measurement is low-passed at `cutoff`, and no velocity feedforward goes with it.
Released, the share's offset decays like the admittance's. diagnostics/follow_probe tries it on a held pose.
"""

import numpy as np
from baxter_core_msgs.msg import CollisionDetectionState, SEAJointState
from rcl_interfaces.msg import FloatingPointRange, ParameterDescriptor, SetParametersResult
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty

from ..settings import STEER, joint_names

SUPPRESS_PERIOD = 0.1  # s: Baxter's suppress topics need more than 5 Hz (Rethink API reference)
TARE_TIME = 0.2  # s
SOFTNESS = {'steer_softness_arm': slice(0, 4), 'steer_softness_wrist': slice(4, 7)}  # of Admittance.damping
FOLLOW = {'steer_follow_arm': slice(0, 4), 'steer_follow_wrist': slice(4, 7)}  # share of the arm's offset
MAX_SOFTNESS = 2.0  # phase margin 46 degrees at a 2000 N/m grip; at 3 it is 9, and unstable at 3000 N/m


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


class Follow:
    """The setpoint's share of the arm's distance from the plan, see the module docstring."""

    def __init__(self, cutoff, decay, max_rate, max_offset):
        self.lag, self.decay = 1.0 / (2 * np.pi * float(cutoff)), float(decay)
        self.max_rate, self.max_offset = float(max_rate), float(max_offset)
        self.share, self.offset = np.zeros(7), np.zeros(7)

    def step(self, dt, q_plan, q_measured, held):
        """The offset one period on. Released, it only decays."""
        following = held and q_measured is not None
        target = self.share * (q_measured - q_plan) if following else np.zeros(7)
        change = (target - self.offset) * dt / (dt + (self.lag if following else self.decay))
        limit = self.max_rate * dt
        self.offset = np.clip(self.offset + np.clip(change, -limit, limit), -self.max_offset, self.max_offset)
        return self.offset


class CuffSteer:
    """What the replay stream adds to each setpoint: called with its period, gives (offset, rate)."""

    def __init__(self, node, arm, cuff, stream):
        self.cuff, self.stream, self.node = cuff, stream, node
        self.admittance = Admittance(**{
            k: STEER[k] for k in ('damping', 'deadband', 'decay', 'cutoff', 'max_rate', 'max_offset')
        })
        self.follow = Follow(STEER['cutoff'], STEER['decay'], STEER['follow_max_rate'], STEER['follow_max_offset'])
        self.follow.share[:] = float(STEER['follow'])  # the sliders' start
        self.q_measured = None
        self._added = None  # (the plan's joints, the joint offset added to them) of the last period
        self.max_error = float(STEER['max_tracking_error'])
        self.tripped, self.contact = False, False
        self._names = joint_names(arm)
        self._damping = self.admittance.damping.copy()
        self._now = lambda: node.get_clock().now().nanoseconds * 1e-9
        ns = f'/robot/limb/{arm}/'
        self._suppress = node.create_publisher(Empty, ns + 'suppress_cuff_interaction', 1)
        self._state = node.create_publisher(JointState, f'/teleop/{arm}/steer_state', 10)
        node.create_subscription(SEAJointState, ns + 'gravity_compensation_torques', self._on_effort, 10)
        node.create_subscription(CollisionDetectionState, ns + 'collision_detection_state', self._on_contact, 10)
        node.create_timer(SUPPRESS_PERIOD, self._keep_suppressed)
        slider = FloatingPointRange(from_value=1.0, to_value=MAX_SOFTNESS, step=0.0)
        for name in SOFTNESS:
            node.declare_parameter(name, 1.0, ParameterDescriptor(floating_point_range=[slider]))
        share = FloatingPointRange(from_value=0.0, to_value=float(STEER['max_follow']), step=0.0)
        for name in FOLLOW:
            node.declare_parameter(name, float(STEER['follow']), ParameterDescriptor(floating_point_range=[share]))
        node.add_on_set_parameters_callback(self._on_parameters)

    @property
    def active(self):
        """A squeeze steers now, rather than giving zero-g: a plan plays and the suppression is sent."""
        return self.stream.active and not self.tripped and not self.contact

    @property
    def held(self):
        return self.active and self.cuff.squeezed

    def live(self, q_measured):
        """`active`, after checking the arm still follows its setpoints: a freed arm does not."""
        position = self.stream.commanded
        if self.held and position is not None:
            error = np.abs(position - q_measured)
            if error.max() > self.max_error:
                self.tripped = True
                joint = int(error.argmax())
                self.node.get_logger().error(
                    f'Cuff steering off: {self._names[joint]} {error[joint]:.3f} rad from its setpoint with no '
                    f'contact, hand torque {self.hand_torque()[joint]:.1f} N m: little torque means the arm is free'
                )
        return self.active

    def reset(self):
        self.admittance.offset[:] = 0.0
        self.follow.offset = np.zeros(7)
        self._added = None

    def __call__(self, dt, q_plan):
        """(offset, rate) one period on. The rate is the admittance's: following sends no feedforward."""
        added, rate = self.admittance.step(dt, self.held)
        added = added + self.follow.step(dt, q_plan, self.q_measured, self.held)
        self._added = (np.array(q_plan, float), added)  # replaced whole: read from the teleop's thread
        return added, rate

    def tcp_offset(self, tcp):
        """The correction as a TCP offset (6,), in a VR nudge's format: the translation and rotation vector,
        base frame, from the plan's TCP, while steering adds anything; zeros otherwise. Held, to the arm's
        measured TCP: the command is only the share a of the way there, and a policy without the hand goes
        where it is commanded, so where the arm went is what it must learn, whatever a was. Released, to the
        commanded TCP: the hand is gone and the offset decays back to the plan (a nudge's handback), and the
        arm's own tracking error is not the person's. `tcp(q)` gives the TCP (position, rotation matrix) at
        joints q (kinematics.ArmKinematics.tcp)."""
        if self._added is None or not self._added[1].any():
            return np.zeros(6)
        q_plan, added = self._added
        q_to = self.q_measured if self.held and self.q_measured is not None else q_plan + added
        (p0, R0), (p1, R1) = tcp(q_plan), tcp(q_to)
        return np.r_[p1 - p0, Rotation.from_matrix(R1 @ R0.T).as_rotvec()]

    def hand_torque(self):
        a = self.admittance
        return np.zeros(7) if a.effort is None else a.tare - a.effort

    def _on_parameters(self, parameters):
        for parameter in parameters:
            if parameter.name in FOLLOW:
                self.follow.share[FOLLOW[parameter.name]] = parameter.value
                self.node.get_logger().info(f'{parameter.name} {parameter.value:.2f}')
            elif parameter.name in SOFTNESS:
                joints = SOFTNESS[parameter.name]
                damping = self.admittance.damping.copy()
                damping[joints] = self._damping[joints] / parameter.value
                self.admittance.damping = damping
                self.node.get_logger().info(f'{parameter.name} {parameter.value:.2f}: damping {damping}')
        return SetParametersResult(successful=True)

    def _on_contact(self, msg):
        contact = bool(msg.collision_state)
        if contact and not self.contact and self.held:
            self.node.get_logger().warning(
                f'Cuff steering paused: Baxter contact safety, hand torque {np.round(self.hand_torque(), 1)} N m'
            )
        self.contact = contact

    def _keep_suppressed(self):
        if self.active:
            self._suppress.publish(Empty())

    def _on_effort(self, msg):
        i = [msg.name.index(name) for name in self._names]
        self.q_measured = np.array(msg.actual_position)[i]
        effort = np.array(msg.actual_effort)[i] - np.array(msg.gravity_model_effort)[i]
        self.admittance.measure(effort, self._now(), self.held)
        if self.active:
            a = self.admittance
            self._state.publish(
                JointState(
                    header=msg.header,
                    name=self._names,
                    position=(a.offset + self.follow.offset).tolist(),
                    velocity=a.rate.tolist(),
                    effort=self.hand_torque().tolist(),
                )
            )

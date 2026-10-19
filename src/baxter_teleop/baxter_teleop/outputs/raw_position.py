"""Stream a jerk-limited joint setpoint to Baxter in RAW_POSITION_MODE, with optional inverse dynamics.

Why a separate stream: the teleop's control cycle runs at ~76 Hz with jitter, but Baxter's raw
position mode passes each setpoint straight to the joint controller boards, so the setpoint
must itself be smooth. RawPositionOutput owns a JointSetpointGenerator and steps it on its own
thread at a fixed rate; the control loop only updates the target velocity (command()).

With inverse dynamics, each setpoint is also sent as /robot/limb/{arm}/inverse_dynamics_command
(q, qdot, qddot), so Baxter's joint controllers add the model torque for the motion instead of
waiting for tracking error, as the SDK's position_w_id joint trajectory action server does
(baxter_interface changelog: "raw position mode", "inverse dynamics feedforward"). Measured on
the right arm: joint setpoint -> actual lag 175-245 ms (velocity or raw) -> 0-45 ms (raw + ID).

Safety, in this order of precedence:
  - stop() deactivates the stream at once; the caller then puts the arm on position hold
    (Limb.exit_control_mode). Nothing is published while inactive.
  - No command() for stale_after: the target velocity becomes zero, so the setpoint
    decelerates within the jerk limits and holds (e.g. the control loop stalled).
  - The setpoint never leads the last measured arm by more than max_lead per joint (0.3 rad: a
    backstop for a blocked arm, above the ~0.24 rad that 200 ms of raw-mode joint lag at
    1.2 rad/s produces in normal motion).
RAW_POSITION_MODE bypasses Baxter's collision avoidance.
"""

import threading
import time

import numpy as np

from .setpoint import JointLimits, JointSetpointGenerator, Setpoint

# Jerk limit = JERK_FACTOR x the acceleration limit, so full acceleration builds up in 0.1 s. A tuning
# choice: short against the ~0.3 s velocity transitions (adds little lag), long enough that the
# setpoint has no steps for the raw joint loop to ring on.
JERK_FACTOR = 10.0
RATE = 200.0  # Hz, setpoints per second: ~2.6 per control cycle (~76 Hz)


class RawPositionOutput:
    """Fixed-rate setpoint stream fed by a velocity target; publishing is delegated to `send`.

    `send(setpoint)` is called once per period while active (from the stream thread, or from
    tick() in tests), under the lock, so it must be quick (a publish). Thread-safe:
    command()/stop() may be called from the control loop while the stream runs.
    """

    def __init__(self, generator: JointSetpointGenerator, send, max_lead=0.3, stale_after=0.1, clock=time.monotonic):
        self.generator = generator
        self.send = send
        self.max_lead = float(max_lead)
        self.stale_after = float(stale_after)
        self.clock = clock
        self._lock = threading.Lock()
        self._active = False
        self._last_command = -np.inf
        self._measured = None
        self._thread = None
        self._running = False

    @property
    def active(self) -> bool:
        return self._active

    @property
    def velocity(self) -> np.ndarray:
        """Setpoint velocity now (zeros while inactive), e.g. for telemetry."""
        with self._lock:
            return self.generator.state.velocity.copy() if self._active else np.zeros(self.generator.dof)

    @property
    def position(self) -> np.ndarray | None:
        """Setpoint position now, or None while inactive (e.g. for the teleop to servo on)."""
        with self._lock:
            return self.generator.state.position.copy() if self._active else None

    def command(self, velocity, measured_position) -> None:
        """New target joint velocity from the control loop, with the arm's measured position.

        The first command after inactivity restarts the setpoint at rest from the measured arm.
        """
        with self._lock:
            if not self._active:
                self.generator.reset(measured_position)
                self._active = True
            self.generator.set_target_velocity(velocity)
            self._measured = np.asarray(measured_position, dtype=float)
            self._last_command = self.clock()

    def stop(self) -> None:
        """Stop streaming now; the caller holds the arm (e.g. Limb.exit_control_mode())."""
        with self._lock:
            self._active = False

    def tick(self) -> Setpoint | None:
        """One period: step the setpoint and send it. Returns it, or None while inactive."""
        with self._lock:
            if not self._active:
                return None
            if self.clock() - self._last_command > self.stale_after:
                self.generator.set_target_velocity(np.zeros(self.generator.dof))
            setpoint = self.generator.step()
            if self.generator.hold_within(self._measured, self.max_lead):
                setpoint = self.generator.state
            # Sent under the lock: once stop() returns, no raw setpoint can follow the caller's
            # position hold (it would put the arm back in raw mode at a stale setpoint).
            self.send(setpoint)
        return setpoint

    def start(self) -> 'RawPositionOutput':
        """Run tick() every generator period on a daemon thread, on absolute deadlines."""
        if self._thread is not None:
            raise RuntimeError('already started')
        self._running = True
        self._thread = threading.Thread(target=self._run, name='raw_joint_output', daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        self.stop()
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _run(self) -> None:
        period = self.generator.dt
        deadline = time.monotonic()
        while self._running:
            self.tick()
            deadline += period
            delay = deadline - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            elif delay < -period:  # fell behind (GIL, scheduling): skip rather than burst
                deadline = time.monotonic()


class JointCommandPublisher:
    """`send` for RawPositionOutput: JointCommand(RAW_POSITION_MODE) and, optionally, the inverse
    dynamics feedforward, both for one Baxter arm. ROS message construction only."""

    def __init__(self, node, arm, joint_names, limb_joint_names, inverse_dynamics=True):
        from baxter_core_msgs.msg import JointCommand
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from trajectory_msgs.msg import JointTrajectoryPoint

        self._JointCommand, self._JointTrajectoryPoint = JointCommand, JointTrajectoryPoint
        # As baxter_interface.Limb's command publisher: the newest command is the only one that matters.
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        ns = f'/robot/limb/{arm}/'
        self.names = list(joint_names)
        self._command_pub = node.create_publisher(JointCommand, ns + 'joint_command', qos)
        self._id_pub = (
            node.create_publisher(JointTrajectoryPoint, ns + 'inverse_dynamics_command', qos)
            if inverse_dynamics
            else None
        )
        # The feedforward point has no joint names: it is in the limb's own joint order.
        self._id_order = [self.names.index(name) for name in limb_joint_names]

    def __call__(self, setpoint: Setpoint) -> None:
        self._command_pub.publish(
            self._JointCommand(
                mode=self._JointCommand.RAW_POSITION_MODE,
                command=[float(x) for x in setpoint.position],
                names=self.names,
            )
        )
        if self._id_pub is not None:
            i = self._id_order
            self._id_pub.publish(
                self._JointTrajectoryPoint(
                    positions=[float(x) for x in setpoint.position[i]],
                    velocities=[float(x) for x in setpoint.velocity[i]],
                    accelerations=[float(x) for x in setpoint.acceleration[i]],
                )
            )


def start_raw_position_output(
    node, arm, joint_names, limb_joint_names, velocity, acceleration, low, high, inverse_dynamics
):
    """A started RawPositionOutput for one arm, publishing JointCommand (and the inverse dynamics
    feedforward with inverse_dynamics), within these joint limits (rad/s, rad/s^2, rad)."""
    acceleration = np.asarray(acceleration, dtype=np.float32)
    limits = JointLimits(
        velocity=np.asarray(velocity, dtype=np.float32),
        acceleration=acceleration,
        jerk=JERK_FACTOR * acceleration,
        position_low=np.asarray(low, dtype=np.float32),
        position_high=np.asarray(high, dtype=np.float32),
    )
    send = JointCommandPublisher(node, arm, joint_names, limb_joint_names, inverse_dynamics=inverse_dynamics)
    return RawPositionOutput(JointSetpointGenerator(limits, 1.0 / RATE), send).start()

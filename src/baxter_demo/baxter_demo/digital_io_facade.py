"""Hardware abstraction for a single Baxter digital IO component.

Callers create a DigitalIO instance and call on()/off()/toggle().
All ROS2 plumbing — publishers, subscriptions, QoS, callback groups —
is contained here and invisible to the calling node.
"""

from baxter_core_msgs.msg import DigitalIOState, DigitalOutputCommand
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy


class DigitalIO:
    def __init__(self, node: Node, component_id: str) -> None:
        self._node = node
        self._id = component_id
        self._state: bool | None = None

        # MutuallyExclusiveCallbackGroup means only one callback in this group
        # runs at a time. With the default SingleThreadedExecutor this is
        # redundant — callbacks are already serialized. It becomes load-bearing
        # the moment you switch to MultiThreadedExecutor, so declaring it now
        # makes the thread-safety contract explicit.
        self._cbg = MutuallyExclusiveCallbackGroup()

        # RELIABLE + VOLATILE + depth=1 for commands:
        # Every toggle must be delivered (RELIABLE), we don't need the bridge
        # to replay past commands to late joiners (VOLATILE), and only the
        # current desired state matters (depth=1).
        self._pub = node.create_publisher(
            DigitalOutputCommand,
            '/robot/digital_io/command',
            QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            ),
        )

        # State subscription: TRANSIENT_LOCAL would replay the last message to
        # us immediately on connect, removing the need to wait for the first
        # publish. Try swapping VOLATILE → TRANSIENT_LOCAL here as a Stage 2
        # experiment — it only works if the bridge publisher also uses
        # TRANSIENT_LOCAL (QoS must match on both sides for durability).
        node.create_subscription(
            DigitalIOState,
            f'/robot/digital_io/{component_id}/state',
            self._on_state,
            QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            ),
            callback_group=self._cbg,
        )

    def _on_state(self, msg: DigitalIOState) -> None:
        self._state = bool(msg.state)

    @property
    def state(self) -> bool | None:
        """Current known state, or None if no message has arrived yet."""
        return self._state

    def on(self) -> None:
        self._publish(True)

    def off(self) -> None:
        self._publish(False)

    def toggle(self) -> None:
        # If no state message has arrived yet, default to turning on.
        self._publish(not self._state if self._state is not None else True)

    def _publish(self, value: bool) -> None:
        cmd = DigitalOutputCommand()
        cmd.name = self._id
        cmd.value = value
        self._pub.publish(cmd)

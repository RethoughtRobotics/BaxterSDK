from baxter_core_msgs.msg import DigitalIOState, DigitalOutputCommand
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy


class DigitalIO:
    def __init__(self, node: Node, component_id: str):
        self._node = node
        self.state = None
        self._component_type = 'digital_io'
        self._id = component_id
        type_ns = '/robot/' + self._component_type
        base_topic = type_ns + '/' + self._id
        self._is_input_only = True

        self._pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._sub_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._sub = node.create_subscription(
            DigitalIOState, base_topic + '/state', self._callback, qos_profile=self._sub_qos
        )
        self._pub_output = node.create_publisher(DigitalOutputCommand, type_ns + '/command', qos_profile=self._pub_qos)

    def _callback(self, msg):
        """Get current state of Digital IO"""
        self.state = msg.state

    def on(self):
        """Turn on Digital IO"""
        cmd = DigitalOutputCommand()
        cmd.name = self._id
        cmd.value = True
        self._pub_output.publish(cmd)

    def off(self):
        """Turn off Digital IO"""
        cmd = DigitalOutputCommand()
        cmd.name = self._id
        cmd.value = False
        self._pub_output.publish(cmd)

    def toggle(self):
        """Toggle the state of Digital IO"""
        if self.state == DigitalIOState.ON:
            self.off()
        else:
            self.on()

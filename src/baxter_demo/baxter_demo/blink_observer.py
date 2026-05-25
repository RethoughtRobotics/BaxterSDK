"""Blink Baxter navigator lights using a ROS2 timer callback."""

import sys

import rclpy
from baxter_core_msgs.msg import DigitalOutputCommand
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy


class BlinkNode(Node):
    def __init__(self):
        super().__init__('blink')
        self._lit = False

        # RELIABLE: every toggle must be delivered — a dropped command leaves
        # the light stuck on or off until the next tick.
        # VOLATILE: no need to replay past commands to nodes that join late.
        # depth=1: only the current desired state matters; stale commands are useless.
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._pub = self.create_publisher(DigitalOutputCommand, '/robot/digital_io/command', qos)

        # The executor calls _blink at 1 Hz.
        # No while loop, no time.sleep, no spin_once.
        self._timer = self.create_timer(1.0, self._blink)

    def _blink(self):
        self._lit = not self._lit
        for name in ('left_inner_light', 'right_inner_light'):
            cmd = DigitalOutputCommand()
            cmd.name = name
            cmd.value = self._lit
            self._pub.publish(cmd)


def main():
    rclpy.init()
    node = BlinkNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())

"""Blink Baxter navigator lights using the DigitalIO hardware abstraction."""

import sys

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from baxter_demo.digital_io_facade import DigitalIO


class BlinkNode(Node):
    def __init__(self) -> None:
        super().__init__('blink_demo')

        # No publishers, no subscriptions, no QoS visible here.
        # DigitalIO owns all of that. This node only knows about
        # hardware components and what to do with them.
        self._left = DigitalIO(self, 'left_inner_light')
        self._right = DigitalIO(self, 'right_inner_light')

        self._timer = self.create_timer(1.0, self._blink)

    def _blink(self) -> None:
        self._left.toggle()
        self._right.toggle()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BlinkNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())

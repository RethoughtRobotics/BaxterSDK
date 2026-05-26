#!/usr/bin/env python3

# Copyright (c) 2013-2015, Rethink Robotics
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of the Rethink Robotics nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
Baxter RSDK Head Action Client Example
"""

import sys

import rclpy
from control_msgs.action import SingleJointPosition
from rclpy.action import ActionClient

import baxter_interface


class HeadClient(object):
    def __init__(self, node):
        self._node = node
        ns = 'robot/head/head_action'
        self._client = ActionClient(node, SingleJointPosition, ns)
        self._goal = SingleJointPosition.Goal()
        self._goal_handle = None
        self._result_future = None

        self._node.get_logger().info('Waiting for head action server...')
        if not self._client.wait_for_server(timeout_sec=10.0):
            self._node.get_logger().error('Exiting - Head Action Server Not Found')
            sys.exit(1)
        self.clear()

    def command(self, position, velocity):
        self._goal.position = position
        self._goal.max_velocity = velocity
        send_future = self._client.send_goal_async(self._goal)
        while not send_future.done():
            rclpy.spin_once(self._node, timeout_sec=0.1)
        self._goal_handle = send_future.result()
        if not self._goal_handle.accepted:
            self._node.get_logger().error('Goal rejected')
            return
        self._result_future = self._goal_handle.get_result_async()

    def stop(self):
        if self._goal_handle is not None:
            cancel_future = self._goal_handle.cancel_goal_async()
            while not cancel_future.done():
                rclpy.spin_once(self._node, timeout_sec=0.1)

    def wait(self, timeout=5.0):
        if self._result_future is None:
            return None
        deadline = self._node.get_clock().now().nanoseconds * 1e-9 + timeout
        while not self._result_future.done():
            rclpy.spin_once(self._node, timeout_sec=0.1)
            if self._node.get_clock().now().nanoseconds * 1e-9 > deadline:
                self._node.get_logger().warning('wait() timed out')
                return None
        return self._result_future.result().result

    def clear(self):
        self._goal = SingleJointPosition.Goal()
        self._goal_handle = None
        self._result_future = None


def main():
    """RSDK Head Example: Action Client

    Demonstrates creating a client of the Head Action Server,
    which enables sending commands of standard action type
    control_msgs/SingleJointPosition.

    The example will command the head to a position.
    Be sure to start Baxter's head_action_server before running this example.
    """
    print('Initializing node... ')
    rclpy.init()
    node = rclpy.create_node('rsdk_head_action_client')
    print('Getting robot state... ')
    rs = baxter_interface.RobotEnable(node=node)
    print('Enabling robot... ')
    rs.enable(node)
    print('Running. Ctrl-c to quit')

    hc = HeadClient(node)
    hc.command(position=0.0, velocity=1.0)
    hc.wait()
    hc.command(position=1.57, velocity=0.1)
    hc.wait()
    hc.command(position=0.0, velocity=0.8)
    hc.wait()
    hc.command(position=-1.0, velocity=0.4)
    hc.wait()
    hc.command(position=0.0, velocity=0.6)
    print(hc.wait())
    print('Exiting - Head Action Test Example Complete')

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

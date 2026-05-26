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
Baxter RSDK Gripper Action Client Example
"""

import argparse
import sys

import rclpy
from control_msgs.action import GripperCommand
from rclpy.action import ActionClient
from rclpy.utilities import remove_ros_args

import baxter_interface


class GripperClient(object):
    def __init__(self, gripper, node):
        self._node = node
        ns = 'robot/end_effector/' + gripper + '_gripper/gripper_action'
        self._client = ActionClient(node, GripperCommand, ns)
        self._goal = GripperCommand.Goal()
        self._goal_handle = None
        self._result_future = None

        self._node.get_logger().info('Waiting for %s gripper action server...' % gripper.capitalize())
        if not self._client.wait_for_server(timeout_sec=10.0):
            self._node.get_logger().error('Exiting - %s Gripper Action Server Not Found' % gripper.capitalize())
            sys.exit(1)
        self.clear()

    def command(self, position, effort):
        self._goal.command.position = position
        self._goal.command.max_effort = effort
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
        self._goal = GripperCommand.Goal()
        self._goal_handle = None
        self._result_future = None


def main():
    """RSDK Gripper Example: Action Client

    Demonstrates creating a client of the Gripper Action Server,
    which enables sending commands of standard action type
    control_msgs/GripperCommand.

    The example will command the grippers to a number of positions
    while specifying moving force or vacuum sensor threshold. Be sure
    to start Baxter's gripper_action_server before running this example.
    """
    arg_fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=arg_fmt, description=main.__doc__)
    parser.add_argument(
        '-g',
        '--gripper',
        dest='gripper',
        required=True,
        choices=['left', 'right'],
        help='which gripper to send action commands',
    )
    args = parser.parse_args(remove_ros_args(sys.argv)[1:])
    gripper = args.gripper

    print('Initializing node... ')
    rclpy.init()
    node = rclpy.create_node('rsdk_gripper_action_client_%s' % (gripper,))
    print('Getting robot state... ')
    rs = baxter_interface.RobotEnable(node=node)
    print('Enabling robot... ')
    rs.enable(node)
    print('Running. Ctrl-c to quit')

    gc = GripperClient(gripper, node)
    gc.command(position=0.0, effort=50.0)
    gc.wait()
    gc.command(position=100.0, effort=50.0)
    gc.wait()
    gc.command(position=25.0, effort=40.0)
    gc.wait()
    gc.command(position=75.0, effort=20.0)
    gc.wait()
    gc.command(position=0.0, effort=30.0)
    gc.wait()
    gc.command(position=100.0, effort=40.0)
    print(gc.wait())
    print('Exiting - Gripper Action Test Example Complete')

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

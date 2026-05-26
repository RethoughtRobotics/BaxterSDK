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
Baxter RSDK Head Action Server
"""

import time as _wtime
from math import fabs

import rclpy
from baxter_core_msgs.msg import HeadPanCommand
from control_msgs.action import SingleJointPosition
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node

import baxter_interface
from baxter_interface import settings


class HeadActionServer(object):
    def __init__(self, node: Node):
        self._node = node
        self._dyn = self._load_params(node)
        self._ns = 'robot/head/head_action'
        self._head = baxter_interface.Head(node=self._node)

        self._server = ActionServer(
            self._node,
            SingleJointPosition,
            self._ns,
            execute_callback=self._on_head_action,
            callback_group=ReentrantCallbackGroup(),
        )
        self._action_name = self._node.get_name()

        self._fdbk = SingleJointPosition.Feedback()
        self._result = SingleJointPosition.Result()

        self._prm = {'dead_zone': settings.HEAD_PAN_ANGLE_TOLERANCE}
        self._timeout = 5.0

    @staticmethod
    def _load_params(node):
        def _param(name, default):
            if node.has_parameter(name):
                return node.get_parameter(name).value
            return node.declare_parameter(name, default).value

        params = {}
        params['timeout'] = _param('head_timeout', 5.0)
        params['goal'] = _param('head_goal', settings.HEAD_PAN_ANGLE_TOLERANCE)
        return params

    def _get_head_parameters(self):
        self._timeout = self._dyn['timeout']
        self._prm['dead_zone'] = self._dyn['goal']

    def _update_feedback(self, goal_handle):
        self._fdbk.position = self._head.pan()
        goal_handle.publish_feedback(self._fdbk)

    def _command_head(self, position, speed):
        self._head.set_pan(position, speed, timeout=self._timeout)

    def _check_state(self, position):
        return fabs(self._head.pan() - position) < self._prm['dead_zone']

    def _on_head_action(self, goal_handle):
        goal = goal_handle.request
        position = goal.position
        velocity = goal.max_velocity
        if velocity < HeadPanCommand.MIN_SPEED_RATIO:
            velocity = HeadPanCommand.MAX_SPEED_RATIO

        self._get_head_parameters()
        self._update_feedback(goal_handle)

        _period = 1.0 / 20.0
        start_time = self._node.get_clock().now().nanoseconds * 1e-9

        def now_from_start(start):
            return self._node.get_clock().now().nanoseconds * 1e-9 - start

        while (now_from_start(start_time) < self._timeout or self._timeout < 0.0) and rclpy.ok():
            _iter_start = _wtime.time()
            if goal_handle.is_cancel_requested:
                self._node.get_logger().info('%s: Head Action Preempted' % (self._action_name,))
                goal_handle.canceled()
                return self._result
            self._update_feedback(goal_handle)
            if self._check_state(position):
                goal_handle.succeed()
                return self._result
            self._command_head(position, velocity)
            _remaining = _period - (_wtime.time() - _iter_start)
            if _remaining > 0.0:
                _wtime.sleep(_remaining)

        if rclpy.ok():
            self._node.get_logger().error('%s: Head Command Not Achieved in Allotted Time' % (self._action_name,))
        self._update_feedback(goal_handle)
        goal_handle.abort()
        return self._result

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
Baxter RSDK Gripper Action Server
"""

import time
from math import fabs

import rclpy
from control_msgs.action import GripperCommand
from rcl_interfaces.msg import SetParametersResult
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node

import baxter_interface


class GripperActionServer(object):
    def __init__(self, gripper, node: Node):
        self._node = node
        self._dyn = self._load_params(node, gripper)
        self._ee = gripper + '_gripper'

        def _on_set_parameters(params):
            for param in params:
                if param.name in self._dyn:
                    self._dyn[param.name] = param.value
            return SetParametersResult(successful=True)

        node.add_on_set_parameters_callback(_on_set_parameters)
        self._ns = 'robot/end_effector/' + self._ee + '/gripper_action'
        self._gripper = baxter_interface.Gripper(gripper, node=self._node)
        self._type = self._gripper.type()
        if self._type == 'custom':
            msg = 'Stopping %s action server - %s gripper not capable of gripper actions' % (
                self._gripper.name,
                self._type,
            )
            self._node.get_logger().error(msg)
            return

        if self._gripper.error():
            self._gripper.reset()
            if self._gripper.error():
                msg = 'Stopping %s action server - Unable to clear error' % self._gripper.name
                self._node.get_logger().error(msg)
                return
        if not self._gripper.calibrated():
            self._gripper.calibrate()
            if not self._gripper.calibrated():
                msg = 'Stopping %s action server - Unable to calibrate' % self._gripper.name
                self._node.get_logger().error(msg)
                return

        self._server = ActionServer(
            self._node,
            GripperCommand,
            self._ns,
            execute_callback=self._on_gripper_action,
            callback_group=ReentrantCallbackGroup(),
        )
        self._action_name = self._node.get_name()

        self._fdbk = GripperCommand.Feedback()
        self._result = GripperCommand.Result()

        self._prm = self._gripper.parameters()
        self._timeout = 5.0

    @staticmethod
    def _load_params(node, gripper):
        ee = gripper + '_gripper'

        def _param(name, default):
            if node.has_parameter(name):
                return node.get_parameter(name).value
            return node.declare_parameter(name, default).value

        params = {}
        params[ee + '_timeout'] = _param(ee + '_timeout', 5.0)
        params[ee + '_goal'] = _param(ee + '_goal', 0.1)
        params[ee + '_velocity'] = _param(ee + '_velocity', 50.0)
        params[ee + '_moving_force'] = _param(ee + '_moving_force', 40.0)
        params[ee + '_holding_force'] = _param(ee + '_holding_force', 30.0)
        params[ee + '_vacuum_threshold'] = _param(ee + '_vacuum_threshold', 18.0)
        params[ee + '_blow_off'] = _param(ee + '_blow_off', 0.4)
        return params

    def _get_gripper_parameters(self):
        self._timeout = self._dyn[self._ee + '_timeout']
        if self._type == 'electric':
            self._prm['dead_zone'] = self._dyn[self._ee + '_goal']
            self._prm['velocity'] = self._dyn[self._ee + '_velocity']
            self._prm['moving_force'] = self._dyn[self._ee + '_moving_force']
            self._prm['holding_force'] = self._dyn[self._ee + '_holding_force']
        elif self._type == 'suction':
            self._prm['vacuum_sensor_threshold'] = self._dyn[self._ee + '_vacuum_threshold']
            self._prm['blow_off_seconds'] = self._dyn[self._ee + '_blow_off']
        self._gripper.set_parameters(parameters=self._prm)

    def _update_feedback(self, position, goal_handle):
        if self._type == 'electric':
            self._fdbk.position = self._gripper.position()
            self._fdbk.effort = self._gripper.force()
            self._fdbk.stalled = self._gripper.force() > self._gripper.parameters()['moving_force']
            self._fdbk.reached_goal = (
                fabs(self._gripper.position() - position) < self._gripper.parameters()['dead_zone']
            )
        if self._type == 'suction':
            self._fdbk.effort = self._gripper.vacuum_sensor()
            if position >= 100.0:
                self._fdbk.reached_goal = not self._gripper.sucking() and not self._gripper.blowing()
            else:
                self._fdbk.reached_goal = self._gripper.gripping()
        self._result.position = self._fdbk.position
        self._result.effort = self._fdbk.effort
        self._result.stalled = self._fdbk.stalled
        self._result.reached_goal = self._fdbk.reached_goal
        goal_handle.publish_feedback(self._fdbk)

    def _command_gripper(self, position):
        if self._type == 'electric':
            self._gripper.command_position(position, block=False)
        elif self._type == 'suction':
            if position >= 100.0:
                self._gripper.open(block=False)
            else:
                if self._timeout < 0.0:
                    self._timeout = 3600.0
                self._gripper.close(block=False, timeout=self._timeout)

    def _check_state(self, position):
        if self._type == 'electric':
            return (
                self._gripper.force() > self._gripper.parameters()['moving_force']
                or fabs(self._gripper.position() - position) < self._gripper.parameters()['dead_zone']
            )
        elif self._type == 'suction':
            if position >= 100.0:
                return not self._gripper.sucking() and not self._gripper.blowing()
            else:
                return self._gripper.gripping()

    def _on_gripper_action(self, goal_handle):
        goal = goal_handle.request
        position = goal.command.position
        effort = goal.command.max_effort
        if effort == -1.0:
            effort = 100.0

        if self._gripper.error():
            self._node.get_logger().error('%s: Gripper error - please restart action server.' % (self._action_name,))
            goal_handle.abort()
            return self._result

        self._get_gripper_parameters()
        self._update_feedback(position, goal_handle)

        _period = 1.0 / 20.0
        start_time = self._node.get_clock().now().nanoseconds * 1e-9

        if self._type == 'electric':
            if fabs(effort) < 0.0001:
                effort = self._prm['moving_force']
            self._gripper.set_moving_force(effort)
        elif self._type == 'suction':
            if fabs(effort) < 0.0001:
                effort = self._prm['vacuum_sensor_threshold']
            self._gripper.set_vacuum_threshold(effort)

        def now_from_start(start):
            return self._node.get_clock().now().nanoseconds * 1e-9 - start

        while (now_from_start(start_time) < self._timeout or self._timeout < 0.0) and rclpy.ok():
            _iter_start = time.time()
            if goal_handle.is_cancel_requested:
                self._gripper.stop()
                self._node.get_logger().info('%s: Gripper Action Preempted' % (self._action_name,))
                goal_handle.canceled()
                return self._result
            self._update_feedback(position, goal_handle)
            if self._check_state(position):
                goal_handle.succeed()
                return self._result
            self._command_gripper(position)
            _remaining = _period - (time.time() - _iter_start)
            if _remaining > 0.0:
                time.sleep(_remaining)

        self._gripper.stop()
        if rclpy.ok():
            self._node.get_logger().error('%s: Gripper Command Not Achieved in Allotted Time' % (self._action_name,))
        self._update_feedback(position, goal_handle)
        goal_handle.abort()
        return self._result

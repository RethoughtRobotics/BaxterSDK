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
Baxter RSDK Joint Trajectory Action Server
"""

import bisect
import math
import operator
import time as _wtime
from copy import deepcopy

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from std_msgs.msg import (
    UInt16,
)
from trajectory_msgs.msg import (
    JointTrajectoryPoint,
)

import baxter_interface

from . import bezier, minjerk


class JointTrajectoryActionServer(object):
    def __init__(self, limb, reconfig_server=None, node=None, rate=100.0, mode='position_w_id', interpolation='bezier'):
        self._dyn = self._load_params(node)
        self._node = node
        self._ns = 'robot/limb/' + limb
        self._fjt_ns = limb + '_arm/follow_joint_trajectory'
        self._server = ActionServer(
            self._node,
            FollowJointTrajectory,
            self._fjt_ns,
            execute_callback=self._on_trajectory_action,
            callback_group=ReentrantCallbackGroup(),
        )
        self._action_name = self._node.get_name()
        # All I/O on the main node — rmw_zenoh_cpp only flushes
        # publishes when the publisher's node is on the same executor
        # and NOT on a separate _io_node.
        self._limb = baxter_interface.Limb(limb, self._node)
        self._enable = baxter_interface.RobotEnable(node=self._node)
        self._name = limb
        self._interpolation = interpolation
        self._cuff = baxter_interface.DigitalIO('%s_lower_cuff' % (limb,), self._node)
        self._cuff.state_changed.connect(self._cuff_cb)
        # Verify joint control mode
        self._mode = mode
        if self._mode != 'position' and self._mode != 'position_w_id' and self._mode != 'velocity':
            self._node.get_logger().error(
                '%s: Action Server Creation Failed - '
                "Provided Invalid Joint Control Mode '%s' (Options: "
                "'position_w_id', 'position', 'velocity')"
                % (
                    self._action_name,
                    self._mode,
                )
            )
            return
        self._alive = True
        self._cuff_state = False
        # Action Feedback/Result
        self._fdbk = FollowJointTrajectory.Feedback()
        self._result = FollowJointTrajectory.Result()

        # Controller parameters from arguments, messages, and dynamic
        # reconfigure
        self._control_rate = rate  # Hz
        self._control_joints = []
        self._pid_gains = {'kp': dict(), 'ki': dict(), 'kd': dict()}
        self._goal_time = 0.0
        self._stopped_velocity = 0.0
        self._goal_error = dict()
        self._path_thresh = dict()

        # Create our spline coefficients
        self._coeff = [None] * len(self._limb.joint_names())

        # Set joint state publishing to specified control rate
        self._pub_rate = self._node.create_publisher(UInt16, '/robot/joint_state_publish_rate', 10)
        msg = UInt16()
        msg.data = int(self._control_rate)
        self._pub_rate.publish(msg)

        self._pub_ff_cmd = self._node.create_publisher(JointTrajectoryPoint, self._ns + '/inverse_dynamics_command', 1)

    @staticmethod
    def _load_params(node):
        """Declare and load trajectory parameters from the node (replaces dynamic_reconfigure)."""

        def _param(name, default):
            if node.has_parameter(name):
                return node.get_parameter(name).value
            return node.declare_parameter(name, default).value

        params = {}
        params['goal_time'] = _param('goal_time', 0.1)
        params['stopped_velocity_tolerance'] = _param('stopped_velocity_tolerance', 0.20)
        joints = (
            'left_s0',
            'left_s1',
            'left_e0',
            'left_e1',
            'left_w0',
            'left_w1',
            'left_w2',
            'right_s0',
            'right_s1',
            'right_e0',
            'right_e1',
            'right_w0',
            'right_w1',
            'right_w2',
        )
        for jnt in joints:
            params[jnt + '_trajectory'] = _param(jnt + '_trajectory', 0.35)
            params[jnt + '_goal'] = _param(jnt + '_goal', -1.0)
            params[jnt + '_kp'] = _param(jnt + '_kp', 2.0)
            params[jnt + '_ki'] = _param(jnt + '_ki', 0.0)
            params[jnt + '_kd'] = _param(jnt + '_kd', 0.0)
        return params

    def robot_is_enabled(self):
        return self._enable.state().enabled

    def clean_shutdown(self):
        self._alive = False
        self._limb.exit_control_mode()

    def _cuff_cb(self, value):
        self._cuff_state = value

    def _get_trajectory_parameters(self, joint_names, goal, goal_handle):
        # For each input trajectory, if path, goal, or goal_time tolerances
        # provided, we will use these as opposed to reading from the
        # parameter server/dynamic reconfigure

        # Goal time tolerance - time buffer allowing goal constraints to be met
        gt = goal.goal_time_tolerance
        if gt.sec != 0 or gt.nanosec != 0:
            self._goal_time = gt.sec + gt.nanosec * 1e-9
        else:
            self._goal_time = self._dyn['goal_time']
        # Stopped velocity tolerance - max velocity at end of execution
        self._stopped_velocity = self._dyn['stopped_velocity_tolerance']

        # Path execution and goal tolerances per joint
        for jnt in joint_names:
            if jnt not in self._limb.joint_names():
                self._node.get_logger().error(
                    '%s: Trajectory Aborted - Provided Invalid Joint Name %s'
                    % (
                        self._action_name,
                        jnt,
                    )
                )
                self._result.error_code = self._result.INVALID_JOINTS
                goal_handle.abort()
                return False
            # Path execution tolerance
            path_error = self._dyn[jnt + '_trajectory']
            if goal.path_tolerance:
                for tolerance in goal.path_tolerance:
                    if jnt == tolerance.name:
                        if tolerance.position != 0.0:
                            self._path_thresh[jnt] = tolerance.position
                        else:
                            self._path_thresh[jnt] = path_error
            else:
                self._path_thresh[jnt] = path_error
            # Goal error tolerance
            goal_error = self._dyn[jnt + '_goal']
            if goal.goal_tolerance:
                for tolerance in goal.goal_tolerance:
                    if jnt == tolerance.name:
                        if tolerance.position != 0.0:
                            self._goal_error[jnt] = tolerance.position
                        else:
                            self._goal_error[jnt] = goal_error
            else:
                self._goal_error[jnt] = goal_error
        return True

    def _get_current_position(self, joint_names):
        return [self._limb.joint_angle(joint) for joint in joint_names]

    def _get_current_velocities(self, joint_names):
        return [self._limb.joint_velocity(joint) for joint in joint_names]

    def _get_current_error(self, joint_names, set_point):
        current = self._get_current_position(joint_names)
        error = list(map(operator.sub, set_point, current))
        return zip(joint_names, error)

    def _update_feedback(self, cmd_point, jnt_names, cur_time, goal_handle):
        self._fdbk.joint_names = jnt_names
        self._fdbk.desired = cmd_point
        self._fdbk.desired.time_from_start = Duration(sec=int(cur_time), nanosec=int((cur_time % 1) * 1e9))
        self._fdbk.actual.positions = self._get_current_position(jnt_names)
        self._fdbk.actual.time_from_start = Duration(sec=int(cur_time), nanosec=int((cur_time % 1) * 1e9))
        self._fdbk.error.positions = list(map(operator.sub, self._fdbk.desired.positions, self._fdbk.actual.positions))
        self._fdbk.error.time_from_start = Duration(sec=int(cur_time), nanosec=int((cur_time % 1) * 1e9))
        goal_handle.publish_feedback(self._fdbk)

    def _reorder_joints_ff_cmd(self, joint_names, point):
        joint_name_order = self._limb.joint_names()
        pnt = JointTrajectoryPoint()
        pnt.time_from_start = point.time_from_start
        pos_cmd = dict(zip(joint_names, point.positions))
        for jnt_name in joint_name_order:
            pnt.positions.append(pos_cmd[jnt_name])
        if point.velocities:
            vel_cmd = dict(zip(joint_names, point.velocities))
            for jnt_name in joint_name_order:
                pnt.velocities.append(vel_cmd[jnt_name])
        if point.accelerations:
            accel_cmd = dict(zip(joint_names, point.accelerations))
            for jnt_name in joint_name_order:
                pnt.accelerations.append(accel_cmd[jnt_name])
        return pnt

    def _command_stop(self, joint_names, joint_angles, start_time, dimensions_dict):
        """Send one final position command.

        In ROS1 this was a holding while-loop, but in ROS2 the action
        result is only sent when the execute callback returns, so we
        cannot loop here.  A single command is sufficient because
        Baxter holds its last commanded position.
        """
        if self._mode == 'velocity':
            velocities = [0.0] * len(joint_names)
            cmd = dict(zip(joint_names, velocities))
            self._limb.set_joint_velocities(cmd)
        elif self._mode == 'position' or self._mode == 'position_w_id':
            raw_pos_mode = self._mode == 'position_w_id'
            self._limb.set_joint_positions(joint_angles, raw=raw_pos_mode)
            if raw_pos_mode:
                pnt = JointTrajectoryPoint()
                pnt.positions = self._get_current_position(joint_names)
                if dimensions_dict['velocities']:
                    pnt.velocities = [0.0] * len(joint_names)
                if dimensions_dict['accelerations']:
                    pnt.accelerations = [0.0] * len(joint_names)
                elapsed = self._node.get_clock().now().nanoseconds * 1e-9 - start_time
                pnt.time_from_start = Duration(sec=int(elapsed), nanosec=int((elapsed % 1) * 1e9))
                ff_pnt = self._reorder_joints_ff_cmd(joint_names, pnt)
                self._pub_ff_cmd.publish(ff_pnt)

    def _command_joints(self, joint_names, point, start_time, dimensions_dict, goal_handle):
        if goal_handle.is_cancel_requested or not self.robot_is_enabled():
            self._node.get_logger().info('%s: Trajectory Preempted' % (self._action_name,))
            goal_handle.canceled()
            self._command_stop(joint_names, self._limb.joint_angles(), start_time, dimensions_dict)
            return False
        velocities = []
        deltas = self._get_current_error(joint_names, point.positions)
        for delta in deltas:
            if (
                math.fabs(delta[1]) >= self._path_thresh[delta[0]] and self._path_thresh[delta[0]] >= 0.0
            ) or not self.robot_is_enabled():
                self._node.get_logger().error(
                    '%s: Exceeded Error Threshold on %s: %s'
                    % (
                        self._action_name,
                        delta[0],
                        str(delta[1]),
                    )
                )
                self._result.error_code = self._result.PATH_TOLERANCE_VIOLATED
                goal_handle.abort()
                self._command_stop(joint_names, self._limb.joint_angles(), start_time, dimensions_dict)
                return False
        if (self._mode == 'position' or self._mode == 'position_w_id') and self._alive:
            cmd = dict(zip(joint_names, point.positions))
            raw_pos_mode = self._mode == 'position_w_id'
            self._limb.set_joint_positions(cmd, raw=raw_pos_mode)
            if raw_pos_mode:
                ff_pnt = self._reorder_joints_ff_cmd(joint_names, point)
                self._pub_ff_cmd.publish(ff_pnt)
        elif self._alive:
            cmd = dict(zip(joint_names, velocities))
            self._limb.set_joint_velocities(cmd)
        return True

    def _get_bezier_point(self, b_matrix, idx, t, cmd_time, dimensions_dict):
        pnt = JointTrajectoryPoint()
        pnt.time_from_start = Duration(sec=int(cmd_time), nanosec=int((cmd_time % 1) * 1e9))
        num_joints = b_matrix.shape[0]
        pnt.positions = [0.0] * num_joints
        if dimensions_dict['velocities']:
            pnt.velocities = [0.0] * num_joints
        if dimensions_dict['accelerations']:
            pnt.accelerations = [0.0] * num_joints
        for jnt in range(num_joints):
            b_point = bezier.bezier_point(b_matrix[jnt, :, :, :], idx, t)
            # Positions at specified time
            pnt.positions[jnt] = b_point[0]
            # Velocities at specified time
            if dimensions_dict['velocities']:
                pnt.velocities[jnt] = b_point[1]
            # Accelerations at specified time
            if dimensions_dict['accelerations']:
                pnt.accelerations[jnt] = b_point[-1]
        return pnt

    def _compute_bezier_coeff(self, joint_names, trajectory_points, dimensions_dict):
        # Compute Full Bezier Curve
        num_joints = len(joint_names)
        num_traj_pts = len(trajectory_points)
        num_traj_dim = sum(dimensions_dict.values())
        num_b_values = len(['b0', 'b1', 'b2', 'b3'])
        b_matrix = np.zeros(shape=(num_joints, num_traj_dim, num_traj_pts - 1, num_b_values))
        for jnt in range(num_joints):
            traj_array = np.zeros(shape=(len(trajectory_points), num_traj_dim))
            for idx, point in enumerate(trajectory_points):
                current_point = list()
                current_point.append(point.positions[jnt])
                if dimensions_dict['velocities']:
                    current_point.append(point.velocities[jnt])
                if dimensions_dict['accelerations']:
                    current_point.append(point.accelerations[jnt])
                traj_array[idx, :] = current_point
            d_pts = bezier.de_boor_control_pts(traj_array)
            b_matrix[jnt, :, :, :] = bezier.bezier_coefficients(traj_array, d_pts)
        return b_matrix

    def _compute_bezier_with_velocity_coeff(self, joint_names, trajectory_points, dimensions_dict):
        # Compute Full Bezier Curve
        num_joints = len(joint_names)
        num_traj_pts = len(trajectory_points)
        num_traj_dim = sum(dimensions_dict.values())
        num_b_values = len(['b0', 'b1', 'b2', 'b3'])
        b_matrix = np.zeros(shape=(num_joints, num_traj_dim, num_traj_pts - 1, num_b_values))
        for jnt in range(num_joints):
            traj_array = np.zeros(shape=(len(trajectory_points), num_traj_dim))
            for idx, point in enumerate(trajectory_points):
                current_point = list()
                current_point.append(point.positions[jnt])
                if dimensions_dict['velocities']:
                    current_point.append(point.velocities[jnt])
                if dimensions_dict['accelerations']:
                    current_point.append(point.accelerations[jnt])
                traj_array[idx, :] = current_point
            b_matrix[jnt, :, :, :] = bezier.bezier_coefficients(traj_array)
        return b_matrix

    def _get_minjerk_point(self, m_matrix, idx, t, cmd_time, dimensions_dict):
        pnt = JointTrajectoryPoint()
        pnt.time_from_start = Duration(sec=int(cmd_time), nanosec=int((cmd_time % 1) * 1e9))
        num_joints = m_matrix.shape[0]
        pnt.positions = [0.0] * num_joints
        if dimensions_dict['velocities']:
            pnt.velocities = [0.0] * num_joints
        if dimensions_dict['accelerations']:
            pnt.accelerations = [0.0] * num_joints
        for jnt in range(num_joints):
            m_point = minjerk.minjerk_point(m_matrix[jnt, :, :, :], idx, t)
            # Positions at specified time
            pnt.positions[jnt] = m_point[0]
            # Velocities at specified time
            if dimensions_dict['velocities']:
                pnt.velocities[jnt] = m_point[1]
            # Accelerations at specified time
            if dimensions_dict['accelerations']:
                pnt.accelerations[jnt] = m_point[-1]
        return pnt

    def _compute_minjerk_coeff(self, joint_names, trajectory_points, point_duration, dimensions_dict):
        # Compute Full Minimum Jerk Curve
        num_joints = len(joint_names)
        num_traj_pts = len(trajectory_points)
        num_traj_dim = sum(dimensions_dict.values())
        num_m_values = len(['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'tm'])
        m_matrix = np.zeros(shape=(num_joints, num_traj_dim, num_traj_pts - 1, num_m_values))
        for jnt in range(num_joints):
            traj_array = np.zeros(shape=(len(trajectory_points), num_traj_dim))
            for idx, point in enumerate(trajectory_points):
                current_point = list()
                current_point.append(point.positions[jnt])
                if dimensions_dict['velocities']:
                    current_point.append(point.velocities[jnt])
                if dimensions_dict['accelerations']:
                    current_point.append(point.accelerations[jnt])
                traj_array[idx, :] = current_point
            m_matrix[jnt, :, :, :] = minjerk.minjerk_coefficients(traj_array, point_duration)
        return m_matrix

    def _determine_dimensions(self, trajectory_points):
        # Determine dimensions supplied
        position_flag = True
        velocity_flag = len(trajectory_points[0].velocities) != 0 and len(trajectory_points[-1].velocities) != 0
        acceleration_flag = (
            len(trajectory_points[0].accelerations) != 0 and len(trajectory_points[-1].accelerations) != 0
        )
        return {'positions': position_flag, 'velocities': velocity_flag, 'accelerations': acceleration_flag}

    def _on_trajectory_action(self, goal_handle):
        goal = goal_handle.request
        joint_names = list(goal.trajectory.joint_names)
        trajectory_points = list(goal.trajectory.points)
        if not self._get_trajectory_parameters(joint_names, goal, goal_handle):
            return self._result
        num_points = len(trajectory_points)
        if num_points == 0:
            self._node.get_logger().error('%s: Empty Trajectory' % (self._action_name,))
            goal_handle.abort()
            return self._result
        self._node.get_logger().info('%s: Executing requested joint trajectory' % (self._action_name,))

        dimensions_dict = self._determine_dimensions(trajectory_points)

        if num_points == 1:
            first_trajectory_point = JointTrajectoryPoint()
            first_trajectory_point.positions = self._get_current_position(joint_names)
            if dimensions_dict['velocities']:
                first_trajectory_point.velocities = deepcopy(trajectory_points[0].velocities)
            if dimensions_dict['accelerations']:
                first_trajectory_point.accelerations = deepcopy(trajectory_points[0].accelerations)
            first_trajectory_point.time_from_start = Duration(sec=0, nanosec=0)
            trajectory_points.insert(0, first_trajectory_point)
            num_points = len(trajectory_points)

        if dimensions_dict['velocities']:
            trajectory_points[-1].velocities = [0.0] * len(joint_names)
        if dimensions_dict['accelerations']:
            trajectory_points[-1].accelerations = [0.0] * len(joint_names)

        pnt_times = [pnt.time_from_start.sec + pnt.time_from_start.nanosec * 1e-9 for pnt in trajectory_points]
        try:
            if self._interpolation == 'minjerk':
                point_duration = [pnt_times[i + 1] - pnt_times[i] for i in range(len(pnt_times) - 1)]
                m_matrix = self._compute_minjerk_coeff(joint_names, trajectory_points, point_duration, dimensions_dict)
            elif self._interpolation == 'bezier_with_velocity':
                b_matrix = self._compute_bezier_with_velocity_coeff(joint_names, trajectory_points, dimensions_dict)
            else:
                b_matrix = self._compute_bezier_coeff(joint_names, trajectory_points, dimensions_dict)
        except Exception as ex:
            self._node.get_logger().error(
                ('{0}: Failed to compute a {1} trajectory for {2} arm with error "{3}: {4}"').format(
                    self._interpolation, self._action_name, self._name, type(ex).__name__, ex
                )
            )
            goal_handle.abort()
            return self._result

        hdr = goal.trajectory.header.stamp
        start_time = hdr.sec + hdr.nanosec * 1e-9
        if start_time == 0.0:
            start_time = self._node.get_clock().now().nanoseconds * 1e-9

        while self._node.get_clock().now().nanoseconds * 1e-9 < start_time and rclpy.ok():
            _wtime.sleep(0.001)

        _period = 1.0 / self._control_rate

        now_from_start = self._node.get_clock().now().nanoseconds * 1e-9 - start_time
        while now_from_start < pnt_times[-1] and rclpy.ok() and self.robot_is_enabled():
            _iter_start = _wtime.time()
            idx = bisect.bisect_left(pnt_times, now_from_start) - 1
            idx = max(0, min(idx, len(pnt_times) - 2))
            t_seg_start = pnt_times[idx]
            t_seg_end = pnt_times[idx + 1]
            seg_dur = t_seg_end - t_seg_start
            t = (now_from_start - t_seg_start) / seg_dur if seg_dur > 0.0 else 0.0
            t = max(0.0, min(1.0, t))

            if self._interpolation == 'minjerk':
                pnt = self._get_minjerk_point(m_matrix, idx, t, now_from_start, dimensions_dict)
            else:
                pnt = self._get_bezier_point(b_matrix, idx, t, now_from_start, dimensions_dict)

            if not self._command_joints(joint_names, pnt, start_time, dimensions_dict, goal_handle):
                return self._result
            self._update_feedback(pnt, joint_names, now_from_start, goal_handle)
            _remaining = _period - (_wtime.time() - _iter_start)
            if _remaining > 0.0:
                _wtime.sleep(_remaining)
            now_from_start = self._node.get_clock().now().nanoseconds * 1e-9 - start_time

        last = trajectory_points[-1]
        last_time = pnt_times[-1]
        end_angles = dict(zip(joint_names, last.positions))

        def check_goal_state():
            for error in self._get_current_error(joint_names, last.positions):
                if self._goal_error[error[0]] > 0 and self._goal_error[error[0]] < math.fabs(error[1]):
                    return error[0]
            if (
                self._stopped_velocity > 0.0
                and max([abs(cur_vel) for cur_vel in self._get_current_velocities(joint_names)])
                > self._stopped_velocity
            ):
                return False
            return True

        now_from_start = self._node.get_clock().now().nanoseconds * 1e-9 - start_time
        while now_from_start < (last_time + self._goal_time) and rclpy.ok() and self.robot_is_enabled():
            _iter_start = _wtime.time()
            if not self._command_joints(joint_names, last, start_time, dimensions_dict, goal_handle):
                return self._result
            now_from_start = self._node.get_clock().now().nanoseconds * 1e-9 - start_time
            self._update_feedback(deepcopy(last), joint_names, now_from_start, goal_handle)
            _remaining = _period - (_wtime.time() - _iter_start)
            if _remaining > 0.0:
                _wtime.sleep(_remaining)

        now_from_start = self._node.get_clock().now().nanoseconds * 1e-9 - start_time
        self._update_feedback(deepcopy(last), joint_names, now_from_start, goal_handle)

        result = check_goal_state()
        if result is True:
            self._node.get_logger().info(
                '%s: Joint Trajectory Action Succeeded for %s arm' % (self._action_name, self._name)
            )
            self._result.error_code = self._result.SUCCESSFUL
            goal_handle.succeed()
        elif result is False:
            self._node.get_logger().error(
                '%s: Exceeded Max Goal Velocity Threshold for %s arm' % (self._action_name, self._name)
            )
            self._result.error_code = self._result.GOAL_TOLERANCE_VIOLATED
            goal_handle.abort()
        else:
            self._node.get_logger().error(
                '%s: Exceeded Goal Threshold Error %s for %s arm' % (self._action_name, result, self._name)
            )
            self._result.error_code = self._result.GOAL_TOLERANCE_VIOLATED
            goal_handle.abort()
        self._command_stop(goal.trajectory.joint_names, end_angles, start_time, dimensions_dict)
        return self._result

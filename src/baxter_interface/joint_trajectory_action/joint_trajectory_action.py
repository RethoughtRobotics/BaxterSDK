import bisect
import time

import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node

CONTROL_RATE = 100.0  # Hz — Baxter position controller requires high-frequency commands


class JointTrajectoryActionServer:
    """
    ROS2 FollowJointTrajectory action server for one or both Baxter limbs.

    Runs a 100 Hz control loop for the duration of the trajectory, linearly
    interpolating between MoveIt's dense waypoints and continuously streaming
    position commands to Baxter via the Limb class.
    """

    def __init__(self, action_name: str, limbs: list, node: Node):
        self._node = node
        self._limbs = limbs

        self._limb_for_joint = {}
        for limb in limbs:
            for name in limb.joint_names():
                self._limb_for_joint[name] = limb

        self._server = ActionServer(
            node,
            FollowJointTrajectory,
            action_name,
            execute_callback=self._execute_cb,
            goal_callback=lambda _: GoalResponse.ACCEPT,
            cancel_callback=lambda _: CancelResponse.ACCEPT,
            callback_group=ReentrantCallbackGroup(),
        )
        node.get_logger().info(f'Action server ready: {action_name}')

    def destroy(self):
        self._server.destroy()

    def _execute_cb(self, goal_handle):
        trajectory = goal_handle.request.trajectory
        joint_names = trajectory.joint_names
        points = trajectory.points

        if not points:
            goal_handle.succeed()
            return self._result(FollowJointTrajectory.Result.SUCCESSFUL)

        end_time = self._duration_sec(points[-1])

        self._node.get_logger().info(f'Executing trajectory: {len(points)} points, duration {end_time:.2f}s')

        # Set command timeout generously beyond the trajectory duration so the
        # bridge never drops commands mid-execution.
        for limb in self._limbs:
            limb.set_command_timeout(end_time + 5.0)

        traj_times = [self._duration_sec(p) for p in points]
        traj_positions = [list(p.positions) for p in points]
        dt = 1.0 / CONTROL_RATE

        start_time = self._node.get_clock().now()

        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return self._result(FollowJointTrajectory.Result.SUCCESSFUL)

            elapsed = (self._node.get_clock().now() - start_time).nanoseconds / 1e9
            cmd = _interpolate(traj_times, traj_positions, min(elapsed, end_time))

            cmds = {limb: {} for limb in self._limbs}
            for i, name in enumerate(joint_names):
                limb = self._limb_for_joint.get(name)
                if limb is not None:
                    cmds[limb][name] = cmd[i]
            for limb, positions in cmds.items():
                if positions:
                    limb.set_joint_positions(positions)

            if elapsed >= end_time:
                break

            time.sleep(dt)

        goal_handle.succeed()
        return self._result(FollowJointTrajectory.Result.SUCCESSFUL)

    def _result(self, error_code):
        result = FollowJointTrajectory.Result()
        result.error_code = error_code
        return result

    @staticmethod
    def _duration_sec(point):
        return point.time_from_start.sec + point.time_from_start.nanosec / 1e9


def _interpolate(times, positions, t):
    if t <= times[0]:
        return list(positions[0])
    if t >= times[-1]:
        return list(positions[-1])
    idx = bisect.bisect_right(times, t) - 1
    t0, t1 = times[idx], times[idx + 1]
    alpha = (t - t0) / (t1 - t0)
    return [p0 + alpha * (p1 - p0) for p0, p1 in zip(positions[idx], positions[idx + 1])]

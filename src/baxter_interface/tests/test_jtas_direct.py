#!/usr/bin/env python3
"""Send a FollowJointTrajectory goal directly to the JTAS (no MoveIt).
Moves right_s1 by +0.1 rad over 3 seconds.
"""

import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from trajectory_msgs.msg import JointTrajectoryPoint

import baxter_interface


def main():
    rclpy.init()
    node = rclpy.create_node('test_jtas_direct')
    log = node.get_logger()

    # Read current joint angles from Limb
    limb = baxter_interface.Limb('right', node)
    current = limb.joint_angles()
    names = limb.joint_names()
    cur_positions = [current[j] for j in names]
    log.info(f'Current right_s1 = {current["right_s1"]:.4f}')

    # Build target: move right_s1 by +0.1 rad
    target_positions = list(cur_positions)
    s1_idx = names.index('right_s1')
    target_positions[s1_idx] += 0.1

    # Create action client
    client = ActionClient(node, FollowJointTrajectory, 'robot/limb/right/follow_joint_trajectory')
    log.info('Waiting for JTAS action server...')
    if not client.wait_for_server(timeout_sec=5.0):
        log.error('JTAS not available!')
        return

    # Build goal
    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = names

    # Start point (now)
    p0 = JointTrajectoryPoint()
    p0.positions = cur_positions
    p0.velocities = [0.0] * len(names)
    p0.time_from_start = Duration(sec=0, nanosec=0)

    # End point (3 seconds later)
    p1 = JointTrajectoryPoint()
    p1.positions = target_positions
    p1.velocities = [0.0] * len(names)
    p1.time_from_start = Duration(sec=3, nanosec=0)

    goal.trajectory.points = [p0, p1]

    log.info(f'Sending trajectory: right_s1 {cur_positions[s1_idx]:.4f} -> {target_positions[s1_idx]:.4f} over 3s')
    future = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)

    goal_handle = future.result()
    if not goal_handle.accepted:
        log.error('Goal rejected!')
        return
    log.info('Goal accepted, waiting for result...')

    result_future = goal_handle.get_result_async()
    rclpy.spin_until_future_complete(node, result_future, timeout_sec=15.0)

    result = result_future.result()
    log.info(f'Result: error_code={result.result.error_code}')

    # Check actual position
    rclpy.spin_once(node, timeout_sec=0.5)
    after = limb.joint_angle('right_s1')
    delta = abs(after - current['right_s1'])
    log.info(f'AFTER: right_s1 = {after:.4f}, delta = {delta:.4f}')
    if delta > 0.01:
        log.info('*** SUCCESS: JTAS MOVED THE ARM ***')
    else:
        log.warn('*** FAIL: JTAS DID NOT MOVE THE ARM ***')

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

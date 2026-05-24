#!/usr/bin/env python3
"""Minimal test: move one joint using the Limb class directly (no MoveIt, no JTAS).
Mimics what tuck_arms does: spin_once + set_joint_positions in a tight loop.
"""

import time

import rclpy

import baxter_interface


def main():
    rclpy.init()
    node = rclpy.create_node('test_limb_move')
    log = node.get_logger()

    log.info('Creating right Limb...')
    limb = baxter_interface.Limb('right', node)
    log.info('Limb ready. Joint names: %s' % limb.joint_names())

    # Read current positions
    before = limb.joint_angles()
    target_joint = 'right_s1'
    original = before.get(target_joint, 0.0)
    target_val = original + 0.10  # move 0.1 rad (~6 degrees)
    log.info(f'BEFORE: {target_joint} = {original:.4f}, target = {target_val:.4f}')

    # Build command: all joints at current position, except target_joint
    cmd = dict(before)
    cmd[target_joint] = target_val

    # Publish for 3 seconds at 20 Hz (same as tuck)
    log.info('Publishing commands for 3 seconds at 20 Hz (tuck-style)...')
    hz = 20
    duration = 3.0
    iterations = int(hz * duration)
    period = 1.0 / hz

    for i in range(iterations):
        t0 = time.time()
        rclpy.spin_once(node, timeout_sec=0.005)
        limb.set_joint_positions(cmd)
        elapsed = time.time() - t0
        if elapsed < period:
            time.sleep(period - elapsed)

        if i == iterations // 2:
            mid = limb.joint_angle(target_joint)
            log.info(f'  @{i}/{iterations}: {target_joint} = {mid:.4f} (moved={abs(mid - original) > 0.005})')

    after_val = limb.joint_angle(target_joint)
    delta = abs(after_val - original)
    log.info(f'AFTER: {target_joint} = {after_val:.4f}, delta = {delta:.4f}')
    if delta > 0.01:
        log.info('*** SUCCESS: ARM MOVED ***')
    else:
        log.warn('*** FAIL: ARM DID NOT MOVE ***')

    # Return to original position
    log.info('Returning to original position...')
    cmd[target_joint] = original
    for _ in range(40):  # 2 seconds
        rclpy.spin_once(node, timeout_sec=0.005)
        limb.set_joint_positions(cmd)
        time.sleep(period)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

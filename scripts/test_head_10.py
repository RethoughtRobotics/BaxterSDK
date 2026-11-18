#!/usr/bin/env python3
"""Move the head to 10 different poses to prove it works."""

import rclpy

import baxter_interface

POSES = [-1.2, 0.8, -0.5, 1.0, -1.0, 0.3, -0.8, 1.2, 0.0, -0.6]

rclpy.init()
node = rclpy.create_node('head_10_test')

rs = baxter_interface.RobotEnable(node)
rs.enable(node)
head = baxter_interface.Head(node)

for i, target in enumerate(POSES):
    print(f'[{i + 1}/10] -> {target:+.1f} rad  (now {head.pan():+.4f})')
    head.set_pan(target, speed=0.3, timeout=10.0)
    print(f'         reached {head.pan():+.4f}')

print('Done: 10 poses.')
rclpy.shutdown()

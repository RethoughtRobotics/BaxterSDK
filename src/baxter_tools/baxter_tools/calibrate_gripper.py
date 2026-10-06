#!/usr/bin/env python3

# Copyright (c) 2013-2015, Rethink Robotics
# All rights reserved.

import argparse
import sys

import rclpy

from baxter_interface import Gripper


def main(args=None):
    parser = argparse.ArgumentParser(description='Calibrate Baxter gripper')
    parser.add_argument(
        '-g',
        '--gripper',
        dest='gripper',
        choices=['left', 'right', 'both'],
        default='both',
        help='Gripper(s) to calibrate {left, right, both} (default: both)',
    )

    rclpy.init(args=args)
    parsed_args, _ = parser.parse_known_args()

    node = rclpy.create_node('rsdk_calibrate_gripper')
    logger = node.get_logger()

    grippers_to_calibrate = []
    if parsed_args.gripper == 'both':
        grippers_to_calibrate = ['left', 'right']
    else:
        grippers_to_calibrate = [parsed_args.gripper]

    results = {}
    for gripper_name in grippers_to_calibrate:
        try:
            logger.info(f'Calibrating {gripper_name} gripper...')
            gripper = Gripper(gripper_name, node=node)

            # Check current state
            logger.info(
                f'{gripper_name}: calibrated={gripper.calibrated()}, ready={gripper.ready()}, error={gripper.error()}'
            )

            # Attempt calibration
            success = gripper.calibrate(block=True, timeout=10.0)

            # Verify result
            calibrated_after = gripper.calibrated()
            logger.info(
                f'{gripper_name}: Calibration {"SUCCESSFUL" if success else "FAILED"} - calibrated={calibrated_after}'
            )
            results[gripper_name] = success and calibrated_after
        except Exception as e:
            logger.error(f'Error calibrating {gripper_name}: {e}')
            results[gripper_name] = False

    logger.info('\n=== Calibration Summary ===')
    for gripper_name, success in results.items():
        status = '✓ SUCCESS' if success else '✗ FAILED'
        logger.info(f'{gripper_name}: {status}')

    rclpy.shutdown()

    # Return non-zero exit code if any calibration failed
    sys.exit(0 if all(results.values()) else 1)


if __name__ == '__main__':
    main()

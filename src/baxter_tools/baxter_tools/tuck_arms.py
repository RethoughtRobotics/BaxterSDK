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
Tool to tuck/untuck Baxter's arms to/from the shipping pose
"""

import argparse
import time
from copy import deepcopy

import rclpy
from baxter_core_msgs.msg import CollisionAvoidanceState
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Empty

import baxter_interface


class Tuck(object):
    def __init__(self, node, tuck_cmd, arm=None):
        self._node = node
        self._done = False
        self._arm_filter = arm  # None means both, 'left' or 'right' means only that arm
        self._limbs = (arm,) if arm else ('left', 'right')
        self._arms = {
            'left': baxter_interface.Limb('left', node),
            'right': baxter_interface.Limb('right', node),
        }
        self._tuck = tuck_cmd
        self._period = 1.0 / 20.0  # Exactly 20 Hz
        self._tuck_threshold = 0.2  # radians
        self._peak_angle = -1.6  # radians
        self._arm_state = {
            'tuck': {'left': 'none', 'right': 'none'},
            'collide': {'left': False, 'right': False},
            'flipped': {'left': False, 'right': False},
        }
        self._joint_moves = {
            'tuck': {
                'left': [-1.0, -2.07, 3.0, 2.55, 0.0, 0.01, 0.0],
                'right': [1.0, -2.07, -3.0, 2.55, -0.0, 0.01, 0.0],
            },
            'untuck': baxter_interface.settings.UNTUCK_POSITIONS,
        }

        self._collide_lsub = self._node.create_subscription(
            CollisionAvoidanceState,
            'robot/limb/left/collision_avoidance_state',
            lambda msg: self._update_collision(msg, 'left'),
            10,
        )
        self._collide_rsub = self._node.create_subscription(
            CollisionAvoidanceState,
            'robot/limb/right/collision_avoidance_state',
            lambda msg: self._update_collision(msg, 'right'),
            10,
        )

        # Strict QoS to push through ros1_bridge at high frequency
        # depth=1 prevents the bridge from batching Empty messages
        bridge_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1)

        self._disable_pub = {
            'left': self._node.create_publisher(Empty, 'robot/limb/left/suppress_collision_avoidance', bridge_qos),
            'right': self._node.create_publisher(Empty, 'robot/limb/right/suppress_collision_avoidance', bridge_qos),
        }
        self._rs = baxter_interface.RobotEnable(self._node)
        self._enable_pub = self._node.create_publisher(Bool, 'robot/set_super_enable', bridge_qos)

    def _update_collision(self, data, limb):
        self._arm_state['collide'][limb] = len(data.collision_object) > 0
        self._check_arm_state()

    def _check_arm_state(self):
        def diff_check(a, b):
            return abs(a - b) <= self._tuck_threshold

        for limb in self._limbs:
            joint_names = [limb + '_' + j for j in ['s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2']]
            angles = [self._arms[limb].joint_angle(joint) for joint in joint_names]

            untuck_goal = list(map(diff_check, angles, self._joint_moves['untuck'][limb]))
            tuck_goal = list(map(diff_check, angles[0:2], self._joint_moves['tuck'][limb][0:2]))

            if all(untuck_goal):
                self._arm_state['tuck'][limb] = 'untuck'
            elif all(tuck_goal):
                self._arm_state['tuck'][limb] = 'tuck'
            else:
                self._arm_state['tuck'][limb] = 'none'

            self._arm_state['flipped'][limb] = self._arms[limb].joint_angle(limb + '_s1') <= self._peak_angle

    def _enforce_rate(self, start_time):
        elapsed = time.time() - start_time
        if elapsed < self._period:
            time.sleep(self._period - elapsed)

    def _prepare_to_tuck(self):
        head = baxter_interface.Head(self._node)
        start_disabled = not self._rs.state().enabled

        def at_goal():
            return abs(head.pan()) <= baxter_interface.settings.HEAD_PAN_ANGLE_TOLERANCE

        self._node.get_logger().info('Moving head to neutral position')
        while not at_goal() and rclpy.ok():
            loop_start = time.time()
            rclpy.spin_once(self._node, timeout_sec=0.005)

            if start_disabled:
                [pub.publish(Empty()) for pub in self._disable_pub.values()]
            if not self._rs.state().enabled:
                self._enable_pub.publish(Bool(data=True))

            head.set_pan(0.0, 0.5, timeout=0)
            self._enforce_rate(loop_start)

        if start_disabled:
            while self._rs.state().enabled and rclpy.ok():
                loop_start = time.time()
                rclpy.spin_once(self._node, timeout_sec=0.005)

                [pub.publish(Empty()) for pub in self._disable_pub.values()]
                self._enable_pub.publish(Bool(data=False))
                self._enforce_rate(loop_start)

    def _move_to(self, tuck, disabled):
        if any(disabled.values()):
            [pub.publish(Empty()) for pub in self._disable_pub.values()]

        while any(self._arm_state['tuck'][limb] != goal for limb, goal in tuck.items()) and rclpy.ok():
            loop_start = time.time()

            rclpy.spin_once(self._node, timeout_sec=0.005)

            if not self._rs.state().enabled:
                self._enable_pub.publish(Bool(data=True))

            for limb in self._limbs:
                if disabled[limb]:
                    self._disable_pub[limb].publish(Empty())
                if limb in tuck:
                    joint_names = [limb + '_' + j for j in ['s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2']]
                    self._arms[limb].set_joint_positions(dict(zip(joint_names, self._joint_moves[tuck[limb]][limb])))

            self._check_arm_state()
            self._enforce_rate(loop_start)

        if any(self._arm_state['collide'].values()):
            self._rs.disable(self._node)
        return

    def supervised_tuck(self):
        self._prepare_to_tuck()
        self._check_arm_state()

        if self._tuck:
            if all(self._arm_state['tuck'][limb] == 'tuck' for limb in self._limbs):
                self._node.get_logger().info("Tucking: Arms already in 'Tucked' position.")
                self._done = True
                return
            else:
                self._node.get_logger().info('Tucking: One or more arms not Tucked.')
                any_flipped = not all(self._arm_state['flipped'].values())
                if any_flipped:
                    self._node.get_logger().info(
                        f'Moving to neutral start position with collision {"on" if any_flipped else "off"}.'
                    )

                self._check_arm_state()
                actions = dict()
                disabled = {'left': True, 'right': True}
                for limb in self._limbs:
                    if not self._arm_state['flipped'][limb]:
                        actions[limb] = 'untuck'
                        disabled[limb] = False
                self._move_to(actions, disabled)

                self._node.get_logger().info('Tucking: Tucking with collision avoidance off.')
                actions = {'left': 'tuck', 'right': 'tuck'}
                disabled = {'left': True, 'right': True}
                self._move_to(actions, disabled)
                self._done = True
                return

        else:
            if any(self._arm_state['flipped'].values()):
                self._node.get_logger().info(
                    'Untucking: One or more arms Tucked; Disabling Collision Avoidance and untucking.'
                )
                self._check_arm_state()
                suppress = deepcopy(self._arm_state['flipped'])
                actions = {'left': 'untuck', 'right': 'untuck'}
                self._move_to(actions, suppress)
                self._done = True
                return
            else:
                self._node.get_logger().info('Untucking: Arms already Untucked; Moving to neutral position.')
                self._check_arm_state()
                suppress = deepcopy(self._arm_state['flipped'])
                actions = {'left': 'untuck', 'right': 'untuck'}
                self._move_to(actions, suppress)
                self._done = True
                return

    def clean_shutdown(self):
        if not self._done:
            self._node.get_logger().warn('Aborting: Shutting down safely...')
        if any(self._arm_state['collide'].values()):
            while self._rs.state().enabled and rclpy.ok():
                loop_start = time.time()
                rclpy.spin_once(self._node, timeout_sec=0.005)
                [pub.publish(Empty()) for pub in self._disable_pub.values()]
                self._enable_pub.publish(Bool(data=False))
                self._enforce_rate(loop_start)


def main(args=None):
    parser = argparse.ArgumentParser()
    tuck_group = parser.add_mutually_exclusive_group(required=True)
    tuck_group.add_argument(
        '-t',
        '--tuck',
        dest='tuck',
        nargs='?',
        const='both',
        choices=['left', 'right', 'both'],
        default=None,
        help='tuck arms {left, right, both} (default: both)',
    )
    tuck_group.add_argument(
        '-u',
        '--untuck',
        dest='untuck',
        nargs='?',
        const='both',
        choices=['left', 'right', 'both'],
        default=None,
        help='untuck arms {left, right, both} (default: both)',
    )

    rclpy.init(args=args)
    parsed_args, _ = parser.parse_known_args()
    tuck = parsed_args.tuck is not None
    arm = parsed_args.tuck if parsed_args.tuck else parsed_args.untuck

    node = rclpy.create_node('rsdk_tuck_arms')
    node.get_logger().info('Initializing node... ')
    node.get_logger().info('%sucking %s' % ('T' if tuck else 'Unt', 'both arms' if arm == 'both' else f'{arm} arm'))

    tucker = Tuck(node, tuck, None if arm == 'both' else arm)

    try:
        tucker.supervised_tuck()
        node.get_logger().info('Finished tuck')
    except KeyboardInterrupt:
        pass
    finally:
        tucker.clean_shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

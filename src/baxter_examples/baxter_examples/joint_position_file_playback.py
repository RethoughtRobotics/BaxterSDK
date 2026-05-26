#!/usr/bin/env python

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
Baxter RSDK Joint Position Example: file playback
"""

from __future__ import print_function

import argparse
import sys

import rclpy
from rclpy.node import Node

import baxter_interface


class JointPosePlayback:
    def __init__(self, node: Node):
        self._node = node
        self._frames = []
        self._frame_idx = 0
        self._start_time = None
        self._loop_count = 0
        self._loops = 1
        self._left = None
        self._right = None
        self._grip_left = None
        self._grip_right = None
        self._last_lg = None
        self._last_rg = None
        self._timer = None

    @staticmethod
    def try_float(x):
        try:
            return float(x)
        except ValueError:
            return None

    def clean_line(self, line, names):
        line = [self.try_float(x) for x in line.rstrip().split(',')]
        combined = zip(names[1:], line[1:])
        cleaned = [x for x in combined if x[1] is not None]
        command = dict(cleaned)
        left_command = dict((key, command[key]) for key in command.keys() if key[:-2] == 'left_')
        right_command = dict((key, command[key]) for key in command.keys() if key[:-2] == 'right_')
        return (command, left_command, right_command, line)

    def map_file(self, filename, loops=1):
        left = baxter_interface.Limb('left', node=self._node)
        right = baxter_interface.Limb('right', node=self._node)
        grip_left = baxter_interface.Gripper('left', node=self._node)
        grip_right = baxter_interface.Gripper('right', node=self._node)

        for grip, name in [(grip_left, 'left'), (grip_right, 'right')]:
            if grip.type() == 'custom':
                continue
            print('Calibrating %s gripper...' % name)
            if not grip.calibrate():
                print('Failed to calibrate %s gripper.' % name)
                return False

        print('Playing back: %s' % (filename,))
        with open(filename, 'r') as f:
            lines = f.readlines()
        keys = lines[0].rstrip().split(',')

        frames = []
        for line in lines[1:]:
            cmd, lcmd, rcmd, values = self.clean_line(line, keys)
            frames.append((values[0], cmd, lcmd, rcmd))

        print('Moving to start position...')
        left.move_to_joint_positions(frames[0][2])
        right.move_to_joint_positions(frames[0][3])

        # Re-send CMD_CONFIGURE now that the bridge is established.
        # The CMD_CONFIGURE sent in Gripper.__init__ is lost because DDS
        # discovery hasn't completed yet at that point, so the gripper
        # controller never gets configured and silently drops CMD_GO commands.
        grip_left.set_parameters(defaults=True)
        grip_right.set_parameters(defaults=True)

        # Diagnostic: verify CMD_GO works right NOW (before joint commands start).
        # If this moves the gripper, joint commands during _tick are the problem.
        # If this also fails, the issue is something earlier (bridge state, firmware).
        print('DIAG: sending blocking CMD_GO to left gripper (position=50)...')
        ok = grip_left.command_position(50.0, block=True, timeout=4.0)
        print('DIAG: CMD_GO result: %s  actual position: %.2f' % (ok, grip_left.position()))

        self._left = left
        self._right = right
        self._grip_left = grip_left
        self._grip_right = grip_right
        self._frames = frames
        self._loops = loops
        self._loop_count = 0
        self._frame_idx = 0
        self._start_time = self._node.get_clock().now().nanoseconds / 1e9

        self._timer = self._node.create_timer(0.01, self._tick)
        return True

    def _tick(self):
        now = self._node.get_clock().now().nanoseconds / 1e9
        elapsed = now - self._start_time

        # Advance to the latest frame whose timestamp has been reached
        while self._frame_idx + 1 < len(self._frames) and elapsed >= self._frames[self._frame_idx + 1][0]:
            self._frame_idx += 1

        _ts, cmd, lcmd, rcmd = self._frames[self._frame_idx]

        if len(lcmd):
            self._left.set_joint_positions(lcmd)
        if len(rcmd):
            self._right.set_joint_positions(rcmd)
        lg = cmd.get('left_gripper')
        if lg is not None and self._grip_left.type() != 'custom':
            if self._last_lg is None or abs(lg - self._last_lg) > 1.0:
                self._grip_left.command_position(lg)
                self._last_lg = lg
        rg = cmd.get('right_gripper')
        if rg is not None and self._grip_right.type() != 'custom':
            if self._last_rg is None or abs(rg - self._last_rg) > 1.0:
                self._grip_right.command_position(rg)
                self._last_rg = rg

        loopstr = str(self._loops) if self._loops > 0 else 'forever'
        sys.stdout.write(
            '\r Record %d of %d, loop %d of %s'
            % (self._frame_idx + 1, len(self._frames), self._loop_count + 1, loopstr)
        )
        sys.stdout.flush()

        if elapsed > self._frames[-1][0]:
            self._loop_count += 1
            print()
            if self._loops > 0 and self._loop_count >= self._loops:
                self._timer.cancel()
                rclpy.shutdown()
                return
            # Reset for next loop
            self._frame_idx = 0
            self._start_time = self._node.get_clock().now().nanoseconds / 1e9


def main():
    """RSDK Joint Position Example: File Playback

    Uses Joint Position Control mode to play back a series of
    recorded joint and gripper positions.

    Run the joint_recorder.py example first to create a recording
    file for use with this example. This example uses position
    control to replay the recorded positions in sequence.

    Note: This version of the playback example simply drives the
    joints towards the next position at each time stamp. Because
    it uses Position Control it will not attempt to adjust the
    movement speed to hit set points "on time".
    """
    epilog = """
Related examples:
  joint_recorder.py; joint_trajectory_file_playback.py.
    """
    arg_fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=arg_fmt, description=main.__doc__, epilog=epilog)
    parser.add_argument('-f', '--file', metavar='PATH', required=True, help='path to input file')
    parser.add_argument(
        '-l', '--loops', type=int, default=1, help='number of times to loop the input file. 0=infinite.'
    )
    args = parser.parse_args()

    print('Initializing node... ')
    rclpy.init()
    node = rclpy.create_node('rsdk_joint_position_file_playback')
    print('Getting robot state... ')
    rs = baxter_interface.RobotEnable(node)
    init_state = rs.state().enabled

    def clean_shutdown():
        print('\nExiting example...')
        if not init_state:
            print('Disabling robot...')
            rs.disable(node)

    rclpy.get_default_context().on_shutdown(clean_shutdown)

    print('Enabling robot... ')
    rs.enable(node)

    playback = JointPosePlayback(node)
    if playback.map_file(args.file, args.loops):
        rclpy.spin(node)


if __name__ == '__main__':
    main()

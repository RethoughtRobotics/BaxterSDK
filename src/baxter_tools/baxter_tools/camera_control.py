#!/usr/bin/python3

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

import argparse
import sys

import rclpy
import rclpy.node as Node
import std_srvs.srv
from baxter_core_msgs.srv import (
    ListCameras,
)
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from baxter_interface.camera import CameraController


class Camera(object):
    def __init__(self, node: Node, name: str = 'camera_control'):
        self._node = node
        self._name = name

        self.list_cameras_cb_group = MutuallyExclusiveCallbackGroup()
        self.reset_cb_group = MutuallyExclusiveCallbackGroup()

    def list_cameras(self, *_args, **_kwds):
        ls = self._node.create_client(ListCameras, 'cameras/list', callback_group=self.list_cameras_cb_group)

        while not ls.wait_for_service(timeout_sec=10):
            self._node.get_logger().warn('CameraController: Waiting for service /cameras/list to become available...')
            rclpy.spin_once(self._node, timeout_sec=0.1)

        future = ls.call_async(ListCameras.Request())
        rclpy.spin_until_future_complete(self._node, future)
        resp = future.result()
        if len(resp.cameras):
            cam_topics = {cam: f'/cameras/{cam}/image' for cam in resp.cameras}
            open_cams = {cam: False for cam in resp.cameras}
            topics = self._node.get_topic_names_and_types()
            for topic_name, _ in topics:
                for cam in resp.cameras:
                    if topic_name == cam_topics[cam]:
                        open_cams[cam] = True
            for cam in resp.cameras:
                print('%s%s' % (cam, ('  -  (open)' if open_cams[cam] else '')))
        else:
            print('No cameras found')

    def reset_cameras(self, *_args, **_kwds):
        reset_srv = self._node.create_client(std_srvs.srv.Empty, 'cameras/reset', callback_group=self.reset_cb_group)
        while not reset_srv.wait_for_service(timeout_sec=10):
            self._node.get_logger().warn('CameraController: Waiting for service /cameras/reset to become available...')
        future = reset_srv.call_async(std_srvs.srv.Empty.Request())
        rclpy.spin_until_future_complete(self._node, future)

    def enum_cameras(self, *_args, **_kwds):
        try:
            self.reset_cameras()
        except:
            srv_ns = 'cameras/reset'
            self._node.get_logger().error(f'Failed to call reset devices service at {srv_ns}')
            raise
        else:
            self.list_cameras()

    def open_camera(self, camera, res, fps=None, exposure=None, gain=None, white_balance=None):
        """Open (or re-open) a camera; settings left as None keep the camera's automatic control."""
        cam = CameraController(self._node, camera)
        cam.resolution = res
        if fps is not None:
            cam.fps = fps
        if exposure is not None:
            cam.exposure = exposure
        if gain is not None:
            cam.gain = gain
        if white_balance is not None:
            cam.white_balance_red, cam.white_balance_green, cam.white_balance_blue = white_balance
        cam.open()

    def close_camera(self, camera, *_args, **_kwds):
        cam = CameraController(self._node, camera)
        cam.close()


def control(value):
    """A camera control value: an int, or 'auto' for the camera's automatic control."""
    return CameraController.CONTROL_AUTO if value == 'auto' else int(value)


def main():
    str_res = ['%rx%r' % (r[0], r[1]) for r in CameraController.MODES]
    fmt_res = 'Valid resolutions:\n[' + ('%s, ' * (len(CameraController.MODES) - 1)) + '%s]'
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=(fmt_res % tuple(str_res))
    )
    action_grp = parser.add_mutually_exclusive_group(required=True)
    action_grp.add_argument('-o', '--open', metavar='CAMERA', help='Open specified camera')
    parser.add_argument(
        '-r', '--resolution', metavar='[X]x[Y]', default='1280x800', help='Set camera resolution (default: 1280x800)'
    )
    # Applied with --open. Each re-opens the camera, so pass the resolution and fps it should keep.
    parser.add_argument('--fps', type=float, help='Frame rate (default: 20)')
    parser.add_argument('--exposure', type=control, help='Exposure 0-100, or auto (default)')
    parser.add_argument('--gain', type=control, help='Gain 0-79, or auto (default)')
    parser.add_argument(
        '--white-balance', type=control, nargs=3, metavar=('R', 'G', 'B'), help='White balance 0-4095 each, or auto'
    )
    action_grp.add_argument('-c', '--close', metavar='CAMERA', help='Close specified camera')
    action_grp.add_argument('-l', '--list', action='store_true', help='List available cameras')
    action_grp.add_argument('-e', '--enumerate', action='store_true', help='Clear and re-enumerate connected devices')
    args = parser.parse_args()

    cam_name = None
    res = (1280, 800)

    if args.open:
        cam_name = args.open
        lres = args.resolution.split('x')
        if len(lres) != 2:
            print(fmt_res % tuple(str_res))
            parser.error('Invalid resolution format: %s. Use (X)x(Y).')
        res = (int(lres[0]), int(lres[1]))
        if not any((res[0] == r[0] and res[1] == r[1]) for r in CameraController.MODES):
            print(fmt_res % tuple(str_res))
            parser.error('Invalid resolution provided.')
    elif args.close:
        cam_name = args.close

    rclpy.init()
    node = rclpy.create_node('rsdk_camera_control')
    cam = Camera(name='camera', node=node)

    if args.list:
        cam.list_cameras()
    elif args.open:
        cam.open_camera(cam_name, res, args.fps, args.exposure, args.gain, args.white_balance)
    elif args.close:
        cam.close_camera(cam_name)
    elif args.enumerate:
        cam.enum_cameras()

    return 0


if __name__ == '__main__':
    sys.exit(main())

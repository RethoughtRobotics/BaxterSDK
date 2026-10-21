"""The simulated Baxter's cameras, rendered in their own process (launch_sim.sh --ros --cameras).

  subscribes  /sim/state (the sim's MuJoCo qpos, stamped, 25 Hz), /sim/lighting
  publishes   /cameras/{arm}_hand_camera/image (640 x 400 bgra8, as the robot) and, in the cup
              scene, the lab's third_eye (lab.toml), each with its camera_info

The sim (baxter_ros) only streams its state: rendering three cameras inside it starved its physics
loop of Python's GIL, so the arms ran at 83 % of real time. This process builds the same scene,
poses it from each state message and renders the cameras someone subscribes to, through EGL, so it
also runs beside the sim's viewer (GLX offscreen contexts beside a viewer crash Xorg on driver 535).
"""

import argparse
import os

import mujoco
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Float64MultiArray

from baxter_mujoco import scenes
from baxter_mujoco.sim import make_env

ARMS = ('left', 'right')
# The hand cameras as mounted on the robot (baxter.urdf {arm}_hand_camera, in {arm}_hand): looking out
# along the fingers, which are at the image top; field of view from their camera_info (fy 403.9 px of
# 800 rows).
HAND_CAMERA_POS = (0.03825, 0.012, 0.015355)  # m
HAND_CAMERA_AXES = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]).T  # columns: image right, up, back
HAND_CAMERA_FOVY = 89.4  # deg
HAND_CAMERA_SIZE = (640, 400)  # px


class Camera:
    """A camera's image and camera_info publishers, as the robot and its RealSense name them, and the
    geoms hidden from it."""

    def __init__(self, node, name, topic, frame_id, size, focal, principal, encoding, hidden):
        self.name, self.encoding, self.hidden, self.renderer = name, encoding, hidden, None
        self.pub = node.create_publisher(Image, topic, 5)
        self.info_pub = node.create_publisher(CameraInfo, topic.rsplit('/', 1)[0] + '/camera_info', 5)
        (w, h), (fx, fy), (cx, cy) = size, focal, principal
        self.info = CameraInfo(width=int(w), height=int(h), distortion_model='plumb_bob', d=[0.0] * 5)
        self.info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        self.info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        self.info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        self.info.header.frame_id = frame_id

    def publish(self, m, d, stamp, option):
        """Publish camera_info and, while someone subscribes, the rendered image."""
        self.info.header.stamp = stamp
        self.info_pub.publish(self.info)
        if not self.pub.get_subscription_count():
            return
        w, h = self.info.width, self.info.height
        if self.renderer is None:
            self.renderer = mujoco.Renderer(m, h, w)
        alpha = m.geom_rgba[self.hidden, 3].copy()
        m.geom_rgba[self.hidden, 3] = 0.0
        self.renderer.update_scene(d, camera=self.name, scene_option=option)
        m.geom_rgba[self.hidden, 3] = alpha
        rgb = self.renderer.render()
        data = np.dstack([rgb[..., ::-1], np.full((h, w), 255, np.uint8)]) if self.encoding == 'bgra8' else rgb
        msg = Image(height=h, width=w, encoding=self.encoding, step=w * data.shape[2], data=data.tobytes())
        msg.header.stamp, msg.header.frame_id = stamp, self.info.header.frame_id
        self.pub.publish(msg)


class SimCameras:
    """The robot's hand cameras and, in the cup scene, the lab's third_eye, posed from /sim/state."""

    def __init__(self, node, scene='lift', table=True):
        self.env = make_env(render=False)
        self.env.reset()
        if scene == 'cup':
            self.env = scenes.build_cup_scene(self.env)
        m = self.m = self.env.sim.model._model
        self.d = self.env.sim.data._data
        if not table:  # as the sim: the table in geom group 5, which cameras do not draw
            body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'table')
            m.geom_group[np.flatnonzero(m.geom_bodyid == body)] = 5
        self.option = mujoco.MjvOption()
        self.option.geomgroup[0] = 0  # robosuite's collision geoms, which its viewer hides too
        self.cameras = []
        w, h = HAND_CAMERA_SIZE
        f = h / 2 / np.tan(np.radians(HAND_CAMERA_FOVY) / 2)
        for side in ARMS:
            cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, f'robot0_{side}_eye_in_hand')
            m.cam_pos[cam] = HAND_CAMERA_POS
            m.cam_quat[cam] = Rotation.from_matrix(HAND_CAMERA_AXES).as_quat(scalar_first=True)
            m.cam_fovy[cam] = HAND_CAMERA_FOVY
            # robosuite's gripper housing is wider than the real one, which the camera sees past
            housing = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f'gripper0_{side}_gripper_base')
            self.cameras.append(
                Camera(
                    node, f'robot0_{side}_eye_in_hand', f'/cameras/{side}_hand_camera/image', f'{side}_hand_camera',
                    (w, h), (f, f), (w / 2, h / 2), 'bgra8', np.flatnonzero(m.geom_bodyid == housing),
                )
            )  # fmt: skip
        if scene == 'cup':
            eye = scenes.LAB['third_eye']
            self.cameras.append(
                Camera(
                    node, 'third_eye', eye['topic'], eye['frame_id'], eye['size'], eye['focal'], eye['principal'],
                    eye['encoding'], [],
                )
            )  # fmt: skip
            node.create_subscription(
                Float64MultiArray, '/sim/lighting', lambda msg: scenes.apply_lighting(self.env, list(msg.data)), 10
            )
            node.create_subscription(
                Float64MultiArray, '/sim/cup_color', lambda msg: scenes.apply_cup_color(self.env, list(msg.data)), 10
            )
        node.create_subscription(JointState, '/sim/state', self.on_state, 1)

    def on_state(self, msg):
        m, d = self.m, self.d
        d.qpos[:] = msg.position
        mujoco.mj_kinematics(m, d)  # body, camera and light poses from qpos
        mujoco.mj_comPos(m, d)
        mujoco.mj_camlight(m, d)
        for camera in self.cameras:
            camera.publish(m, d, msg.header.stamp, self.option)


def main():
    parser = argparse.ArgumentParser(description="The simulated Baxter's cameras")
    parser.add_argument('--scene', choices=('lift', 'cup'), default='lift')
    parser.add_argument('--no-table', action='store_true')
    args, ros_args = parser.parse_known_args()
    if os.environ.get('MUJOCO_GL') != 'egl':
        parser.error('render through EGL: set MUJOCO_GL=egl (launch_sim.sh does)')
    rclpy.init(args=['sim_cameras', *ros_args])
    node = rclpy.create_node('sim_cameras')
    SimCameras(node, args.scene, table=not args.no_table)
    print('Simulated cameras up.', flush=True)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == '__main__':
    main()

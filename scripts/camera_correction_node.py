#!/usr/bin/env python3
"""ROS2 node: subscribe to wrist camera images, apply fixed corrections, republish.

Reads camera_corrections.yaml (from calibrate_wrist_cameras.py) and republishes
processed images on /cameras/{name}/image/processed. View in rqt_image_view.

    ros2 run --prefix 'python3' baxter_tools camera_correction_node.py
    # or just:
    python3 scripts/camera_correction_node.py
"""

from pathlib import Path

import cv2
import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from sensor_msgs.msg import Image

CAMERAS = ('left_hand_camera', 'right_hand_camera')
YAML_PATH = Path(__file__).resolve().parent / 'camera_corrections.yaml'


class CorrectionNode(Node):
    def __init__(self):
        super().__init__('camera_corrections')
        if not YAML_PATH.exists():
            self.get_logger().error(f'No corrections file at {YAML_PATH}; run calibrate_wrist_cameras.py first')
            raise SystemExit(1)

        with open(YAML_PATH) as f:
            data = yaml.safe_load(f)

        self._corrections = {}
        self._pubs = {}
        for cam in CAMERAS:
            if cam not in data:
                continue
            entry = data[cam]
            vig = None
            vig_path = YAML_PATH.parent / entry.get('vignetting_file', '')
            if vig_path.exists():
                vig = np.load(vig_path)
            self._corrections[cam] = {
                'vignetting': vig,
                'wb': entry['wb_gains_bgr'],
                'gamma': entry['gamma'],
                'lut': np.array([((i / 255.0) ** (1.0 / entry['gamma'])) * 255 for i in range(256)], dtype=np.uint8),
            }
            self._pubs[cam] = self.create_publisher(Image, f'/cameras/{cam}/image/processed', 1)
            self.create_subscription(Image, f'/cameras/{cam}/image', lambda msg, c=cam: self._on_image(c, msg), 1)
            self.get_logger().info(f'{cam}: wb={entry["wb_gains_bgr"]}, gamma={entry["gamma"]}')

    def _on_image(self, cam, msg):
        corr = self._corrections[cam]
        h, w = msg.height, msg.width
        raw = np.frombuffer(msg.data, dtype=np.uint8)

        if msg.encoding == 'bgra8':
            bgr = raw.reshape(h, w, 4)[:, :, :3].copy()
        elif msg.encoding == 'rgb8':
            bgr = cv2.cvtColor(raw.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
        else:
            bgr = raw.reshape(h, w, 3).copy()

        img = bgr.astype(np.float32)
        if corr['vignetting'] is not None:
            vig = corr['vignetting']
            if vig.shape != (h, w):
                vig = cv2.resize(vig, (w, h))
            for c in range(3):
                img[:, :, c] *= vig
        img[:, :, 0] *= corr['wb'][0]
        img[:, :, 2] *= corr['wb'][2]
        np.clip(img, 0, 255, out=img)
        result = cv2.LUT(img.astype(np.uint8), corr['lut'])

        out = Image()
        out.header = msg.header
        out.height = h
        out.width = w
        out.encoding = 'bgr8'
        out.step = w * 3
        out.data = result.tobytes()
        self._pubs[cam].publish(out)


def main():
    rclpy.init()
    node = CorrectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

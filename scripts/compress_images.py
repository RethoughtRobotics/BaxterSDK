#!/usr/bin/env python3
"""JPEG copies of camera topics on <topic>/compressed (sensor_msgs/CompressedImage), for rollout
recordings: raw images are ~3 GB a minute. Stands in for image_transport's compressed plugin,
which is not installed (apt: ros-kilted-image-transport-plugins).
Usage: compress_images.py TOPIC [TOPIC ...]   (Ctrl+C stops)"""

import sys

import cv2
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image

QUALITY = 90
CHANNELS = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4}


def main():
    rclpy.init()
    node = rclpy.create_node('compress_images')
    for topic in sys.argv[1:]:
        pub = node.create_publisher(CompressedImage, topic + '/compressed', 10)

        def on_image(msg, pub=pub):
            a = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.width, CHANNELS[msg.encoding])[..., :3]
            bgr = a if msg.encoding.startswith('bgr') else a[..., ::-1]
            ok, jpeg = cv2.imencode('.jpg', np.ascontiguousarray(bgr), [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
            if ok:
                pub.publish(
                    CompressedImage(header=msg.header, format='bgr8; jpeg compressed bgr8', data=jpeg.tobytes())
                )

        node.create_subscription(Image, topic, on_image, qos_profile_sensor_data)
    try:
        rclpy.spin(node)
    except Exception:  # Ctrl+C: rclpy shuts the context down under spin
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()

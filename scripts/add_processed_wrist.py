#!/usr/bin/env python3
"""Add the processed wrist cameras to recorded bags that hold only the raw ones.

Policies see the wrist cameras through camera_correction_node.py (vignetting gain, white balance, gamma), on
/cameras/{camera}/image/processed. Bags recorded before record_teleop.sh kept that topic hold only the raw
JPEGs (/cameras/{camera}/image/compressed). This writes each raw frame through the same correct() into
/cameras/{camera}/image/processed/compressed, JPEG quality 90 as compress_images.py, with the raw frame's
header and receive time. The input is JPEG, where the live node corrects the uncompressed image, so a frame
can differ from a live one by the raw JPEG's noise, not in content.

    python3 scripts/add_processed_wrist.py BAG [BAG ...]      # each bag directory is rewritten in place
"""

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message, serialize_message
from sensor_msgs.msg import CompressedImage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from camera_correction_node import CAMERAS, correct, load_corrections  # noqa: E402

QUALITY = 90  # compress_images.py's


def add_processed(bag, corrections):
    """Rewrite `bag` with every raw wrist frame also on its processed topic. Returns frames added per camera."""
    bag = Path(bag)
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id='mcap'), rosbag2_py.ConverterOptions('', ''))
    topics = reader.get_all_topics_and_types()
    names = {t.name for t in topics}
    raw = {f'/cameras/{cam}/image/compressed': cam for cam in CAMERAS if f'/cameras/{cam}/image/compressed' in names}
    out_topic = {tp: f'/cameras/{cam}/image/processed/compressed' for tp, cam in raw.items()}
    if any(t in names for t in out_topic.values()):
        raise SystemExit(f'{bag}: already has processed wrist frames')
    tmp = bag.with_name(bag.name + '.processing')
    if tmp.exists():
        raise SystemExit(f'{tmp} exists from an interrupted run; remove it first')
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(tmp), storage_id='mcap'), rosbag2_py.ConverterOptions('', ''))
    for i, t in enumerate(topics):
        writer.create_topic(rosbag2_py.TopicMetadata(i, t.name, t.type, t.serialization_format))
    for j, tp in enumerate(out_topic.values()):
        writer.create_topic(rosbag2_py.TopicMetadata(len(topics) + j, tp, 'sensor_msgs/msg/CompressedImage', 'cdr'))
    added = dict.fromkeys(raw.values(), 0)
    while reader.has_next():
        topic, data, stamp = reader.read_next()
        writer.write(topic, data, stamp)
        if topic in raw:
            msg = deserialize_message(data, CompressedImage)
            bgr = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
            ok, jpeg = cv2.imencode('.jpg', correct(bgr, corrections[raw[topic]]), [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
            if not ok:
                raise RuntimeError(f'{bag}: JPEG encoding failed at {stamp}')
            out = CompressedImage(header=msg.header, format='bgr8; jpeg compressed bgr8', data=jpeg.tobytes())
            writer.write(out_topic[topic], serialize_message(out), stamp)
            added[raw[topic]] += 1
    del writer, reader  # close both files before the swap
    for f in bag.iterdir():  # keep the bag's provenance and other files beside the mcap
        if not f.name.endswith(('.mcap', 'metadata.yaml')):
            shutil.move(str(f), tmp / f.name)
    shutil.rmtree(bag)
    tmp.rename(bag)
    return added


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('bags', nargs='+', type=Path)
    corrections = load_corrections()
    for bag in parser.parse_args().bags:
        print(bag.name, add_processed(bag, corrections), flush=True)


if __name__ == '__main__':
    main()

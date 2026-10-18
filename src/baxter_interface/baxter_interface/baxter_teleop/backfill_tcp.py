"""Add /teleop/{arm}/tcp to episodes recorded before the teleop published it.

Each arm /robot/joint_states message gets a TCP pose (tcp.py, the teleop's own model) written at
the same log time and stamped with it, as the live topic is stamped with the host clock. Every
other message is copied unchanged. The rewritten bag is checked against the original (same
messages per topic, plus one TCP per arm state) before it replaces it; bags that already have
the topic are skipped.

  python -m baxter_interface.baxter_teleop.backfill_tcp --arm left --tcp-offset X Y Z EPISODE...
"""

import argparse
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import rosbag2_py
from builtin_interfaces.msg import Time
from rclpy.serialization import deserialize_message, serialize_message
from sensor_msgs.msg import JointState

from .tcp import pose_msg, tcp_pose
from .urdf_tools import extract_arm_chain_urdf, resolve_baxter_urdf

JOINTS = ('s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2')


def counts(uri):
    info = rosbag2_py.Info().read_metadata(str(uri), 'mcap')
    return {t.topic_metadata.name: t.message_count for t in info.topics_with_message_count}


def backfill(episode, arm, tcp_offset, ee_transform, preset):
    episode = Path(episode).resolve()
    topic = f'/teleop/{arm}/tcp'
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(episode), storage_id='mcap'), rosbag2_py.ConverterOptions('', ''))
    topics = reader.get_all_topics_and_types()
    if any(t.name == topic for t in topics):
        return f'{episode.name}: already has {topic}, skipped'

    staged = episode.parent / '.tcp_staging' / episode.name
    shutil.rmtree(staged, ignore_errors=True)
    staged.parent.mkdir(exist_ok=True)
    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(staged), storage_id='mcap', storage_preset_profile=preset),
        rosbag2_py.ConverterOptions('cdr', 'cdr'),
    )
    for t in topics:
        writer.create_topic(t)
    writer.create_topic(rosbag2_py.TopicMetadata(len(topics) + 1, topic, 'geometry_msgs/msg/PoseStamped', 'cdr'))

    names = [f'{arm}_{j}' for j in JOINTS]
    added = 0
    while reader.has_next():
        name, data, t = reader.read_next()
        writer.write(name, data, t)
        if name == '/robot/joint_states':
            m = deserialize_message(data, JointState)
            if names[0] in m.name:
                q = jnp.asarray([m.position[m.name.index(n)] for n in names], dtype=jnp.float32)
                position, rotation = tcp_pose(ee_transform(q), tcp_offset)
                stamp = Time(sec=t // 1_000_000_000, nanosec=t % 1_000_000_000)
                writer.write(topic, serialize_message(pose_msg(position, rotation, stamp)), t)
                added += 1
    del writer, reader  # close both bags (writes the metadata)

    before, after = counts(episode), counts(staged)
    expected = dict(before, **{topic: added})
    if after != expected or added == 0:
        shutil.rmtree(staged)
        return f'{episode.name}: CHECK FAILED, original kept ({before} vs {after})'
    old_mb = sum(f.stat().st_size for f in episode.iterdir()) // 1_000_000
    new_mb = sum(f.stat().st_size for f in staged.iterdir()) // 1_000_000
    shutil.rmtree(episode)
    staged.rename(episode)
    return f'{episode.name}: +{added} TCP poses, {old_mb} MB -> {new_mb} MB'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('episodes', nargs='+')
    parser.add_argument('--arm', choices=['left', 'right'], required=True)
    parser.add_argument(
        '--tcp-offset',
        type=float,
        nargs=3,
        required=True,
        metavar=('X', 'Y', 'Z'),
        help='TCP in the hand frame (m), as the teleop measures it at startup',
    )
    parser.add_argument('--preset', default='zstd_small', help='storage preset for the rewritten bag')
    args = parser.parse_args()

    import frax

    names = [f'{args.arm}_{j}' for j in JOINTS]
    robot = frax.core.manipulator.Manipulator(
        extract_arm_chain_urdf(resolve_baxter_urdf(), args.arm, names), joint_ordering=names
    )
    ee_transform = jax.jit(robot.ee_transform)
    tcp_offset = np.asarray(args.tcp_offset)
    for episode in args.episodes:
        print(backfill(episode, args.arm, tcp_offset, ee_transform, args.preset), flush=True)


if __name__ == '__main__':
    main()

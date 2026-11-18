"""A recorded attempt's camera frames rendered again with one scene factor changed: the scope benchmark
for a correction memory's key (does it ignore what a correction is not about, and tell apart what it is?).

From an evaluator bag with /sim/state (the full qpos) the frames are posed exactly as they ran and
rendered once per variant; every variant differs from the anchor in one factor only:

  anchor      as recorded: the attempt's cup colour and its seed's lighting (generate.conditions)
  container   the container elsewhere on the table, turned (as far from the cup as a layout allows)
  lighting    another draw of the lab's lighting (scenes.sample_lighting)
  distractor  a box on the table, clear of the cup and the arm's way to it
  blue        the cup blue, the colour the base was trained on

A factor is drawn once per attempt, seeded by the attempt's seed, so every frame of an attempt sees the
same changed scene. Frames are kept every --every frames while the cup has not been lifted (the reach
and grasp, where the container and the room are not what a correction is about).

    MUJOCO_GL=egl uv run --group ros python -m baxter_mujoco.datagen.variants BAG... --cup-color red

writes <bag>/variants.npz: t (N,) ns, the /sim/state stamps; per variant and camera
<variant>/<camera> (N, H, W, 3) uint8; layout and lighting of each variant (JSON in meta).
"""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from baxter_mujoco import scenes
from baxter_mujoco.cameras import Rig
from baxter_mujoco.datagen.generate import TASKS, conditions

VARIANTS = ('anchor', 'container', 'lighting', 'distractor', 'blue')
CAMERAS = ('left_hand', 'third_eye')  # the sim base's cameras
EVERY = 4  # frames: the base plans every 4
LIFT = 0.02  # m, the cup above where it stood: lifted (outcome.LIFT)
DISTRACTOR = (0.04, 0.04, 0.04)  # m, half sizes
DISTRACTOR_RGBA = '0.2 0.7 0.2 1'
CLEAR = 0.15  # m, the distractor's centre from the cup and from the line home to the cup


def read(bag):
    """(stamps (N,) ns, qpos (N, nq)) of the bag's /sim/state."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import JointState

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id='mcap'), rosbag2_py.ConverterOptions('', ''))
    reader.set_filter(rosbag2_py.StorageFilter(topics=['/sim/state']))
    stamps, qpos = [], []
    while reader.has_next():
        _, data, t = reader.read_next()
        stamps.append(t)
        qpos.append(deserialize_message(data, JointState).position)
    if not stamps:
        raise ValueError(f'{bag}: no /sim/state (bags from before the evaluator recorded it)')
    return np.asarray(stamps), np.asarray(qpos)


def with_distractor(rig):
    """The rig's scene with a static box `distractor` (no joint, so the qpos is unchanged), hidden
    until a variant shows it; returns its geom id."""
    root = ET.fromstring(rig.env.sim.model.get_xml())
    body = ET.SubElement(root.find('worldbody'), 'body', name='distractor', pos='0 0 0')  # geom_pos is world
    ET.SubElement(
        body, 'geom', name='distractor', type='box', pos='0 0 -1', size=' '.join(map(str, DISTRACTOR)),
        rgba=DISTRACTOR_RGBA, contype='0', conaffinity='0',
        group='1',  # cameras hide group 0, robosuite's collision geoms
    )  # fmt: skip
    rig.env.reset_from_xml_string(ET.tostring(root, encoding='unicode'))
    rig.m, rig.d = rig.env.sim.model._model, rig.env.sim.data._data
    rig.renderers = {}
    return mujoco.mj_name2id(rig.m, mujoco.mjtObj.mjOBJ_GEOM, 'distractor')


def joint_address(m, name):
    return m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f'{name}_joint')]


def to_world(rig, xy, z):
    pb, Rb = scenes.base_pose(rig.env)
    return pb + Rb @ np.array([xy[0], xy[1], z])


def draws(seed, cup_xy):
    """Per attempt: the container's (xy, yaw), the lighting, the distractor's xy, each its own stream."""
    rng = np.random.default_rng([seed, 7])
    far = max(
        ([rng.uniform(*scenes.CONTAINER_RANGE[0]), rng.uniform(*scenes.CONTAINER_RANGE[1])] for _ in range(200)),
        key=lambda xy: np.hypot(*(np.asarray(xy) - cup_xy)),
    )
    container = (np.asarray(far), float(rng.uniform(-np.pi, np.pi)))
    lighting = scenes.sample_lighting(np.random.default_rng([seed, 8]))
    rng = np.random.default_rng([seed, 9])
    home = np.array([0.6, 0.6])  # the left arm's side of the table, roughly where it starts
    while True:
        xy = np.array([rng.uniform(*scenes.CUP_RANGE[0]), rng.uniform(*scenes.CUP_RANGE[1])])
        along = np.clip(np.dot(xy - home, cup_xy - home) / np.dot(cup_xy - home, cup_xy - home), 0, 1)
        if np.hypot(*(xy - cup_xy)) > CLEAR and np.hypot(*(xy - home - along * (cup_xy - home))) > CLEAR:
            return container, lighting, xy


def render(bag, cup_color, every=EVERY, task='cup'):
    seed = int(Path(bag).name.rsplit('_', 1)[-1])
    stamps, qpos = read(bag)
    rig = Rig('cup')
    distractor = with_distractor(rig)
    m = rig.m
    cup, box = joint_address(m, 'cup'), joint_address(m, 'container')
    _, lighting = next(conditions(TASKS[task], seed))
    pb, Rb = scenes.base_pose(rig.env)
    cup_xy = (Rb.T @ (qpos[0, cup : cup + 3] - pb))[:2]
    (box_xy, box_yaw), other_lighting, distractor_xy = draws(seed, cup_xy)
    rest = qpos[0, cup + 2]
    keep = np.flatnonzero(np.cumsum(qpos[:, cup + 2] > rest + LIFT) == 0)[::every]
    colors = {name: [float(v) for v in rgba.split()] for name, rgba in scenes.CUP_COLORS.items()}
    out = {}
    for variant in VARIANTS:
        scenes.apply_lighting(rig.env, other_lighting if variant == 'lighting' else lighting)
        scenes.apply_cup_color(rig.env, colors['blue' if variant == 'blue' else cup_color])
        m.geom_pos[distractor] = (
            to_world(rig, distractor_xy, scenes.TABLE_TOP + DISTRACTOR[2]) if variant == 'distractor' else [0, 0, -1]
        )
        frames = {c: [] for c in CAMERAS}
        for i in keep:
            q = qpos[i].copy()
            if variant == 'container':
                q[box : box + 3] = to_world(rig, box_xy, (Rb.T @ (q[box : box + 3] - pb))[2])
                q[box + 3 : box + 7] = [np.cos(box_yaw / 2), 0, 0, np.sin(box_yaw / 2)]
            rig.pose(q)
            for c in CAMERAS:
                frames[c].append(rig.render(c))
        for c in CAMERAS:
            out[f'{variant}/{c}'] = np.stack(frames[c])
    meta = {
        'seed': seed,
        'cup_color': cup_color,
        'every': every,
        'frames': [int(i) for i in keep],
        'container': {'xy': [float(v) for v in box_xy], 'yaw': box_yaw},
        'lighting': {'anchor': lighting, 'lighting': other_lighting},
        'distractor': {'xy': [float(v) for v in distractor_xy]},
    }
    np.savez_compressed(Path(bag) / 'variants.npz', t=stamps[keep], meta=json.dumps(meta), **out)
    return len(keep)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('bags', nargs='+')
    parser.add_argument('--cup-color', choices=sorted(scenes.CUP_COLORS), required=True)
    parser.add_argument('--every', type=int, default=EVERY)
    args = parser.parse_args()
    for bag in args.bags:
        print(f'{bag}: {render(bag, args.cup_color, args.every)} frames', flush=True)


if __name__ == '__main__':
    main()

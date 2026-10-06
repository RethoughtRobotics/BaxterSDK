"""Replay recorded Baxter joint angles in the MuJoCo scene and list what the arms touched.

  uv run python -m baxflow.contacts <bag>/joints.npz   (written by bringbackbaxter's bag_report)

Kinematic replay: objects (the pot) sit at their start pose (or out of reach if the run used
--empty), so contacts with them are where the arm reached them, not what happened afterwards.
"""

import sys

import mujoco
import numpy as np

from baxflow.sim import make_env

TOUCH = 0.002  # m: closer than this counts as contact (recorded poses are where physics already pushed things apart)


def main(path):
    data = np.load(path)
    env = make_env()
    env.reset()
    m, d = env.sim.model._model, env.sim.data._data
    if 'empty' in data and bool(data['empty']):  # the run had objects moved out of reach (--empty)
        for j in np.flatnonzero(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE):
            d.qpos[m.jnt_qposadr[j]:m.jnt_qposadr[j] + 3] = [0.0, -3.0, 0.2]
    adr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f'robot0_{n}')] for n in data['names']]
    body = lambda geom: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[geom]) or '?'
    found = {}  # pair -> [first t, last t, samples, deepest m]
    for t, q in zip(data['t'][::5], data['q'][::5]):
        d.qpos[adr] = q
        mujoco.mj_forward(m, d)
        for c in d.contact[: d.ncon]:
            pair = tuple(sorted((body(c.geom1), body(c.geom2))))
            if c.dist > TOUCH or not any(s in ''.join(pair) for s in ('left', 'right')):
                continue
            hit = found.setdefault(pair, [t, t, 0, 0.0])
            hit[1], hit[2], hit[3] = t, hit[2] + 1, min(hit[3], c.dist)
    print('\nContacts (kinematic replay in the MuJoCo scene):' if found else '\nNo contacts.')
    for (a, b), (first, last, n, depth) in sorted(found.items(), key=lambda kv: kv[1][0]):
        print(f'  {a:28s} <-> {b:28s} {first:6.1f}-{last:6.1f} s  {n:4d} samples  closest {1e3 * depth:5.0f} mm')


if __name__ == '__main__':
    main(sys.argv[1])

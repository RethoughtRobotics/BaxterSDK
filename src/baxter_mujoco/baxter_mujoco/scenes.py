"""The cup scene: the BlueCupRedContainer task on a table at the real robot's height.

Measured on the 56 real demonstrations of BlueCupRedContainer_kin_pose (teleop TCP, robot base frame):
the gripper closes at z = -0.045 m (median, std 0.01) with its fingertips just above the tabletop in
the wrist camera, so the table top is put at z = -0.06 m, 6.4 cm above robosuite's default table. The
cup was grasped at x 0.46 to 0.95 m, y -0.20 to 0.44 m and released at x 0.42 to 0.94 m, y -0.28 to
0.36 m; the table is enlarged to cover both. The cup is 6 cm across, the widest the sim gripper's
fingertips (7.7 cm apart, open) close around with their pads, and 9 cm tall.

Poses are given in the robot base frame, as the teleop and the recorded datasets use it.
"""

import xml.etree.ElementTree as ET

import mujoco
import numpy as np

TABLE_TOP = -0.06  # m, base frame
TABLE_CENTER = (0.70, 0.08)  # m, base frame
TABLE_HALF = (0.45, 0.62, 0.025)  # m: x 0.25 to 1.15, y -0.54 to 0.70
CUP_RADIUS, CUP_HEIGHT, CUP_WALL = 0.03, 0.09, 0.004  # m
CONTAINER_HALF, CONTAINER_HEIGHT, CONTAINER_WALL = 0.06, 0.05, 0.005  # m: 12 cm square inside
CUP_RANGE = ((0.46, 0.95), (-0.20, 0.44))  # m, base frame: where real cups were grasped
CONTAINER_RANGE = ((0.42, 0.94), (-0.28, 0.36))  # m, base frame: where real cups were released
MIN_SEPARATION = 0.18  # m between cup and container centres
DEFAULT_CUP, DEFAULT_CONTAINER = (0.85, 0.17), (0.69, -0.12)  # m, base frame: the real medians
BLUE, RED = '0.12 0.32 0.85 1', '0.85 0.12 0.10 1'


def _f(values):
    return ' '.join(f'{v:.5f}' for v in values)


def _cup(name):
    """A hollow cylinder: a floor and 16 wall slabs, free to move."""
    body = ET.Element('body', name=name, pos='0 0 0')
    ET.SubElement(body, 'freejoint', name=f'{name}_joint')
    ET.SubElement(
        body,
        'geom',
        name=f'{name}_floor',
        type='cylinder',
        size=_f([CUP_RADIUS, CUP_WALL / 2]),
        pos=_f([0, 0, CUP_WALL / 2]),
        rgba=BLUE,
        friction='1.5 0.01 0.001',
        density='600',
    )
    n = 16
    half_width = CUP_RADIUS * np.tan(np.pi / n) + 0.001
    for i in range(n):
        a = 2 * np.pi * i / n
        r = CUP_RADIUS - CUP_WALL / 2
        ET.SubElement(
            body,
            'geom',
            name=f'{name}_wall{i}',
            type='box',
            size=_f([CUP_WALL / 2, half_width, CUP_HEIGHT / 2]),
            pos=_f([r * np.cos(a), r * np.sin(a), CUP_HEIGHT / 2]),
            euler=_f([0, 0, a]),
            rgba=BLUE,
            friction='1.5 0.01 0.001',
            density='600',
        )
    return body


def _container(name):
    """An open box: a floor and 4 walls, free to move."""
    body = ET.Element('body', name=name, pos='0 0 0')
    ET.SubElement(body, 'freejoint', name=f'{name}_joint')
    h, t, H = CONTAINER_HALF, CONTAINER_WALL, CONTAINER_HEIGHT
    ET.SubElement(
        body, 'geom', name=f'{name}_floor', type='box', size=_f([h + t, h + t, t / 2]), pos=_f([0, 0, t / 2]), rgba=RED
    )
    for i, (x, y, sx, sy) in enumerate((
        (h + t / 2, 0, t / 2, h + t),
        (-h - t / 2, 0, t / 2, h + t),
        (0, h + t / 2, h, t / 2),
        (0, -h - t / 2, h, t / 2),
    )):
        ET.SubElement(
            body, 'geom', name=f'{name}_wall{i}', type='box', size=_f([sx, sy, H / 2]), pos=_f([x, y, H / 2]), rgba=RED
        )
    return body


def base_pose(env):
    """World position and rotation of the robot base."""
    m, d = env.sim.model._model, env.sim.data._data
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'robot0_base')
    return d.xpos[b].copy(), d.xmat[b].reshape(3, 3).copy()


def build_cup_scene(env):
    """Rebuild env's model with the raised, enlarged table, the cup, the container and a third-eye
    camera; the task's pot is moved out of reach. Returns env, reset onto the new model."""
    pb, Rb = base_pose(env)
    root = ET.fromstring(env.sim.model.get_xml())
    world = root.find('worldbody')
    table = next(b for b in root.iter('body') if b.get('name') == 'table')
    top = pb + Rb @ np.array([TABLE_CENTER[0], TABLE_CENTER[1], TABLE_TOP])
    table.set('pos', _f(top - np.array([0, 0, TABLE_HALF[2]])))
    for geom in table.iter('geom'):
        if geom.get('name') in ('table_collision', 'table_visual'):
            geom.set('size', _f(TABLE_HALF))
        elif 'leg' in (geom.get('name') or ''):
            geom.set('rgba', '0 0 0 0')  # the legs are sized for the old table
    world.append(_cup('cup'))
    world.append(_container('container'))
    # Above the table's far edge, looking back down at it with the robot on the left of the image, as
    # the real overhead camera does.
    eye = pb + Rb @ np.array([1.25, -0.35, 1.05])
    target = pb + Rb @ np.array([0.65, 0.10, TABLE_TOP])
    z = (eye - target) / np.linalg.norm(eye - target)  # MuJoCo cameras look along -z
    x = np.cross([0, 0, 1], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    ET.SubElement(world, 'camera', name='third_eye', pos=_f(eye), xyaxes=_f([*x, *y]), fovy='58')
    env.reset_from_xml_string(ET.tostring(root, encoding='unicode'))
    m, d = env.sim.model._model, env.sim.data._data
    for j in np.flatnonzero(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE):
        if 'pot' in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ''):
            d.qpos[m.jnt_qposadr[j] : m.jnt_qposadr[j] + 3] = [0.0, -3.0, 0.2]
    for g in range(m.ngeom):  # the task's pot: out of reach, unseen, untouchable
        if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or '').startswith('pot'):
            m.geom_rgba[g, 3], m.geom_contype[g], m.geom_conaffinity[g] = 0.0, 0, 0
    place(env, DEFAULT_CUP, DEFAULT_CONTAINER)
    return env


def place(env, cup_xy, container_xy, cup_yaw=0.0, container_yaw=0.0):
    """Set the cup and container on the table at base-frame (x, y), at rest."""
    m, d = env.sim.model._model, env.sim.data._data
    pb, Rb = base_pose(env)
    for name, xy, yaw in (('cup', cup_xy, cup_yaw), ('container', container_xy, container_yaw)):
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f'{name}_joint')
        a, v = m.jnt_qposadr[j], m.jnt_dofadr[j]
        d.qpos[a : a + 3] = pb + Rb @ np.array([xy[0], xy[1], TABLE_TOP + 0.001])
        d.qpos[a + 3 : a + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        d.qvel[v : v + 6] = 0
    mujoco.mj_forward(m, d)


def sample_layout(rng):
    """Cup and container positions (base frame) inside the real demonstrations' ranges, apart."""
    while True:
        cup = [rng.uniform(*CUP_RANGE[0]), rng.uniform(*CUP_RANGE[1])]
        box = [rng.uniform(*CONTAINER_RANGE[0]), rng.uniform(*CONTAINER_RANGE[1])]
        if np.hypot(cup[0] - box[0], cup[1] - box[1]) >= MIN_SEPARATION:
            return np.array(cup), np.array(box), rng.uniform(-np.pi, np.pi)


def object_xy(env, name):
    """Base-frame position of a free body."""
    m, d = env.sim.model._model, env.sim.data._data
    pb, Rb = base_pose(env)
    return Rb.T @ (d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)] - pb)


if __name__ == '__main__':
    # Live view: a new random layout every 5 s (uv run python -m baxter_mujoco.scenes).
    import time

    import mujoco.viewer

    from baxter_mujoco.sim import make_env

    env = build_cup_scene(make_env())
    m, d = env.sim.model._model, env.sim.data._data
    rng = np.random.default_rng(0)
    with mujoco.viewer.launch_passive(m, d) as viewer:
        pb, Rb = base_pose(env)
        viewer.cam.lookat[:] = pb + Rb @ np.array([0.70, 0.08, TABLE_TOP])
        viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 1.6, -45, 180
        while viewer.is_running():
            cup, box, yaw = sample_layout(rng)
            place(env, cup, box, container_yaw=yaw)
            end = time.monotonic() + 5.0
            while viewer.is_running() and time.monotonic() < end:
                mujoco.mj_step(m, d)
                viewer.sync()
                time.sleep(m.opt.timestep)

"""The cup scene: the BlueCupRedContainer task in the lab where it was demonstrated.

The lab (its table and third_eye camera) is data, lab.toml; the task's objects and ranges are here,
from the 56 real demonstrations of BlueCupRedContainer_kin_pose. The cup is 6 cm across and 9 cm
tall, wider than the gripper opens (4.2 cm), so it is held by its wall.

Poses are given in the robot base frame, as the teleop and the recorded datasets use it.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import tomllib
from scipy.spatial.transform import Rotation

LAB = tomllib.loads((Path(__file__).parent / 'lab.toml').read_text())
TABLE_TOP = LAB['table']['top']  # m, base frame
TABLE_CENTER, TABLE_HALF = LAB['table']['center'], LAB['table']['half']  # m, base frame
CUP_RADIUS, CUP_HEIGHT, CUP_WALL = 0.03, 0.09, 0.004  # m
CONTAINER_HALF, CONTAINER_HEIGHT, CONTAINER_WALL = 0.06, 0.05, 0.005  # m: 12 cm square inside
# Where real cups were grasped (x 0.46 to 0.95, y -0.20 to 0.44) and released (x 0.42 to 0.94, y -0.28
# to 0.36), capped at the sim arm's reach: its fingers are 4.5 cm shorter than the real ones, and every
# failed attempt of the first 60-episode run had the cup (x 0.872 or more) or the container (0.883 or
# more) beyond 0.87 m.
REACH_X = 0.87  # m
CUP_RANGE = ((0.46, REACH_X), (-0.20, 0.44))  # m, base frame
CONTAINER_RANGE = ((0.42, REACH_X), (-0.28, 0.36))  # m, base frame
MIN_SEPARATION = 0.18  # m between cup and container centres
DEFAULT_CUP, DEFAULT_CONTAINER = (0.85, 0.17), (0.69, -0.12)  # m, base frame: the real medians
BLUE, RED = '0.12 0.32 0.85 1', '0.85 0.12 0.10 1'
# Lighting randomisation for data generation, around the lab's (lab.toml): per light a position offset
# (which moves its shadows), an intensity scale, a colour tint and whether it casts shadows; and the
# ambient level. /sim/lighting carries [ambient scale, then per light dx dy dz scale r g b shadow].
LIGHT_OFFSET = (0.3, 0.3, 0.2)  # m, +- in base x y z
LIGHT_SCALE = (0.6, 1.6)  # shadows from faint to clear, never darker than the lit table
LIGHT_TINT = (0.85, 1.15)  # per colour channel
LIGHT_SHADOW_P = 0.5  # probability a light casts shadows
AMBIENT_SCALE = (0.85, 1.1)


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
        group='1',
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
            group='1',
        )
    return body


def _container(name):
    """An open box: a floor and 4 walls, free to move."""
    body = ET.Element('body', name=name, pos='0 0 0')
    ET.SubElement(body, 'freejoint', name=f'{name}_joint')
    h, t, H = CONTAINER_HALF, CONTAINER_WALL, CONTAINER_HEIGHT
    ET.SubElement(
        body,
        'geom',
        name=f'{name}_floor',
        type='box',
        size=_f([h + t, h + t, t / 2]),
        pos=_f([0, 0, t / 2]),
        rgba=RED,
        group='1',
    )
    for i, (x, y, sx, sy) in enumerate((
        (h + t / 2, 0, t / 2, h + t),
        (-h - t / 2, 0, t / 2, h + t),
        (0, h + t / 2, h, t / 2),
        (0, -h - t / 2, h, t / 2),
    )):
        ET.SubElement(
            body,
            'geom',
            name=f'{name}_wall{i}',
            type='box',
            size=_f([sx, sy, H / 2]),
            pos=_f([x, y, H / 2]),
            rgba=RED,
            group='1',
        )
    return body


def _camera(name, cam, pb, Rb):
    """A pinhole camera from a calibration: base-frame pose and camera_info intrinsics, in pixels."""
    (w, h), (fx, fy), (cx, cy) = cam['size'], cam['focal'], cam['principal']
    pixel = 1e-5  # m, any pitch: MuJoCo uses focal / pitch
    R = Rb @ Rotation.from_quat(cam['quat'], scalar_first=True).as_matrix()
    return ET.Element(
        'camera',
        name=name,
        pos=_f(pb + Rb @ np.array(cam['pos'])),
        xyaxes=_f([*R[:, 0], *R[:, 1]]),
        resolution=f'{w} {h}',
        sensorsize=_f([w * pixel, h * pixel]),
        focal=_f([fx * pixel, fy * pixel]),
        # MuJoCo's principal point is an offset from the image centre, y up; camera_info's is from
        # the top-left corner, y down
        principal=_f([(cx - w / 2) * pixel, (h / 2 - cy) * pixel]),
    )


def _floor(root, floor):
    """The floor's texture as a builtin speckle."""
    asset = root.find('asset')
    name = next(m for m in asset.iter('material') if m.get('name') == 'floorplane').get('texture')
    for texture in [t for t in asset.iter('texture') if t.get('name') == name]:
        asset.remove(texture)
    ET.SubElement(
        asset,
        'texture',
        name=name,
        type='2d',
        builtin='flat',
        mark='random',
        rgb1=_f(floor['rgb1']),
        rgb2=_f(floor['rgb2']),
        random=f'{floor["random"]}',
        width='512',
        height='512',
    )


def _lights(root, lighting, pb, Rb):
    """The lab's lights in place of robosuite's, and its headlight."""
    world = root.find('worldbody')
    for light in world.findall('light'):
        world.remove(light)
    for i, light in enumerate(lighting['lights']):
        ET.SubElement(
            world,
            'light',
            name=f'lab_light{i}',
            type='spot',
            pos=_f(pb + Rb @ np.array(light['pos'])),
            dir=_f(Rb @ np.array([0.0, 0.0, -1.0])),
            diffuse=_f([light['diffuse']] * 3),
            specular='0.1 0.1 0.1',
            cutoff=f'{lighting["cutoff"]}',
            exponent=f'{lighting["exponent"]}',
            castshadow='true',
        )
    visual = root.find('visual')
    if visual is None:
        visual = ET.SubElement(root, 'visual')
    headlight = visual.find('headlight')
    if headlight is None:
        headlight = ET.SubElement(visual, 'headlight')
    headlight.set('ambient', _f([lighting['headlight_ambient']] * 3))
    headlight.set('diffuse', _f([lighting['headlight_diffuse']] * 3))
    headlight.set('specular', '0 0 0')


def apply_lighting(env, values):
    """Set the lights from a /sim/lighting vector, relative to the lab's (lab.toml)."""
    m = env.sim.model._model
    pb, Rb = base_pose(env)
    lighting = LAB['lighting']
    m.vis.headlight.ambient[:] = lighting['headlight_ambient'] * values[0]
    for i, light in enumerate(lighting['lights']):
        dx, dy, dz, scale, r, g, b, shadow = values[1 + 8 * i : 9 + 8 * i]
        lid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_LIGHT, f'lab_light{i}')
        m.light_pos[lid] = pb + Rb @ (np.array(light['pos']) + [dx, dy, dz])
        m.light_diffuse[lid] = light['diffuse'] * scale * np.array([r, g, b])
        m.light_castshadow[lid] = shadow > 0.5


def lab_lighting():
    """The /sim/lighting vector of the lab's own lighting."""
    return [1.0] + [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0] * len(LAB['lighting']['lights'])


def sample_lighting(rng):
    """A random /sim/lighting vector around the lab's lighting."""
    values = [rng.uniform(*AMBIENT_SCALE)]
    for _ in LAB['lighting']['lights']:
        values += [rng.uniform(-o, o) for o in LIGHT_OFFSET]
        values += [rng.uniform(*LIGHT_SCALE), *rng.uniform(*LIGHT_TINT, size=3)]
        values += [float(rng.random() < LIGHT_SHADOW_P)]
    return values


def base_pose(env):
    """World position and rotation of the robot base."""
    m, d = env.sim.model._model, env.sim.data._data
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'robot0_base')
    return d.xpos[b].copy(), d.xmat[b].reshape(3, 3).copy()


def build_cup_scene(env):
    """Rebuild env's model as the lab (lab.toml: its table and third_eye camera) with the cup and the
    container; the task's pot is moved out of reach. Returns env, reset onto the new model."""
    pb, Rb = base_pose(env)
    root = ET.fromstring(env.sim.model.get_xml())
    world = root.find('worldbody')
    table = next(b for b in root.iter('body') if b.get('name') == 'table')
    top = pb + Rb @ np.array([TABLE_CENTER[0], TABLE_CENTER[1], TABLE_TOP])
    table.set('pos', _f(top - np.array([0, 0, TABLE_HALF[2]])))
    for geom in table.iter('geom'):
        if geom.get('name') in ('table_collision', 'table_visual'):
            geom.set('size', _f(TABLE_HALF))
            geom.attrib.pop('material', None)
            geom.set('rgba', _f(LAB['table']['rgba']))
        elif 'leg' in (geom.get('name') or ''):
            geom.set('rgba', '0 0 0 0')  # the legs are sized for the old table
    world.append(_cup('cup'))
    world.append(_container('container'))
    world.append(_camera('third_eye', LAB['third_eye'], pb, Rb))
    _floor(root, LAB['floor'])
    _lights(root, LAB['lighting'], pb, Rb)
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

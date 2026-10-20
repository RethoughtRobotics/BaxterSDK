"""The scripted expert at any frame of an attempt: its chunk, what the demonstrations' labels would have
been had it taken over there.

Task.replan gives the plan from the state the sim reports (the TCP, the gripper, the objects); a
chunk samples it as BlueCupRedContainer_sim's actions are recorded: HORIZON steps at FPS of
[x y z qx qy qz qw gripper], the measured TCP (labels "measured", no lead) and the measured gripper
(0 open .. 1 closed), step 0 the state now. The targets keep PolicyArm.follow's timing (a minimum-jerk
move per waypoint, then its gripper command, then its hold, all STRETCH times longer as it runs them);
the measured TCP trails them by LAG s and the gripper moves at GRIP_RATE towards the task's closed
value GRIP_DELAY s after a command. An arm on one of the expert's moves is taken over where the expert
would be on it; elsewhere a moving arm's first move starts at its speed. Fitted on 10 episodes of
BlueCupRedContainer_sim (0, 6 .. 54), from every third frame once the arm moves: the chunk's first 16
positions are a median 1.9 mm from the recorded labels (95th percentile 14 mm), 0.9% of frames more
than 3 cm (long transit moves, where the sim arm leaves the straight line), the gripper 0.004.

Offline, over evaluator bags (python -m baxter_mujoco.datagen.expert BAG...): the attempt's states at
FPS from the engage to the release, on the bag's recording clock, each taking the latest message at or
before it as the dataset export does, and every frame's expert chunk, in <bag>/expert.npz:
  t          (N,) ns, recording time of each frame
  tcp        (N, 7) TCP position and quaternion [x y z w], base frame
  gripper    (N,) 0 open .. 1 closed (1 - opening / 100)
  cup, container  (N, 7) base-frame poses from /sim/objects
  chunk      (N, HORIZON, 8) the expert's chunk; NaN where it has no plan (valid False)
  valid      (N,) bool
  home       (7,) the TCP at the first frame (the attempt starts at home), where every plan ends
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from baxter_mujoco.datagen.motion import min_jerk, move_duration

FPS = 25.0  # the datasets' rate
HORIZON = 32  # steps, the sim checkpoints' chunk
LAG = 0.0  # s, the measured TCP behind the expert's target: none beyond STRETCH in the fit
# PolicyArm streams its targets with a sleep of 1 / RATE per step, so its moves and holds take longer
# than planned by this factor (fitted on the demonstrations)
STRETCH = 1.16
GRIP_RATE = 1.0  # per s, the measured gripper's speed
GRIP_DELAY = 0.1  # s from a gripper command to the fingers moving
DT = 0.005  # s, integration step
ON_PATH = 0.01  # m, the arm on the expert's move under way
OBJECTS = ('cup', 'container')  # /sim/objects order (generate.OBJECTS)


def schedule(replan, start, velocity=None):
    """The expert's targets from the measured TCP `start` (position, Rotation): moves [(t0, t1, from, to,
    initial velocity)] and gripper commands [(t, grip)], times in s from now, as PolicyArm.follow runs
    them; the measured TCP is the target LAG s earlier. With no wait, a TCP on the move from the
    Replan's origin (within ON_PATH) is on the expert's move under way: it continues, its target LAG s
    ahead. Else, with `velocity` (m/s), the arm keeps that speed for LAG s and the first move starts
    there at that speed (the arm taken over moving)."""
    moves, grips, t, pose = [], [(0.0, replan.grip)], STRETCH * replan.wait, start
    v0 = np.zeros(3)
    if replan.wait == 0 and replan.waypoints:
        first = replan.waypoints[0]
        u = resumed(replan.origin, first, start[0])
        if u:
            goal = (np.asarray(first.position, float), first.rotation)
            t = -LAG - u * STRETCH * move_duration(replan.origin, goal, first.slow)
            pose = replan.origin
        elif velocity is not None:
            v0 = np.asarray(velocity, float)
            ahead = (start[0] + LAG * v0, start[1])
            moves.append((-LAG, 0.0, start, ahead, None))
            pose = ahead
    for w in replan.waypoints:
        goal = (np.asarray(w.position, float), w.rotation)
        duration = STRETCH * move_duration(pose, goal, w.slow)
        moves.append((t, t + duration, pose, goal, v0))
        t, pose, v0 = t + duration, goal, np.zeros(3)
        grips.append((t, w.grip))
        t += STRETCH * w.hold
    return moves, grips, pose


def resumed(origin, waypoint, position):
    """The phase (0 .. 1) of the minimum-jerk move from `origin` to `waypoint` at `position`, if it lies
    on that move (within ON_PATH of the line, strictly between the ends); else 0."""
    if origin is None:
        return 0.0
    a, d = np.asarray(origin[0], float), np.asarray(waypoint.position, float) - np.asarray(origin[0], float)
    length = np.linalg.norm(d)
    if length < ON_PATH:
        return 0.0
    f = float(np.dot(position - a, d) / length**2)
    if not 0.0 < f < 1.0 or np.linalg.norm(position - a - f * d) > ON_PATH:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(40):  # min_jerk is increasing on [0, 1]
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if min_jerk(mid) < f else (lo, mid)
    return (lo + hi) / 2


def quintic(u, distance, v0):
    """Fraction of `distance` covered at u in [0, 1] from initial velocity v0 (per unit u) to rest:
    minimum jerk when v0 is 0."""
    return (10 * distance - 6 * v0) * u**3 + (-15 * distance + 8 * v0) * u**4 + (6 * distance - 3 * v0) * u**5 + v0 * u


def targets(replan, start, times, velocity=None):
    """The target positions (N, 3), Rotations and gripper commands (N,) at `times` (s from now, rising),
    the TCP before the first move."""
    moves, grips, end = schedule(replan, start, velocity)
    times = np.asarray(times, float)
    positions = np.repeat(end[0][None], len(times), 0)
    rotations = [end[1]] * len(times)
    for t0, t1, a, b, v0 in reversed(moves):
        before = times < t0
        positions[before] = a[0]
        for i in np.flatnonzero(before):
            rotations[i] = a[1]
        inside = np.flatnonzero((times >= t0) & (times < t1))
        if len(inside):
            u = (times[inside] - t0) / (t1 - t0)
            if v0 is None:  # keeping its speed
                positions[inside] = a[0] + u[:, None] * (b[0] - a[0])
                rotations_inside = [a[1]] * len(inside)
            else:
                positions[inside] = a[0] + quintic(u[:, None], b[0] - a[0], v0 * (t1 - t0))
                rotations_inside = Slerp([0, 1], Rotation.concatenate([a[1], b[1]]))(np.clip(min_jerk(u), 0, 1))
            for k, i in enumerate(inside):
                rotations[i] = rotations_inside[k]
    commands = np.zeros(len(times))
    for t, g in grips:
        commands[times >= t] = g
    commands[times < 0] = grips[0][1]
    return positions, Rotation.concatenate(rotations), commands


def chunk(replan, tcp, gripper, closed, velocity=None, horizon=HORIZON, fps=FPS):
    """(horizon, 8) the expert's chunk from the measured TCP (position, Rotation), gripper and TCP
    velocity (m/s, None at rest): the measured TCP and gripper it would record, quaternions on the
    TCP's side (qy >= 0, as recorded)."""
    steps = np.arange(horizon) / fps
    position, rotation, _ = targets(replan, tcp, steps - LAG, velocity)
    fine = np.arange(int(round(horizon / fps / DT)) + 1) * DT
    _, _, command = targets(replan, tcp, fine - GRIP_DELAY, velocity)
    every = int(round(1 / fps / DT))
    grip, g = np.zeros(horizon), float(gripper)
    for i in range(len(fine)):
        if i % every == 0 and i // every < horizon:
            grip[i // every] = g
        g += float(np.clip(closed * command[i] - g, -GRIP_RATE * DT, GRIP_RATE * DT))
    quat = rotation.as_quat()
    reference = np.asarray(tcp[1].as_quat())
    reference = -reference if reference[1] < 0 else reference
    quat[quat @ reference < 0] *= -1
    return np.column_stack([position, quat, grip])


def expert_chunk(task, tcp, gripper, objects, home, velocity=None, layout=None, horizon=HORIZON, fps=FPS):
    """The expert's chunk at a state (Task.replan's arguments, and the TCP's velocity), or None where
    it has no plan."""
    replan = task.replan(tcp, gripper, objects, home, layout)
    if replan is None:
        return None
    return chunk(replan, tcp, gripper, task.grip_closed, velocity, horizon, fps)


def pose(row):
    """(position, Rotation) of a [x y z qx qy qz qw] row."""
    return np.asarray(row[:3], float), Rotation.from_quat(row[3:7])


def velocities(positions, t):
    """(N, 3) m/s of the TCP at each frame, from the frame before (zero at the first)."""
    v = np.zeros_like(positions)
    v[1:] = np.diff(positions, axis=0) / (np.diff(t)[:, None] / 1e9)
    return v


def label(task, frames, horizon=HORIZON, fps=FPS):
    """Every frame's expert chunk (N, horizon, 8), NaN where there is none, and the valid mask; `frames`
    as states() returns them."""
    home = pose(frames['home'])
    out = np.full((len(frames['t']), horizon, 8), np.nan)
    velocity = velocities(frames['tcp'][:, :3], frames['t'])
    layout = task.layout({name: (frames[name][0, :3], frames[name][0, 3:]) for name in OBJECTS})
    for i in range(len(out)):
        objects = {name: (frames[name][i, :3], frames[name][i, 3:]) for name in OBJECTS}
        tcp = pose(frames['tcp'][i])
        c = expert_chunk(task, tcp, frames['gripper'][i], objects, home, velocity[i], layout, horizon, fps)
        if c is not None:
            out[i] = c
    return out, ~np.isnan(out[:, 0, 0])


def states(bag, arm='left', fps=FPS):
    """The attempt's states at `fps` from the policy's engage to its release (or the bag's end), each the
    latest message at or before the frame: {t, tcp, gripper, cup, container, home} as in expert.npz."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    topics = {
        f'/teleop/{arm}/tcp': 'tcp',
        f'/robot/end_effector/{arm}_gripper/state': 'gripper',
        '/sim/objects': 'objects',
        f'/policy/{arm}/engaged': 'engaged',
    }
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id='mcap'), rosbag2_py.ConverterOptions('', ''))
    types = {x.name: x.type for x in reader.get_all_topics_and_types()}
    missing = [tp for tp in topics if tp not in types]
    if missing:
        raise ValueError(f'{bag}: no {", ".join(missing)} (an evaluator bag, --bags)')
    reader.set_filter(rosbag2_py.StorageFilter(topics=list(topics)))
    series = {name: [] for name in topics.values()}
    while reader.has_next():
        topic, raw, stamp = reader.read_next()
        msg = deserialize_message(raw, get_message(types[topic]))
        name = topics[topic]
        if name == 'tcp':
            p, o = msg.pose.position, msg.pose.orientation
            value = [p.x, p.y, p.z, o.x, o.y, o.z, o.w]
        elif name == 'gripper':
            value = 1.0 - float(msg.position) / 100.0
        elif name == 'objects':
            value = [
                [
                    p.position.x,
                    p.position.y,
                    p.position.z,
                    p.orientation.x,
                    p.orientation.y,
                    p.orientation.z,
                    p.orientation.w,
                ]
                for p in msg.poses
            ]
        else:
            value = bool(msg.data)
        series[name].append((stamp, value))
    on = [s for s, v in series['engaged'] if v]
    if not on:
        raise ValueError(f'{bag}: the policy was never engaged')
    off = [s for s, v in series['engaged'] if not v and s > on[0]]
    first = max(on[0], *(series[n][0][0] for n in ('tcp', 'gripper', 'objects')))
    last = min(off[0] if off else np.inf, *(series[n][-1][0] for n in ('tcp', 'gripper', 'objects')))
    t = np.arange(first, last + 1, int(round(1e9 / fps)), dtype=np.int64)

    def at(name):
        stamps = np.array([s for s, _ in series[name]])
        index = np.searchsorted(stamps, t, side='right') - 1
        return np.array([series[name][i][1] for i in index], float)

    objects = at('objects')
    frames = {'t': t, 'tcp': at('tcp'), 'gripper': at('gripper')}
    frames.update({name: objects[:, k] for k, name in enumerate(OBJECTS)})
    frames['home'] = frames['tcp'][0]
    return frames


def write(bag, task, out='expert.npz', arm='left', horizon=HORIZON, fps=FPS):
    """states() and label() of `bag` to <bag>/<out>; returns its path."""
    frames = states(bag, arm, fps)
    frames['chunk'], frames['valid'] = label(task, frames, horizon, fps)
    meta = {'fps': fps, 'horizon': horizon, 'arm': arm, 'task': task.name}
    path = Path(bag) / out
    np.savez_compressed(path, meta=json.dumps(meta), **frames)
    return path


def main():
    from baxter_mujoco.datagen.generate import TASKS

    parser = argparse.ArgumentParser(description="The scripted expert's chunk at every frame of evaluator bags")
    parser.add_argument('bags', nargs='+', help='bag directories (the evaluator --bags writes seed_NNNN)')
    parser.add_argument('--task', choices=sorted(TASKS), default='cup')
    parser.add_argument('--arm', default='left')
    parser.add_argument('--horizon', type=int, default=HORIZON)
    parser.add_argument('--out', default='expert.npz', help='file name inside each bag directory')
    parser.add_argument('--force', action='store_true', help='rewrite an existing one')
    args = parser.parse_args()
    for bag in args.bags:
        if (Path(bag) / args.out).exists() and not args.force:
            print(f'{bag}: {args.out} exists', flush=True)
            continue
        path = write(bag, TASKS[args.task], args.out, args.arm, args.horizon)
        z = np.load(path)
        print(f'{bag}: {len(z["t"])} frames, {int(z["valid"].sum())} with a plan', flush=True)


if __name__ == '__main__':
    main()

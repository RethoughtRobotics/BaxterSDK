#!/usr/bin/env python3

"""Control benchmark runner: one-factor sweeps of the teleop's control parameters on the robot.

Every run plays the ISO 9283-style protocol (iso9283.iso_protocol: pose, linear,
circle) through scripted_trajectory with one set of parameters, recorded to its own bag. Parameters are
varied one at a time around a reference configuration (REFERENCE); SWEEPS lists the values. A
condition that occurs in several sweeps (e.g. the reference) is run once and symlinked into each
sweep's folder. Runs are shuffled (seeded) so slow drift (temperature, cable routing, the operator
moving around the cell) does not line up with any sweep, and the runner resumes where it stopped.

  <root>/protocol.yaml                 the plan: reference, sweeps, repeats, seed
  <root>/<sweep>/bags/<label>__r<k>/   one bag per run (symlink when shared), with run.yaml
  <root>/runs.csv                      one line per finished run (status, abort reason, duration)

  python -m baxter_teleop.diagnostics.benchmark <root> [--sweeps gain,speed] [--repeats 3] [--dry-run]

Needs the robot (or the sim) up on ROS_DOMAIN_ID and scripts/record_teleop.sh (BAG_PATH).
"""

import argparse
import csv
import itertools
import os
import random
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field

import yaml

# Reference configuration: the best combination found on the robot (2026-10-16 circle and VR tests).
REFERENCE = {
    'output': 'raw_id',
    'gain': 6.0,
    'feedforward': True,
    'servo_feedback': 'setpoint',
    'joint_accel_scale': 1.0,
    'speed': 0.15,
}

# sweep name -> list of parameter overrides. Output stages are compared on measured feedback, the
# only setting all four support (velocity and raw_id_step cannot close on a streamed setpoint).
SWEEPS = {
    'output_stage': [{'output': o, 'servo_feedback': 'measured'} for o in ('velocity', 'raw_id_step', 'raw', 'raw_id')],
    'gain': [{'gain': float(k), 'feedforward': ff} for ff in (False, True) for k in range(1, 11)],
    'servo_feedback': [{'servo_feedback': f} for f in ('measured', 'setpoint')],
    'accel_limit': [{'joint_accel_scale': s} for s in (1.0, 1.5, 2.0, 2.5, 3.0)],
    'speed': [{'speed': round(0.05 * i, 2)} for i in range(1, 9)],
}

# Approximate run time: homing + recorder start/stop + protocol (60 s at 0.15 m/s; longer when slow).
OVERHEAD_S = 20.0


def protocol_seconds(speed):
    from .iso9283 import iso_protocol

    return len(iso_protocol(speed=speed)[0]) * 0.01


def label(params):
    """Readable, sortable, file-system safe name of a full parameter set."""
    return (
        f'out-{params["output"]}_K-{params["gain"]:04.1f}_ff-{"on" if params["feedforward"] else "off"}'
        f'_fb-{params["servo_feedback"]}_acc-x{params["joint_accel_scale"]:.1f}_v-{params["speed"]:.2f}'
    )


@dataclass(frozen=True)
class Run:
    params: tuple  # sorted (key, value) pairs, hashable
    repeat: int
    sweeps: tuple = field(default=())  # every sweep this run belongs to; the first holds the bag

    @property
    def p(self):
        return dict(self.params)

    @property
    def name(self):
        return f'{label(self.p)}__r{self.repeat}'

    def path(self, root, sweep=None):
        return os.path.join(root, sweep or self.sweeps[0], 'bags', self.name)


def plan(sweeps=None, repeats=3, seed=0):
    """Unique runs for the selected sweeps (each shared condition once), in seeded random order."""
    owners = {}
    for sweep in sweeps or SWEEPS:
        for override in SWEEPS[sweep]:
            key = tuple(sorted({**REFERENCE, **override}.items()))
            owners.setdefault(key, [])
            if sweep not in owners[key]:
                owners[key].append(sweep)
    runs = [Run(key, r, tuple(owners[key])) for key, r in itertools.product(owners, range(1, repeats + 1))]
    random.Random(seed).shuffle(runs)
    return runs


def is_done(run, root):
    meta = os.path.join(run.path(root), 'run.yaml')
    if not os.path.exists(meta):
        return False
    with open(meta) as f:
        return yaml.safe_load(f).get('status') in ('completed', 'aborted')


def trajectory_args(params, arm='right'):
    return [
        '--arm', arm, '--protocol', 'iso', '--safety',
        '--output', params['output'],
        '--gain', str(params['gain']),
        '--feedforward' if params['feedforward'] else '--no-feedforward',
        '--servo-feedback', params['servo_feedback'],
        '--joint-accel-scale', str(params['joint_accel_scale']),
        '--speed', str(params['speed']),
    ]  # fmt: skip


class Runner:
    """Plays runs on the robot: home, record, scripted_trajectory, stop recording, write run.yaml."""

    def __init__(self, root, scripts_dir, python):
        self.root, self.scripts_dir, self.python = root, scripts_dir, python

    def _run(self, run):
        bag = run.path(self.root)
        if os.path.exists(bag):  # a partial bag from an interrupted run
            subprocess.run(['rm', '-rf', bag], check=True)
        os.makedirs(os.path.dirname(bag), exist_ok=True)
        subprocess.run(['ros2', 'run', 'baxter_tools', 'tuck_arms', '-u'], capture_output=True, timeout=60)
        log = open(bag + '.record.log', 'w')
        env = {**os.environ, 'BAG_PATH': bag}
        recorder = subprocess.Popen(
            [os.path.join(self.scripts_dir, 'record_teleop.sh'), 'benchmark'],
            env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )  # fmt: skip
        try:
            deadline = time.monotonic() + 30
            while not os.path.isdir(bag) and time.monotonic() < deadline:
                time.sleep(0.2)
            time.sleep(3.0)  # every topic subscribed before the motion starts
            t0 = time.monotonic()
            result = subprocess.run(
                [self.python, '-m', 'baxter_teleop.diagnostics.scripted_trajectory', *trajectory_args(run.p)],
                capture_output=True, text=True,
            )  # fmt: skip
            duration = time.monotonic() - t0
            time.sleep(1.0)  # the hold after the end
        finally:
            os.killpg(recorder.pid, signal.SIGINT)
            try:
                recorder.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(recorder.pid, signal.SIGKILL)
            log.close()
        aborted = [line[len('ABORTED: ') :] for line in result.stdout.splitlines() if line.startswith('ABORTED: ')]
        status = 'aborted' if result.returncode == 3 else 'completed' if result.returncode == 0 else 'failed'
        info = {
            'status': status,
            'abort_reason': aborted[0] if aborted else None,
            'returncode': result.returncode,
            'params': run.p,
            'repeat': run.repeat,
            'sweeps': list(run.sweeps),
            'duration_s': round(duration, 2),
            'finished': time.strftime('%Y-%m-%dT%H:%M:%S'),
        }
        if status == 'failed':
            info['stderr_tail'] = result.stderr[-2000:]
        if os.path.isdir(bag):  # keep the recorder's log with its bag
            os.replace(bag + '.record.log', os.path.join(bag, 'record.log'))
        with open(os.path.join(bag, 'run.yaml'), 'w') as f:
            yaml.safe_dump(info, f, sort_keys=False)
        for sweep in run.sweeps[1:]:  # shared condition: link it into the other sweeps
            link = run.path(self.root, sweep)
            os.makedirs(os.path.dirname(link), exist_ok=True)
            if not os.path.lexists(link):
                os.symlink(os.path.relpath(bag, os.path.dirname(link)), link)
        return info

    def run_all(self, runs):
        todo = [r for r in runs if not is_done(r, self.root)]
        print(f'{len(runs) - len(todo)} of {len(runs)} runs already done; {len(todo)} to go')
        with open(os.path.join(self.root, 'runs.csv'), 'a', newline='') as f:
            out = csv.writer(f)
            for i, run in enumerate(todo, 1):
                print(f'[{i}/{len(todo)}] {run.name}  ({", ".join(run.sweeps)})', flush=True)
                info = self._run(run)
                print(f'    {info["status"]}{": " + info["abort_reason"] if info["abort_reason"] else ""}', flush=True)
                if info['status'] == 'failed':
                    print(info['stderr_tail'], file=sys.stderr)
                out.writerow([
                    run.name,
                    info['status'],
                    info['abort_reason'] or '',
                    info['duration_s'],
                    info['finished'],
                ])
                f.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('root', help='experiment folder (created; reused to resume)')
    parser.add_argument('--sweeps', default=','.join(SWEEPS), help=f'comma-separated subset of {",".join(SWEEPS)}')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dry-run', action='store_true', help='print the plan and its duration only')
    args = parser.parse_args()
    sweeps = [s for s in args.sweeps.split(',') if s]
    unknown = set(sweeps) - set(SWEEPS)
    if unknown:
        parser.error(f'unknown sweeps: {", ".join(sorted(unknown))}')
    runs = plan(sweeps, args.repeats, args.seed)
    seconds = sum(OVERHEAD_S + protocol_seconds(r.p['speed']) for r in runs if not is_done(r, args.root))
    print(f'{len(runs)} runs ({len(runs) // args.repeats} conditions x {args.repeats}) over {", ".join(sweeps)}')
    print(f'about {seconds / 3600:.1f} h of robot time left')
    if args.dry_run:
        for run in runs:
            print(f'  {run.name}  [{", ".join(run.sweeps)}]')
        return
    os.makedirs(args.root, exist_ok=True)
    with open(os.path.join(args.root, 'protocol.yaml'), 'w') as f:
        yaml.safe_dump(
            {'reference': REFERENCE, 'sweeps': {s: SWEEPS[s] for s in sweeps}, 'repeats': args.repeats, 'seed': args.seed,
             'trajectories': 'iso9283.iso_protocol (pose, linear, circle)'},
            f, sort_keys=False,
        )  # fmt: skip
    scripts_dir = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), *['..'] * 5))
    if not os.path.exists(os.path.join(scripts_dir, 'record_teleop.sh')):
        scripts_dir = os.environ.get('BAXTER_SCRIPTS', scripts_dir)
    Runner(args.root, scripts_dir, sys.executable).run_all(runs)


if __name__ == '__main__':
    main()

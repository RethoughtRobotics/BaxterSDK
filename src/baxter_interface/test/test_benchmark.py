"""Benchmark plan: shared conditions run once, labels, deterministic order, resume, CLI arguments."""

import os

import yaml

from baxter_interface.baxter_teleop.benchmark import REFERENCE, SWEEPS, is_done, label, plan, trajectory_args


def test_each_condition_runs_once_per_repeat_and_shared_ones_are_owned_by_all_their_sweeps():
    runs = plan(repeats=3)
    conditions = {r.params for r in runs}
    assert len(runs) == 3 * len(conditions)
    reference = tuple(sorted(REFERENCE.items()))
    ref_runs = [r for r in runs if r.params == reference]
    assert len(ref_runs) == 3
    # The reference is in gain (K=6, ff on), servo_feedback (setpoint), accel_limit (x1) and speed (0.15).
    assert set(ref_runs[0].sweeps) == {'gain', 'servo_feedback', 'accel_limit', 'speed'}
    # One-factor sweeps: every condition differs from the reference in at most its sweep's parameters.
    total = sum(len(v) for v in SWEEPS.values())
    assert len(conditions) < total  # shared conditions were merged


def test_labels_are_unique_and_name_every_parameter():
    runs = plan(repeats=1)
    names = [r.name for r in runs]
    assert len(set(names)) == len(names)
    name = label(REFERENCE)
    assert name == 'out-raw_id_K-06.0_ff-on_fb-setpoint_acc-x1.0_v-0.15'


def test_order_is_shuffled_but_reproducible():
    a, b, c = plan(seed=0), plan(seed=0), plan(seed=1)
    assert [r.name for r in a] == [r.name for r in b] != [r.name for r in c]
    assert sorted(r.name for r in a) == sorted(r.name for r in c)


def test_subset_of_sweeps():
    runs = plan(['speed'], repeats=2)
    assert len(runs) == 2 * len(SWEEPS['speed']) and all(r.sweeps == ('speed',) for r in runs)


def test_resume_counts_completed_and_aborted_runs_only(tmp_path):
    runs = plan(['servo_feedback'], repeats=1)
    for run, status in zip(runs, ('completed', 'failed')):
        os.makedirs(run.path(tmp_path))
        with open(os.path.join(run.path(tmp_path), 'run.yaml'), 'w') as f:
            yaml.safe_dump({'status': status}, f)
    assert [is_done(r, tmp_path) for r in runs] == [True, False]


def test_trajectory_args_carry_every_parameter():
    args = trajectory_args({**REFERENCE, 'feedforward': False, 'gain': 3.0})
    assert args[args.index('--output') + 1] == 'raw_id' and '--no-feedforward' in args
    assert (
        args[args.index('--gain') + 1] == '3.0' and '--safety' in args and args[args.index('--protocol') + 1] == 'iso'
    )

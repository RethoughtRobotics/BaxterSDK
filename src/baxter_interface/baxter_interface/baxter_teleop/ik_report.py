#!/usr/bin/env python3

"""Report comparing IK solvers on trajectory_test bags: figures (PDF + PNG), tables (CSV + LaTeX).

  python -m baxter_interface.baxter_teleop.ik_report <report_dir> <bag> [<bag> ...]

Each bag is one trajectory_test run (compare_ik.sh records them). Errors are measured against the
servo goal the teleop published (/teleop/<arm>/goal), the hand pose the robot reported
(/robot/limb/<arm>/endpoint_state) and the joint states, aligned on the robot's clock.
"""

import csv
import json
import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

import baxter_interface  # noqa: E402

from .bag_report import load, save_joints  # noqa: E402
from .cartesian_delta_teleop import CartesianDeltaTeleop, frax, jax, jnp  # noqa: E402
from .urdf_tools import extract_arm_chain_urdf, resolve_baxter_urdf  # noqa: E402

SUFFIXES = CartesianDeltaTeleop.JOINT_SUFFIXES
LABELS = {'dls': 'DLS', 'sns': 'SNS with posture bias'}
COLORS = {'dls': '#2a78d6', 'sns': '#eb6834'}  # validated categorical slots 1 and 2 (dataviz palette)
REFERENCE = '#52514e'  # commanded path, neutral
BAND = '#f1f1ef'
MUTED = '#6f6e69'
STYLE = {
    'font.family': 'serif',
    'font.serif': ['Latin Modern Roman', 'CMU Serif', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'font.size': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7.5,
    'axes.titlesize': 8,
    'axes.titleweight': 'normal',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.6,
    'axes.edgecolor': '#3d3c39',
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'axes.grid': True,
    'axes.grid.axis': 'y',
    'grid.color': '#e4e4e0',
    'grid.linewidth': 0.5,
    'grid.linestyle': '-',
    'lines.linewidth': 1.1,
    'lines.solid_capstyle': 'round',
    'legend.frameon': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
}
FULL_WIDTH = 7.0  # inches, two-column page width
PHASE_NAMES = {
    'square x3': 'Square, three loops',
    'down and up': 'Down and up',
    'elbow limit': 'Elbow limit',
    'circle x2': 'Circle, two loops',
    'whole run': 'Whole run',
}


def run(bag):
    """Time series of one trajectory_test bag, t = 0 at the trajectory start."""
    d = load(bag)
    save_joints(bag, d)
    test = d['config']['trajectory_test']
    goal, ep, js = d['goal'], d['ep'], d['js']
    t0 = goal[0, 0]
    index = np.searchsorted(goal[:, 0], ep[:, 0], side='right') - 1
    ok = (index >= 0) & (ep[:, 0] <= goal[-1, 0])
    index = index[ok]
    tcp, goal_at = ep[ok], goal[index]
    q = js[:, 1:8]
    home = np.asarray(baxter_interface.settings.UNTUCK_POSITIONS[d['arm']])
    names = d['joints'][:7]
    robot = frax.core.manipulator.Manipulator(
        extract_arm_chain_urdf(resolve_baxter_urdf(), d['arm'], names), joint_ordering=names
    )
    jacobian = jax.jit(robot.ee_jacobian)
    sub = slice(None, None, 10)
    return {
        'ik': test['ik'],
        'bag': os.path.basename(bag.rstrip('/')),
        'git': d['config'].get('git', ''),
        'phases': [(name, a, b) for name, a, b in test['phases']],
        't': tcp[:, 0] - t0,
        'goal': goal_at[:, 1:4],
        'tcp': tcp[:, 1:4],
        'position_error': 1e3 * np.linalg.norm(goal_at[:, 1:4] - tcp[:, 1:4], axis=1),
        'orientation_error': np.degrees(
            (Rotation.from_quat(goal_at[:, 4:8]) * Rotation.from_quat(tcp[:, 4:8]).inv()).magnitude()
        ),
        'tq': js[:, 0] - t0,
        'q': q,
        'deviation': np.degrees(np.abs(q - home).max(axis=1)),
        'tc': js[sub, 0] - t0,
        'cond': np.array([np.linalg.cond(np.asarray(jacobian(jnp.asarray(r, dtype=jnp.float32)))) for r in q[sub]]),
        'home': home,
    }


def phase_metrics(r):
    low, high = (np.asarray(x) for x in CartesianDeltaTeleop.JOINT_POSITION_LIMITS)
    at_limit = ((r['q'] - low < 0.03) | (high - r['q'] < 0.03)).any(axis=1)
    rows = []
    for name, a, b in r['phases'] + [('whole run', 0.0, r['t'][-1])]:
        m = (r['t'] >= a) & (r['t'] <= b)
        mq = (r['tq'] >= a) & (r['tq'] <= b)
        end = min(np.searchsorted(r['tq'], b), len(r['tq']) - 1)
        rows.append({
            'solver': LABELS[r['ik']],
            'phase': PHASE_NAMES.get(name, name),
            'rms position error (mm)': np.sqrt(np.mean(r['position_error'][m] ** 2)),
            'max position error (mm)': r['position_error'][m].max(),
            'max orientation error (deg)': r['orientation_error'][m].max(),
            'max joint deviation from home (deg)': r['deviation'][mq].max(),
            'joint deviation from home at end (deg)': r['deviation'][end],
            'time at a joint limit (%)': 100 * at_limit[mq].mean(),
        })
    return rows


def shade(ax, phases, label=False):
    for k, (name, a, b) in enumerate(phases):
        if k % 2 == 0:
            ax.axvspan(a, b, color=BAND, lw=0, zorder=0)
        if label:
            ax.text(
                (a + b) / 2,
                1.02,
                PHASE_NAMES.get(name, name),
                transform=ax.get_xaxis_transform(),
                ha='center',
                va='bottom',
                fontsize=7,
                color=MUTED,
            )


def legend(fig, runs, extra=()):
    handles = [plt.Line2D([], [], color=COLORS[r['ik']], lw=1.4) for r in runs] + [h for h, _ in extra]
    fig.legend(
        handles,
        [LABELS[r['ik']] for r in runs] + [t for _, t in extra],
        loc='upper center',
        ncol=len(handles),
        bbox_to_anchor=(0.5, 1.0),
        handlelength=1.8,
        columnspacing=1.6,
    )


def save(fig, out, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out, 'figures', f'{name}.{ext}'), dpi=300)
    plt.close(fig)


def figure_tracking(runs, out):
    fig, axes = plt.subplots(2, 1, figsize=(FULL_WIDTH, 3.4), sharex=True)
    for ax, key, label in zip(
        axes, ('position_error', 'orientation_error'), ('Position error (mm)', 'Orientation error (deg)')
    ):
        shade(ax, runs[0]['phases'], label=ax is axes[0])
        for r in runs:
            ax.plot(r['t'], r[key], color=COLORS[r['ik']])
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
    axes[-1].set_xlabel('Time (s)')
    axes[-1].set_xlim(0, max(r['t'][-1] for r in runs))
    legend(fig, runs)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, out, 'tracking_error')


def figure_configuration(runs, out):
    fig, axes = plt.subplots(3, 1, figsize=(FULL_WIDTH, 4.6), sharex=True)
    for r in runs:
        axes[0].plot(r['tq'], r['deviation'], color=COLORS[r['ik']])
        axes[1].plot(r['tq'], np.degrees(r['q'][:, 2]), color=COLORS[r['ik']])
        axes[2].plot(r['tc'], r['cond'], color=COLORS[r['ik']])
    axes[1].axhline(np.degrees(runs[0]['home'][2]), color=REFERENCE, lw=0.7)
    for ax, label in zip(
        axes, ('Max joint deviation\nfrom home (deg)', 'Elbow roll $e_0$ (deg)', 'Jacobian condition\nnumber')
    ):
        shade(ax, runs[0]['phases'], label=ax is axes[0])
        ax.set_ylabel(label)
    axes[0].set_ylim(bottom=0)
    axes[2].set_yscale('log')
    axes[-1].set_xlabel('Time (s)')
    axes[-1].set_xlim(0, max(r['tq'][-1] for r in runs))
    legend(fig, runs, extra=[(plt.Line2D([], [], color=REFERENCE, lw=0.7), 'Home posture')])
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, out, 'configuration')


def figure_joints(runs, out):
    low, high = (np.degrees(np.asarray(x)) for x in CartesianDeltaTeleop.JOINT_POSITION_LIMITS)
    fig, axes = plt.subplots(4, 2, figsize=(FULL_WIDTH, 6.4), sharex=True)
    for j, ax in enumerate(axes.flat[:7]):
        shade(ax, runs[0]['phases'])  # too narrow for phase names; same bands as the full-width figures
        for r in runs:
            ax.plot(r['tq'], np.degrees(r['q'][:, j]), color=COLORS[r['ik']])
        span = [np.degrees(r['q'][:, j]) for r in runs]
        lo, hi = min(s.min() for s in span), max(s.max() for s in span)
        pad = max(5.0, 0.15 * (hi - lo))
        for limit in (low[j], high[j]):  # drawn when within view
            if lo - pad <= limit <= hi + pad:
                ax.axhline(limit, color=REFERENCE, lw=0.7)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_ylabel(f'${SUFFIXES[j][0]}_{SUFFIXES[j][1]}$ (deg)')
    axes.flat[7].axis('off')
    for ax in axes[-1]:
        ax.set_xlabel('Time (s)')
    axes.flat[6].tick_params(labelbottom=True)
    axes.flat[6].set_xlabel('Time (s)')
    axes.flat[0].set_xlim(0, max(r['tq'][-1] for r in runs))
    handles = [plt.Line2D([], [], color=COLORS[r['ik']], lw=1.4) for r in runs]
    handles.append(plt.Line2D([], [], color=REFERENCE, lw=0.7))
    axes.flat[7].legend(handles, [LABELS[r['ik']] for r in runs] + ['Joint limit'], loc='center')
    fig.tight_layout()
    save(fig, out, 'joints')


def figure_paths(runs, out):
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 2.6))
    views = (
        ('square x3', (0, 1), 'x (mm)', 'y (mm)'),
        ('circle x2', (0, 1), 'x (mm)', 'y (mm)'),
        ('elbow limit', (0, 2), 'x (mm)', 'z (mm)'),
    )
    for ax, (phase, (i, k), xl, yl) in zip(axes, views):
        a, b = next((p[1], p[2]) for p in runs[0]['phases'] if p[0] == phase)
        origin = runs[0]['goal'][0]
        m0 = (runs[0]['t'] >= a) & (runs[0]['t'] <= b)
        g = 1e3 * (runs[0]['goal'][m0] - origin)
        ax.plot(g[:, i], g[:, k], color=REFERENCE, lw=0.8)
        for r in runs:
            m = (r['t'] >= a) & (r['t'] <= b)
            p = 1e3 * (r['tcp'][m] - r['goal'][0])
            ax.plot(p[:, i], p[:, k], color=COLORS[r['ik']], lw=1.0)
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(PHASE_NAMES[phase] + (', side view' if k == 2 else ', top view'), color=MUTED)
        ax.grid(True, axis='both')
    legend(fig, runs, extra=[(plt.Line2D([], [], color=REFERENCE, lw=0.8), 'Commanded')])
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, out, 'paths')


def write_tables(rows, out):
    keys = list(rows[0])
    with open(os.path.join(out, 'tables', 'summary.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: (f'{v:.2f}' if isinstance(v, float) else v) for k, v in row.items()} for row in rows)
    metrics = keys[2:]
    short = [
        'RMS pos.\\ (mm)',
        'Max pos.\\ (mm)',
        'Max orient.\\ (deg)',
        'Max dev.\\ (deg)',
        'End dev.\\ (deg)',
        'At limit (\\%)',
    ]
    lines = [
        '\\begin{tabular}{ll' + 'r' * len(metrics) + '}',
        '\\toprule',
        'Phase & Solver & ' + ' & '.join(short) + ' \\\\',
        '\\midrule',
    ]
    phases = list(dict.fromkeys(r['phase'] for r in rows))
    for n, phase in enumerate(phases):
        for k, row in enumerate(r for r in rows if r['phase'] == phase):
            cells = [phase if k == 0 else '', row['solver']] + [f'{row[m]:.1f}' for m in metrics]
            lines.append(' & '.join(cells) + ' \\\\')
        if n < len(phases) - 1:
            lines.append('\\addlinespace')
    lines += ['\\bottomrule', '\\end{tabular}']
    with open(os.path.join(out, 'tables', 'summary.tex'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


def write_readme(runs, rows, out):
    whole = {r['solver']: r for r in rows if r['phase'] == 'Whole run'}
    table = ['| Phase | Solver | ' + ' | '.join(list(rows[0])[2:]) + ' |', '|' + '---|' * len(rows[0])]
    table += [
        f'| {r["phase"]} | {r["solver"]} | ' + ' | '.join(f'{v:.1f}' for v in list(r.values())[2:]) + ' |' for r in rows
    ]
    text = f"""# IK solver comparison on the simulated Baxter

Runs: {', '.join(f'`{r["bag"]}` ({LABELS[r["ik"]]})' for r in runs)}. Code version `{runs[0]['git']}`.

## Setup

BaxFlow-T MuJoCo Baxter on the real robot's ROS 2 interface, no objects in reach (`--empty`), left
arm, starting from the untuck pose. `trajectory_test` drives the teleop through a scripted TCP
trajectory at 0.1 m/s with the start orientation held, using the same servo as VR and keyboard
input (goal pose, gain 5/s, path velocity as feedforward). The only difference between runs is the
velocity IK:

* **DLS**: damped least squares, minimum norm, the redundant elbow left free; at a joint limit the
  whole motion is scaled to a stop.
* **SNS with posture bias**: Rethink's SNS-IK (Flacco, De Luca, Khatib) with its nullspace bias task
  pulling toward the untuck posture at 1/s; joints saturate at their limits and the remaining
  motion is resolved in the nullspace. Same joint velocity limits and e1 limit as DLS
  (`config/sns_joint_limits.yaml`).

Phases: a 20 cm square three times, 20 cm down and up, a reach 35 cm forward and 30 cm out
(9 cm past where the e1 elbow limit stops the arm, clear of the torso) and back, a 10 cm radius
circle twice. Contacts during each run
(kinematic replay in the MuJoCo scene) are listed at the end.

## Results

{chr(10).join(table)}

Over the whole run, the joint deviation from home at the end is
{whole[LABELS['dls']]['joint deviation from home at end (deg)']:.1f} deg with DLS and
{whole[LABELS['sns']]['joint deviation from home at end (deg)']:.1f} deg with SNS
(the hand returns to the start in both).

## Figures

* `figures/tracking_error`: position and orientation error between the servo goal and the reported
  hand pose. Shaded bands mark the phases.
* `figures/configuration`: largest joint deviation from the home posture, the elbow roll joint
  $e_0$ (the redundant elbow swing) with its home value, and the Jacobian condition number.
* `figures/joints`: all seven joint angles with the joint limits that come into view (phase bands as
  in `figures/tracking_error`). The elbow
  overshoots its 20 deg limit: the IK brakes toward a limit assuming the arm follows its velocity
  commands at once, and the arm lags them by about 0.1 s.
* `figures/paths`: commanded and achieved hand paths, top view of the square and circle, side view
  of the elbow limit reach.

Tables: `tables/summary.csv`, `tables/summary.tex` (booktabs).
"""
    with open(os.path.join(out, 'README.md'), 'w') as f:
        f.write(text)


def main():
    out, bags = sys.argv[1], sys.argv[2:]
    for sub in ('figures', 'tables'):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    plt.rcParams.update(STYLE)
    runs = [run(bag) for bag in bags]
    rows = [row for r in runs for row in phase_metrics(r)]
    order = [PHASE_NAMES.get(p[0], p[0]) for p in runs[0]['phases']] + ['Whole run']
    rows.sort(key=lambda row: order.index(row['phase']))
    figure_tracking(runs, out)
    figure_configuration(runs, out)
    figure_joints(runs, out)
    figure_paths(runs, out)
    write_tables(rows, out)
    write_readme(runs, rows, out)
    with open(os.path.join(out, 'metrics.json'), 'w') as f:
        json.dump(rows, f, indent=1, default=float)
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()

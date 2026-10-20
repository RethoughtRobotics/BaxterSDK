"""How closely replays (baxter_teleop.replay) reproduced their demo: errors, lag, torques, figures.

write_report(out, demo, runs, arms, settings) writes into `out`:
  README.md      a table per arm and mode: joint and TCP error (rms, max), the lag that best aligns
                 the replay with the demo, and the measured joint torque difference
  summary.yaml   the same numbers, and the replay settings
  <arm>_joints.png, <arm>_tcp_error.png, <arm>_torque.png
  replay_<run>.npz  each run's log: per arm t, q, qd, effort, wrench_t, wrench
Runs are labelled by mode ('pose', 'force'), with '#k' for repeats of one command (the run-to-run
table: how far the repeats are from the first, the floor any replay can reach) and 'ilcK' for
learning iterations (the convergence figure).
Errors compare the replay at each demo instant (replay time x speed = demo time). The TCP is the
URDF gripper frame for both, from the same kinematics, so it measures the motion, not the TCP offset.
"""

from pathlib import Path

import numpy as np
import yaml

from ..kinematics import ArmKinematics
from ..settings import JOINTS

LAGS = np.arange(-0.3, 0.3001, 0.005)  # s, searched for the best alignment


def _at(t, t_src, values):
    return np.stack([np.interp(t, t_src, values[:, j]) for j in range(values.shape[1])], axis=1)


def compare(demo_arm, run_arm, speed, kinematics):
    """Metrics of one arm's replay against its demo."""
    t_replay = run_arm['t'] * speed
    t = demo_arm['t'][(demo_arm['t'] >= t_replay[0]) & (demo_arm['t'] <= t_replay[-1])]
    q_demo = _at(t, demo_arm['t'], demo_arm['q'])
    q_replay = _at(t, t_replay, run_arm['q'])
    err = np.degrees(q_replay - q_demo)
    tcp_demo = np.array([kinematics.tcp(q)[0] for q in q_demo[::2]])
    tcp_replay = np.array([kinematics.tcp(q)[0] for q in q_replay[::2]])
    tcp_err = 1000.0 * np.linalg.norm(tcp_replay - tcp_demo, axis=1)
    rms_at = [np.sqrt(np.mean((_at(t, t_replay - lag, run_arm['q']) - q_demo) ** 2)) for lag in LAGS]
    out = {
        'joint_error_rms_deg': float(np.sqrt(np.mean(err**2))),
        'joint_error_max_deg': float(np.max(np.abs(err))),
        'joint_error_rms_deg_per_joint': dict(zip(JOINTS, np.sqrt(np.mean(err**2, axis=0)).round(3).tolist())),
        'tcp_error_rms_mm': float(np.sqrt(np.mean(tcp_err**2))),
        'tcp_error_max_mm': float(np.max(tcp_err)),
        'lag_ms': float(1000.0 * LAGS[int(np.argmin(rms_at))]),
    }
    if np.isfinite(demo_arm['effort']).all() and np.isfinite(run_arm['effort']).all() and len(run_arm['effort']):
        tau_diff = _at(t, t_replay, run_arm['effort']) - _at(t, demo_arm['t'], demo_arm['effort'])
        out['torque_difference_rms_nm'] = float(np.sqrt(np.mean(tau_diff**2)))
    series = {'t': t, 'q_demo': q_demo, 'q_replay': q_replay, 't_tcp': t[::2], 'tcp_error_mm': tcp_err}
    return out, series


def between(run_a, run_b, t, speed, kinematics):
    """TCP distance (mm, rms and max) between two runs at demo times t."""
    tcp = [np.array([kinematics.tcp(q)[0] for q in _at(t, r['t'] * speed, r['q'])[::2]]) for r in (run_a, run_b)]
    d = 1000.0 * np.linalg.norm(tcp[0] - tcp[1], axis=1)
    return float(np.sqrt(np.mean(d**2))), float(np.max(d))


def color(label, index, count):
    """Pose blue, force orange; repeats and iterations from light to dark."""
    base = matplotlib_colors()['tab:blue' if label.startswith('pose') else 'tab:orange']
    shade = 0.35 + 0.65 * (index + 1) / count
    return tuple(1.0 - shade * (1.0 - c) for c in base)


def matplotlib_colors():
    from matplotlib.colors import TABLEAU_COLORS, to_rgb

    return {name: to_rgb(value) for name, value in TABLEAU_COLORS.items()}


def write_report(out, demo, runs, arms, settings):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    speed = float(settings.get('speed', 1.0))
    for label, run in runs.items():  # the raw logs, for analysis beyond this report
        name = label.replace(' #', '_').replace(' ', '_')
        np.savez(out / f'replay_{name}.npz', **{f'{arm}_{k}': v for arm, d in run.items() for k, v in d.items()})
    summary, lines = {'settings': settings, 'results': {}}, []
    same_mode = {label: [lb for lb in runs if lb.split()[0] == label.split()[0]] for label in runs}
    colors = {label: color(label, same_mode[label].index(label), len(same_mode[label])) for label in runs}
    for arm in arms:
        kinematics = ArmKinematics(arm)
        results, series = {}, {}
        for mode, run in runs.items():
            results[mode], series[mode] = compare(demo[arm], run[arm], speed, kinematics)
        summary['results'][arm] = results

        fig, axes = plt.subplots(7, 1, figsize=(11, 13), sharex=True)
        for j, ax in enumerate(axes):
            ax.plot(demo[arm]['t'], np.degrees(demo[arm]['q'][:, j]), color='black', lw=1.6, label='demo')
            for mode, s in series.items():
                ax.plot(s['t'], np.degrees(s['q_replay'][:, j]), color=colors[mode], lw=0.9, label=mode)
            ax.set_ylabel(f'{JOINTS[j]} (deg)')
        axes[0].legend(loc='upper right', ncol=3)
        axes[-1].set_xlabel('demo time (s)')
        fig.suptitle(f'{arm} arm: joint angles, demo and replays')
        fig.tight_layout()
        fig.savefig(out / f'{arm}_joints.png', dpi=110)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 3.5))
        for mode, s in series.items():
            ax.plot(s['t_tcp'], s['tcp_error_mm'], color=colors[mode], lw=0.9, label=mode)
        ax.set_xlabel('demo time (s)')
        ax.set_ylabel('TCP error (mm)')
        ax.set_title(f'{arm} arm: distance of the replayed TCP from the demo TCP')
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f'{arm}_tcp_error.png', dpi=110)
        plt.close(fig)

        if all(np.isfinite(r[arm]['effort']).all() and len(r[arm]['effort']) for r in runs.values()):
            fig, axes = plt.subplots(7, 1, figsize=(11, 13), sharex=True)
            for j, ax in enumerate(axes):
                ax.plot(demo[arm]['t'], demo[arm]['effort'][:, j], color='black', lw=1.6, label='demo')
                for mode, run in runs.items():
                    ax.plot(run[arm]['t'] * speed, run[arm]['effort'][:, j], color=colors[mode], lw=0.9, label=mode)
                ax.set_ylabel(f'{JOINTS[j]} (N m)')
            axes[0].legend(loc='upper right', ncol=3)
            axes[-1].set_xlabel('demo time (s)')
            fig.suptitle(f'{arm} arm: measured joint torque, demo and replays')
            fig.tight_layout()
            fig.savefig(out / f'{arm}_torque.png', dpi=110)
            plt.close(fig)

        lines += [
            f'## {arm} arm',
            '',
            '| mode | joint rms (deg) | joint max (deg) | TCP rms (mm) | TCP max (mm) | lag (ms) | torque diff rms (N m) |',
            '|---|---:|---:|---:|---:|---:|---:|',
        ]
        for mode, r in results.items():
            torque = f'{r["torque_difference_rms_nm"]:.2f}' if 'torque_difference_rms_nm' in r else '-'
            lines.append(
                f'| {mode} | {r["joint_error_rms_deg"]:.2f} | {r["joint_error_max_deg"]:.2f} | '
                f'{r["tcp_error_rms_mm"]:.1f} | {r["tcp_error_max_mm"]:.1f} | {r["lag_ms"]:.0f} | {torque} |'
            )
        repeats = [lb for lb in runs if ' #' in lb]
        if repeats:
            t_demo = demo[arm]['t'][demo[arm]['t'] <= min(r[arm]['t'][-1] * speed for r in runs.values())]
            lines += ['', '| repeat | TCP from the first run, rms (mm) | max (mm) |', '|---|---:|---:|']
            for label in repeats:
                first = label.split(' #')[0] + ' #1'
                if label != first:
                    rms, peak = between(runs[first][arm], runs[label][arm], t_demo, speed, kinematics)
                    results[label]['tcp_from_first_run_rms_mm'] = rms
                    lines.append(f'| {label} | {rms:.2f} | {peak:.2f} |')
        learning = [lb for lb in runs if ' ilc' in lb]
        if learning:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            for mode in sorted({lb.split()[0] for lb in learning}):
                its = [lb for lb in learning if lb.startswith(mode)]
                ax.plot(range(len(its)), [results[lb]['tcp_error_rms_mm'] for lb in its], 'o-', label=mode,
                        color=colors[its[-1]])  # fmt: skip
            ax.set_xlabel('learning iteration')
            ax.set_ylabel('TCP error rms (mm)')
            ax.set_title(f'{arm} arm: replay error as the command learns')
            ax.set_ylim(bottom=0)
            ax.legend()
            fig.tight_layout()
            fig.savefig(out / f'{arm}_learning.png', dpi=110)
            plt.close(fig)
        lines += ['', f'![joints]({arm}_joints.png)', '', f'![TCP error]({arm}_tcp_error.png)', '']
        if learning:
            lines += [f'![learning]({arm}_learning.png)', '']
        if (out / f'{arm}_torque.png').exists():
            lines += [f'![torque]({arm}_torque.png)', '']

    header = [
        f'# Replay of {Path(settings["episode"]).name}',
        '',
        f'Modes: {", ".join(runs)}. Force mode bandwidth {settings.get("omega")} rad/s, speed {speed}, '
        f'lead {settings.get("lead")} s. Lag: positive means the replay trails the demo.',
        '',
    ]
    (out / 'README.md').write_text('\n'.join(header + lines))
    (out / 'summary.yaml').write_text(yaml.safe_dump(summary, sort_keys=False))
    return (
        '\n'.join(lines[i] for i in range(len(lines)) if lines[i].startswith(('##', '|')))
        + f'\nReport: {out}/README.md'
    )

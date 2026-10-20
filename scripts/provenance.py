#!/usr/bin/env python3
"""Record what was running when a recording started, so a replay can reproduce it exactly.

  python3 scripts/provenance.py write BAG_DIR   BAG_DIR/provenance.yaml (+ provenance/<repo>.patch)
  python3 scripts/provenance.py custom-data     key=value pairs for ros2 bag record --custom-data

provenance.yaml holds, at the time of the call: every repository's commit, branch and remote, with
its uncommitted changes as a patch when dirty; the command line of every process of this stack
(the teleop with its arguments, the policy's serve and run, the sim, the recorder); the teleop
settings file the teleop reads (teleop.yaml) verbatim; the robot bridge and VR containers' images;
and the host, ROS distro, domain and middleware. The custom data is the short form, stored in the
bag's own metadata.yaml (rosbag2_bagfile_information/custom_data), which every bag reader shows.
"""

import datetime
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
WS = HERE.parent
REPOS = {
    'baxter_sdk': WS,
    'rethought_ai': WS.parent / 'rethought_ai',
    'libsurvive_ros2': WS.parent / 'libsurvive_ros2',
    'baxter_zenoh': Path.home() / 'baxter-zenoh',
}


def run(*command, cwd=None, ok=(0,)):
    """stdout of a command, '' if it fails (exit status not in ok)."""
    try:
        out = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return ''
    return out.stdout.strip() if out.returncode in ok else ''


def ancestors():
    """This process and the ones that started it (the recorder, the shells): not stack processes."""
    pids, pid = set(), os.getpid()
    while pid > 1:
        pids.add(str(pid))
        try:
            pid = int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    return pids


def repo(path):
    if not (path / '.git').exists():
        return None
    status = run('git', 'status', '--porcelain', cwd=path)  # untracked files count: they may be what ran
    return {
        'path': str(path),
        'commit': run('git', 'rev-parse', 'HEAD', cwd=path),
        'branch': run('git', 'rev-parse', '--abbrev-ref', 'HEAD', cwd=path),
        'remote': run('git', 'remote', 'get-url', 'origin', cwd=path),
        'dirty': bool(status),
    }


def stack_processes():
    """Command lines of this stack's processes (scripts/env.sh STACK_PROCESSES), not this one's."""
    pattern = os.environ.get('STACK_PROCESSES')
    if not pattern:
        return []
    mine = ancestors()
    lines = run('pgrep', '-af', '--', pattern).splitlines()
    return sorted(line.split(' ', 1)[1] for line in lines if ' ' in line and line.split(' ', 1)[0] not in mine)


def containers():
    """Running robot bridge and VR containers: name, image and image id."""
    rows = run('docker', 'ps', '--format', '{{.Names}}\t{{.Image}}\t{{.ID}}').splitlines()
    found = {}
    for row in rows:
        name, image, cid = (row.split('\t') + ['', '', ''])[:3]
        if 'baxter-zenoh' in image or 'libsurvive' in image or 'libsurvive' in name:
            found[name] = {'image': image, 'image_id': run('docker', 'inspect', '-f', '{{.Image}}', cid)}
    return found


def teleop_settings():
    """The teleop.yaml the teleop reads (the installed copy), verbatim."""
    for candidate in (WS / 'install/baxter_teleop/share/baxter_teleop/config/teleop.yaml',
                      WS / 'src/baxter_teleop/config/teleop.yaml'):  # fmt: skip
        if candidate.exists():
            return {'path': str(candidate), 'text': candidate.read_text()}
    return None


def collect():
    return {
        'recorded': datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
        'host': socket.gethostname(),
        'user': os.environ.get('USER', ''),
        'kernel': platform.release(),
        'ros': {
            'distro': os.environ.get('ROS_DISTRO', ''),
            'domain_id': int(os.environ.get('ROS_DOMAIN_ID') or 0),
            'rmw': os.environ.get('RMW_IMPLEMENTATION', ''),
        },
        'repos': {name: info for name, path in REPOS.items() if (info := repo(path)) is not None},
        'processes': stack_processes(),
        'containers': containers(),
        'teleop_settings': teleop_settings(),
    }


def write(bag_dir):
    bag_dir = Path(bag_dir)
    info = collect()
    for name, r in info['repos'].items():
        if r['dirty']:  # the uncommitted changes and new files that ran: git apply on the commit reproduces them
            patch = bag_dir / 'provenance' / f'{name}.patch'
            patch.parent.mkdir(exist_ok=True)
            untracked = run('git', 'ls-files', '--others', '--exclude-standard', cwd=r['path']).splitlines()
            new = [run('git', 'diff', '--no-index', '/dev/null', f, cwd=r['path'], ok=(1,)) for f in untracked]
            patch.write_text('\n'.join([run('git', 'diff', 'HEAD', cwd=r['path']), *new]) + '\n')
            r['patch'] = str(patch.relative_to(bag_dir))
    (bag_dir / 'provenance.yaml').write_text(yaml.safe_dump(info, sort_keys=False, width=120))


def custom_data():
    info = collect()
    pairs = {'provenance': 'provenance.yaml', 'host': info['host'], 'ros_domain_id': info['ros']['domain_id']}
    for name, r in info['repos'].items():
        pairs[f'git_{name}'] = r['commit'][:12] + ('-dirty' if r['dirty'] else '')
    for name, c in info['containers'].items():
        pairs[f'container_{name}'] = c['image']
    print(' '.join(f'{k}={str(v).replace(" ", "_")}' for k, v in pairs.items()))


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == 'write':
        write(sys.argv[2])
    elif len(sys.argv) == 2 and sys.argv[1] == 'custom-data':
        custom_data()
    else:
        sys.exit(__doc__)

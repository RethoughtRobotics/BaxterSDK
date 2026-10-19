"""The teleop's settings file, config/teleop.yaml: its tuned defaults and Baxter's joint limits.

Read once, at import: from the source tree when running from it (tests), else from the installed
package share. DEFAULTS maps option names (servo_feedback) to values; LIMITS holds the joint limits
as tuples in JOINTS order.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

JOINTS = ('s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2')


@dataclass(frozen=True)
class JointLimitTable:
    velocity: tuple  # rad/s
    acceleration: tuple  # rad/s^2
    position_low: tuple  # rad
    position_high: tuple  # rad

    def __post_init__(self):
        for name, values in vars(self).items():
            if len(values) != len(JOINTS):
                raise ValueError(f'teleop.yaml joint_limits.{name}: {len(JOINTS)} values, one per joint, got {values}')


def joint_names(arm):
    """The arm's joint names in joint order, e.g. ['left_s0', ..., 'left_w2']."""
    return [f'{arm}_{joint}' for joint in JOINTS]


def settings_path():
    source = Path(__file__).resolve().parents[1] / 'config' / 'teleop.yaml'
    if source.exists():
        return source
    from ament_index_python.packages import get_package_share_directory

    return Path(get_package_share_directory('baxter_teleop')) / 'config' / 'teleop.yaml'


with open(settings_path()) as f:
    _settings = yaml.safe_load(f)
DEFAULTS = dict(_settings['defaults'])
LIMITS = JointLimitTable(**{name: tuple(float(x) for x in v) for name, v in _settings['joint_limits'].items()})


def servo_feedback_for(output, servo_feedback=None):
    """servo_feedback as given, else the default, except measured for the velocity output (which has
    no setpoint to close on), so --output velocity alone works."""
    if servo_feedback is not None:
        return servo_feedback
    return 'measured' if output == 'velocity' else DEFAULTS['servo_feedback']

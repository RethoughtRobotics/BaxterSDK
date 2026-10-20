"""BlueCupRedContainer: pick the cup up and put it in the container.

Grasp height and orientation are the real demonstrations' (BlueCupRedContainer_kin_pose, 56 episodes:
the gripper closes at TCP z = -0.045 m, from straight above); layouts lie inside their cup and release
ranges (scenes.sample_layout).
"""

import numpy as np
from scipy.spatial.transform import Rotation

from baxter_mujoco import scenes
from baxter_mujoco.datagen.task import Task, Waypoint

GRASP_Z = -0.045  # m, TCP height at grasp
GRASP_QUAT = (-0.0596, 0.998, 0.001, 0.0192)  # TCP orientation at grasp: from straight above
HOVER = 0.12  # m above grasp height, for the approach and the carry
RELEASE_Z = 0.02  # m, TCP height over the container: the cup's base 2 cm above its rim
CLOSE_HOLD, OPEN_HOLD = 1.0, 0.8  # s, holding still while the gripper closes, opens


class CupTask(Task):
    name, scene = 'cup', 'cup'

    def sample(self, rng):
        cup, box, yaw = scenes.sample_layout(rng)
        return [*cup, *box, yaw]

    def plan(self, layout, home):
        cup, box = np.array(layout[0:2]), np.array(layout[2:4])
        down = Rotation.from_quat(GRASP_QUAT)

        def at(xy, z):
            return np.array([xy[0], xy[1], z])

        return [
            Waypoint(at(cup, GRASP_Z + HOVER), down),
            Waypoint(at(cup, GRASP_Z), down, slow=True),
            Waypoint(at(cup, GRASP_Z), down, grip=1.0, hold=CLOSE_HOLD),
            Waypoint(at(cup, GRASP_Z + HOVER), down, grip=1.0, slow=True),
            Waypoint(at(box, GRASP_Z + HOVER), down, grip=1.0),
            Waypoint(at(box, RELEASE_Z), down, grip=1.0, slow=True),
            Waypoint(at(box, RELEASE_Z), down, grip=0.0, hold=OPEN_HOLD),
            Waypoint(at(box, GRASP_Z + HOVER), down, slow=True),
            Waypoint(home[0], home[1], hold=0.5),
        ]

    def success(self, layout, objects):
        box, yaw = np.array(layout[2:4]), layout[4]
        cup = np.asarray(objects['cup'])
        d = Rotation.from_euler('z', -yaw).apply([cup[0] - box[0], cup[1] - box[1], 0.0])
        inside = max(abs(d[0]), abs(d[1])) < scenes.CONTAINER_HALF - scenes.CUP_RADIUS / 2
        return bool(inside and cup[2] > scenes.TABLE_TOP)

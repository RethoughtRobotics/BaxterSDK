"""BlueCupRedContainer: pick the cup up and put it in the container.

Grasped from straight above by its wall, one finger inside the cup, as in the real demonstrations
(BlueCupRedContainer_kin_pose, 56 episodes; the gripper opens 4.2 cm, less than the cup). Their TCP is
at the fingertips, which close 1.5 cm above the table, but the real extended fingers reach about 11 cm
below the palm and the sim gripper's only 6.2 cm: grasped that low in the sim, the rim jams into the
palm and the cup stays wedged when the fingers open. So the sim grasps the wall's upper part, the rim
clear of the palm. Layouts lie inside the real cup and release ranges (scenes.sample_layout).
"""

import numpy as np
from scipy.spatial.transform import Rotation

from baxter_mujoco import scenes
from baxter_mujoco.datagen.task import Task, Waypoint

GRASP_DEPTH = 0.047  # m, fingertips below the cup's rim: the rim 1.5 cm clear of the sim gripper's palm
GRASP_Z = scenes.TABLE_TOP + scenes.CUP_HEIGHT - GRASP_DEPTH  # m, TCP (fingertips) height at grasp
GRASP_QUAT = (-0.0596, 0.998, 0.001, 0.0192)  # TCP orientation at grasp: from straight above
# Grasp point from the cup's centre: the middle of its wall along the fingers' closing axis (the
# TCP's y), on the +y side, so the hand does not reach over the cup
WALL = scenes.CUP_RADIUS - scenes.CUP_WALL / 2  # m
HOVER = 0.12  # m above grasp height, for the approach and the carry
RELEASE_CLEARANCE = 0.01  # m, the cup's base above the container's rim when let go
RELEASE_Z = scenes.TABLE_TOP + scenes.CONTAINER_HEIGHT + RELEASE_CLEARANCE + (GRASP_Z - scenes.TABLE_TOP)
CLOSE_HOLD, OPEN_HOLD = 1.0, 0.8  # s, holding still while the gripper closes, opens
# The right arm in every real demonstration (all 56 within 0.004 rad), out of the left arm's way
RIGHT_PARKED = (-0.3747, -1.2165, -0.4253, 1.5647, 0.3527, 1.3000, -1.7491)  # rad, s0 .. w2


class CupTask(Task):
    name, scene = 'cup', 'cup'
    parked = {'right': RIGHT_PARKED}

    def sample(self, rng):
        cup, box, yaw = scenes.sample_layout(rng)
        return [*cup, *box, yaw]

    def plan(self, layout, home):
        down = Rotation.from_quat(GRASP_QUAT)
        to_wall = WALL * down.apply([0.0, 1.0, 0.0])[:2]
        cup, box = np.array(layout[0:2]) + to_wall, np.array(layout[2:4]) + to_wall

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

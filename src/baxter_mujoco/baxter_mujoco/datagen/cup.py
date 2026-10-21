"""BlueCupRedContainer: pick the cup up and put it in the container.

Grasped from straight above by its wall, one finger inside the cup, as in the real demonstrations
(BlueCupRedContainer_kin_pose, 56 episodes; the gripper opens 4.2 cm, less than the cup). Their TCP is
at the fingertips, which close 1.5 cm above the table, but the real extended fingers reach about 11 cm
below the palm and the sim gripper's only 6.2 cm: grasped that low in the sim, the rim jams into the
palm and the cup stays wedged when the fingers open. So the sim grasps the wall's upper part, the rim
clear of the palm. Layouts lie inside the real cup and release ranges (scenes.sample_layout).

Mid-attempt (CupTask.replan) the expert resumes the same plan from the state the arm and the cup are
in: the cup held, it carries it on (lifting it first if it is low); the cup let go in the container,
it finishes opening and goes home; the cup on the table, it opens the gripper if it is closed (a
missed grasp), rises to the carry height if it is low and away from the cup, and grasps it. The cup
fallen over or off the table has no plan.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from baxter_mujoco import scenes
from baxter_mujoco.datagen.outcome import HOLD, LIFT, cup_centre
from baxter_mujoco.datagen.task import Replan, Task, Waypoint

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
# The measured gripper (0 open .. 1 closed) closed on the cup's wall: BlueCupRedContainer_sim's labels
# hold 0.87 while the cup is carried
GRIP_CLOSED = 0.87
GRIPPING = 0.5  # gripper (0 open .. 1 closed) above which the hand holds, or tries to hold, the cup
AT = 0.01  # m, the TCP at a waypoint
GRIP_DONE = 0.05  # of a close or open left: done (the measured gripper drifts by about this much)
UPRIGHT = 0.87  # cos 30 deg: the cup's axis this close to vertical can be grasped from above
# The right arm in every real demonstration (all 56 within 0.004 rad), out of the left arm's way
RIGHT_PARKED = (-0.3747, -1.2165, -0.4253, 1.5647, 0.3527, 1.3000, -1.7491)  # rad, s0 .. w2


class CupTask(Task):
    name, scene = 'cup', 'cup'
    parked = {'right': RIGHT_PARKED}
    grip_closed = GRIP_CLOSED

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

    def replan(self, tcp, gripper, objects, home, layout=None):
        down = Rotation.from_quat(GRASP_QUAT)
        to_wall = WALL * down.apply([0.0, 1.0, 0.0])[:2]
        here = np.asarray(tcp[0], float)
        cup, cup_quat = np.asarray(objects['cup'][0], float), objects['cup'][1]
        box = np.asarray(objects['container'][0], float)
        grasp, release = cup[:2] + to_wall, box[:2] + to_wall
        hover = GRASP_Z + HOVER

        def at(xy, z):
            return np.array([xy[0], xy[1], z])

        def over(xy):
            return np.linalg.norm(here[:2] - xy) < AT

        def share(x):  # of a close or open hold still to go, from how far the gripper has moved
            return float(np.clip(x, 0.0, 1.0)) if x > GRIP_DONE else 0.0

        def start(position):  # where the expert's move under way began
            return position, down

        carry = [
            Waypoint(at(release, hover), down, grip=1.0),
            Waypoint(at(release, RELEASE_Z), down, grip=1.0, slow=True),
            Waypoint(at(release, RELEASE_Z), down, grip=0.0, hold=OPEN_HOLD),
            Waypoint(at(release, hover), down, slow=True),
            Waypoint(home[0], home[1], hold=0.5),
        ]
        low = here[2] < hover - AT
        held = gripper > GRIPPING and np.linalg.norm(here - cup_centre(cup, cup_quat)) < HOLD
        if held:
            if over(release) and abs(here[2] - RELEASE_Z) < AT:  # at the release: open
                wait = OPEN_HOLD * share(gripper / GRIP_CLOSED)
                return Replan([Waypoint(at(here, hover), down, slow=True), carry[-1]], grip=0.0, wait=wait)
            if over(release):  # above it: lower the cup
                return Replan(carry[1:], grip=1.0, origin=start(at(release, hover)))
            wait = 0.0 if cup[2] > scenes.TABLE_TOP + LIFT else CLOSE_HOLD * share(1 - gripper / GRIP_CLOSED)
            if low:  # lift it first
                lift = Waypoint(at(here, hover), down, grip=1.0, slow=True)
                return Replan([lift, *carry], grip=1.0, wait=wait, origin=start(at(here, GRASP_Z)))
            lifted = start(at(np.asarray(layout[0:2]) + to_wall, hover)) if layout is not None else None
            return Replan(carry, grip=1.0, wait=wait, origin=lifted)
        if self.success(self.layout(objects), {'cup': cup}) and cup[2] < scenes.TABLE_TOP + scenes.CONTAINER_HEIGHT:
            wait = OPEN_HOLD * share(gripper / GRIP_CLOSED)  # let go in the container: finish opening
            if low:
                rise = Waypoint(at(here, hover), down, slow=True)
                return Replan([rise, carry[-1]], grip=0.0, wait=wait, origin=start(at(here, RELEASE_Z)))
            return Replan(carry[-1:], grip=0.0, wait=wait, origin=start(at(release, hover)))
        upright = Rotation.from_quat(cup_quat).apply([0.0, 0.0, 1.0])[2] > UPRIGHT
        standing = abs(cup[2] - scenes.TABLE_TOP) < LIFT
        if not (upright and standing):
            return None
        grasped = [
            Waypoint(at(grasp, GRASP_Z), down, grip=1.0, hold=CLOSE_HOLD),
            Waypoint(at(grasp, hover), down, grip=1.0, slow=True),
            *carry,
        ]
        if over(grasp) and abs(here[2] - GRASP_Z) < AT:  # at the grasp: close
            return Replan(grasped[1:], grip=1.0, wait=CLOSE_HOLD * share(1 - gripper / GRIP_CLOSED))
        wait = OPEN_HOLD * share(gripper / GRIP_CLOSED)  # a missed grasp: open first
        descend = Waypoint(at(grasp, GRASP_Z), down, slow=True)
        if over(grasp) and here[2] < hover + AT:
            return Replan([descend, *grasped], wait=wait, origin=start(at(grasp, hover)))
        approach = [Waypoint(at(grasp, hover), down), descend, *grasped]
        if low:  # rise clear of the cup first
            return Replan([Waypoint(at(here, hover), down, slow=True), *approach], wait=wait)
        return Replan(approach, wait=wait, origin=home)

    def layout(self, objects):
        cup, (box, box_quat) = objects['cup'][0], objects['container']
        return [*cup[:2], *box[:2], float(Rotation.from_quat(box_quat).as_euler('xyz')[2])]

    def success(self, layout, objects):
        box, yaw = np.array(layout[2:4]), layout[4]
        cup = np.asarray(objects['cup'])
        d = Rotation.from_euler('z', -yaw).apply([cup[0] - box[0], cup[1] - box[1], 0.0])
        inside = max(abs(d[0]), abs(d[1])) < scenes.CONTAINER_HALF - scenes.CUP_RADIUS / 2
        return bool(inside and cup[2] > scenes.TABLE_TOP)

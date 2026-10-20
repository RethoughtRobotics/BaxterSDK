"""The MuJoCo Baxter behind the real robot's ROS 2 interface (baxter_core_msgs).

Run `./launch_sim.sh --ros`. baxter_interface (teleop, tuck_arms) and RViz then
drive and show this simulated Baxter exactly as they do the real one, over the same topics:

  publishes   /robot/joint_states, /robot/limb/{arm}/endpoint_state, /robot/state,
              /robot/head/head_state, /robot/end_effector/{arm}_gripper/{state,properties},
              /sim/config (scene, latched), /sim/scene (RViz markers of
              everything else the arms can hit: the table, objects);
              with --no-viewer /cameras/{arm}_hand_camera/image (640 x 400 bgra8, as the robot)
              at 25 Hz while subscribed; with --scene cup also /sim/objects (cup and container
              poses) and, with --no-viewer, the lab's third_eye (lab.toml)
  subscribes  /robot/limb/{arm}/joint_command (position, velocity, torque, raw position modes),
              /robot/limb/{arm}/joint_command_timeout, /robot/limb/{arm}/set_speed_ratio,
              /robot/set_super_enable, /robot/head/command_head_pan,
              /robot/end_effector/{arm}_gripper/command (electric gripper: calibrate, go, ...);
              with --scene cup also /sim/layout ([cup x, y, container x, y, container yaw] in
              the base frame: places both on the table, at rest) and /sim/lighting (the lab's
              lights moved, scaled, tinted, shadows on or off: scenes.apply_lighting)

Baxter's own joint controllers are emulated with computed torque plus gravity compensation
(critically damped at OMEGA, about 0.1 s of lag like the real arm) inside the actuators' torque
limits. The head (not in the MuJoCo model) pans kinematically. As on the robot, position targets persist, while velocity and torque commands older than
the command timeout drop back to holding the current position. The endpoint is at the fingertips,
where the real robot reports it with its electric gripper, in the robot base frame.
"""

import json
import os
import threading
import time

import mujoco
import numpy as np
import rclpy
from baxter_core_msgs.msg import (
    AssemblyState,
    EndEffectorCommand,
    EndEffectorProperties,
    EndEffectorState,
    EndpointState,
    HeadPanCommand,
    HeadState,
    JointCommand,
)
from geometry_msgs.msg import Pose, PoseArray
from rclpy.executors import ExternalShutdownException
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool, Float64, Float64MultiArray, String
from visualization_msgs.msg import Marker, MarkerArray

from baxter_mujoco import scenes
from baxter_mujoco.sim import make_env

ARMS = ('left', 'right')
JOINTS = ('s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2')
MAX_SPEED = np.array([1.5, 1.5, 1.5, 1.5, 4.0, 4.0, 4.0])  # rad/s, URDF velocity limits
HEAD_MAX_SPEED = 1.0  # rad/s
OMEGA = 20.0  # rad/s, joint servo bandwidth (critically damped)
# Endpoint in {arm}_hand: the sim gripper's fingertip ends (robosuite's Rethink gripper,
# gripper0_{arm}_{l,r}_fingertip_g0). The real robot reports its endpoint at its own fingertips
# (wrist + 0.281 m in the BlueCupRedContainer_kin demos), so recorded TCP heights carry over.
TCP_IN_HAND = np.array([0.0, 0.0, 0.123])
SCENE_SHAPES = {
    int(mujoco.mjtGeom.mjGEOM_BOX): Marker.CUBE,
    int(mujoco.mjtGeom.mjGEOM_CYLINDER): Marker.CYLINDER,
    int(mujoco.mjtGeom.mjGEOM_SPHERE): Marker.SPHERE,
}
SCENE_COLORS = {'static': (0.62, 0.60, 0.55, 0.85), 'free': (0.80, 0.42, 0.22, 0.9)}  # table / movable objects
DT_PUBLISH = 0.01  # s, state publishing period (100 Hz)
CAMERA_EVERY = 4  # publishing periods per camera frame: 25 Hz, the datasets' rate
# The hand cameras as mounted on the robot (baxter.urdf {arm}_hand_camera, in {arm}_hand): looking out
# along the fingers, which are at the image top; field of view from their camera_info (fy 403.9 px of
# 800 rows). Published as the robot does, 640 x 400 bgra8.
HAND_CAMERA_POS = (0.03825, 0.012, 0.015355)  # m
HAND_CAMERA_AXES = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]).T  # columns: image right, up, back
HAND_CAMERA_FOVY = 89.4  # deg
HAND_CAMERA_SIZE = (640, 400)  # px
FINGER_JOINTS = (
    'l_gripper_l_finger_joint',
    'l_gripper_r_finger_joint',
    'r_gripper_l_finger_joint',
    'r_gripper_r_finger_joint',
)


def torso_ignores_arms(m):
    """Stop the torso's collision shape from touching the arms; objects and the table still hit it.

    The torso collides as the convex hull of one concave mesh (base_link_collision), which fills
    space the real arms move through: a hand-guided demo on the robot (SweepTrash_Kin episode 0)
    puts the upper elbow 47 mm inside it without contact. Contact needs contype & conaffinity in
    either direction; the arms have contype 2 or 4, everything else contype 1, so a torso with
    contype 0 and conaffinity 1 still collides with all but the arms.
    """
    body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'robot0_torso')
    for g in np.flatnonzero(m.geom_bodyid == body):
        if m.geom_contype[g] or m.geom_conaffinity[g]:
            m.geom_contype[g], m.geom_conaffinity[g] = 0, 1


def remove_table(m):
    """Take the table out of the scene: no collisions, and hidden (geom group 5 is not drawn)."""
    body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'table')
    for g in np.flatnonzero(m.geom_bodyid == body):
        m.geom_contype[g] = m.geom_conaffinity[g] = 0
        m.geom_group[g] = 5


class Gripper:
    """One electric parallel gripper on the real robot's end-effector topics.

    Position is Baxter's 0 (closed) .. 100 (open). Like the real gripper it starts uncalibrated
    and refuses to move until 'calibrate'; 'go' moves toward a position at the configured
    velocity (percent of full speed, full stroke FULL_STROKE_S at 100). Each command is
    acknowledged by echoing its sender and sequence in the state, which baxter_interface waits on.
    """

    # The real gripper's stroke: each finger travels URDF_STROKE (rethink_electric_gripper.xacro), from
    # the paddle tips touching to 4.2 cm apart, too narrow to span a cup, so the cup is held by its wall.
    # robosuite's model closes only to 1.55 cm between the pads and opens to 7.7 cm, so the r finger
    # joint (the l one mirrors it) is given the real stroke, ending where the pads touch.
    URDF_STROKE = 0.020833  # m
    CLOSED = 0.0193  # m, r finger joint with the pads touching
    OPEN = CLOSED - URDF_STROKE
    FULL_STROKE_S = 0.5
    ID = 65538

    def __init__(self, node, m, side):
        self.acts = [
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f'gripper0_{side}_gripper_{f}_finger_joint')
            for f in ('r', 'l')
        ]
        joints = [
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f'gripper0_{side}_{f}_finger_joint') for f in ('r', 'l')
        ]
        for joint, act, sign in zip(joints, self.acts, (1, -1)):
            m.jnt_range[joint] = m.actuator_ctrlrange[act] = sorted((sign * self.OPEN, sign * self.CLOSED))
        self.joint = m.jnt_qposadr[joints[0]]
        self.calibrated, self.target, self.command_pct = False, 100.0, 100.0
        self.velocity, self.dead_zone = 50.0, 5.0
        self.ack = ('', '', 0)
        ns = f'/robot/end_effector/{side}_gripper/'
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.state_pub = node.create_publisher(EndEffectorState, ns + 'state', latched)
        props = node.create_publisher(EndEffectorProperties, ns + 'properties', latched)
        props.publish(
            EndEffectorProperties(
                id=self.ID,
                ui_type=EndEffectorProperties.ELECTRIC_GRIPPER,
                manufacturer='Rethink Research Robot (sim)',
                product='Electric Parallel Gripper',
                firmware_date='2014/07/07 00:00:00',
                has_calibration=True,
                controls_grip=True,
                senses_grip=True,
                controls_force=True,
                senses_force=True,
                controls_position=True,
                senses_position=True,
            )
        )
        node.create_subscription(EndEffectorCommand, ns + 'command', self.on_command, QoSProfile(depth=1))

    def on_command(self, msg):
        args = json.loads(msg.args) if msg.args else {}
        if msg.command == EndEffectorCommand.CMD_CALIBRATE:
            self.calibrated, self.target = True, 100.0
        elif msg.command == EndEffectorCommand.CMD_CLEAR_CALIBRATION:
            self.calibrated = False
        elif msg.command == EndEffectorCommand.CMD_CONFIGURE:
            self.velocity = float(args.get('velocity', self.velocity))
            self.dead_zone = float(args.get('dead_zone', self.dead_zone))
        elif msg.command == EndEffectorCommand.CMD_GO and self.calibrated:
            self.target = float(np.clip(args.get('position', self.target), 0.0, 100.0))
        elif msg.command == EndEffectorCommand.CMD_STOP:
            self.target = self.command_pct
        self.ack = (msg.command, msg.sender, msg.sequence)

    def ctrl(self, dt):
        """Finger actuator targets: the commanded opening moves toward the target at the set speed."""
        step = 100.0 * self.velocity / 100.0 / self.FULL_STROKE_S * dt
        self.command_pct += float(np.clip(self.target - self.command_pct, -step, step))
        r = self.CLOSED + (self.OPEN - self.CLOSED) * self.command_pct / 100.0
        return [r, -r]

    def position(self, d):
        """Measured opening, 0 (closed) .. 100 (open)."""
        return float(np.clip(100.0 * (self.CLOSED - d.qpos[self.joint]) / (self.CLOSED - self.OPEN), 0.0, 100.0))

    def finger_joints(self, d):
        """The real robot's l and r finger joint positions at the measured opening: l opens positive."""
        travel = self.URDF_STROKE * self.position(d) / 100.0
        return [travel, -travel]

    def publish(self, d, stamp):
        position = self.position(d)
        moving = abs(self.target - self.command_pct) > 0.5
        blocked = abs(position - self.command_pct) > self.dead_zone  # fingers held apart by an object
        command, sender, sequence = self.ack
        self.state_pub.publish(
            EndEffectorState(
                timestamp=stamp,
                id=self.ID,
                enabled=1,
                calibrated=int(self.calibrated),
                ready=1,
                moving=int(moving),
                gripping=int(blocked and not moving and self.target < position),
                missed=0,
                error=0,
                position=position,
                force=0.0,
                command=command,
                command_sender=sender,
                command_sequence=sequence,
            )
        )


class Arm:
    """One arm's emulated joint controller state."""

    def __init__(self, m, d, side):
        jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f'robot0_{side}_{j}') for j in JOINTS]
        self.qpos = m.jnt_qposadr[jid]
        self.dof = m.jnt_dofadr[jid]
        self.act = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f'robot0_torq_{side}_j{i}') for i in range(7)]
        self.tau_max = m.actuator_ctrlrange[self.act, 1]
        self.hand = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f'robot0_{side}_hand')
        self.names = [f'{side}_{j}' for j in JOINTS]
        self.mode = JointCommand.POSITION_MODE
        self.command = d.qpos[self.qpos].copy()  # position target / velocity / torque, by mode
        self.q_ref = self.command.copy()
        self.qd_ref = np.zeros(7)
        self.stamp = 0.0
        self.timeout = 0.2
        self.speed_ratio = 0.3
        self.tau = np.zeros(7)

    def hold(self, q):
        self.mode, self.command, self.q_ref, self.qd_ref = JointCommand.POSITION_MODE, q.copy(), q.copy(), np.zeros(7)

    def torque(self, q, qd, M, bias, now, dt, enabled):
        if not enabled:
            self.hold(q)
        elif self.mode != JointCommand.POSITION_MODE and now - self.stamp > self.timeout:
            self.hold(q)  # stale velocity/torque command: hold where the arm is
        if self.mode == JointCommand.TORQUE_MODE:
            tau = self.command + bias
        else:
            if self.mode == JointCommand.POSITION_MODE:  # move to the target at speed_ratio of max speed
                step = np.clip(
                    self.command - self.q_ref, -self.speed_ratio * MAX_SPEED * dt, self.speed_ratio * MAX_SPEED * dt
                )
                self.q_ref, self.qd_ref = self.q_ref + step, step / dt
            elif self.mode == JointCommand.VELOCITY_MODE:
                self.qd_ref = self.command
                self.q_ref = np.clip(self.q_ref + self.command * dt, q - 0.1, q + 0.1)  # no wind-up when blocked
            else:  # RAW_POSITION_MODE
                self.q_ref, self.qd_ref = self.command, np.zeros(7)
            tau = M @ (OMEGA**2 * (self.q_ref - q) + 2 * OMEGA * (self.qd_ref - qd)) + bias
        self.tau = np.clip(tau, -self.tau_max, self.tau_max)
        return self.tau


class SimBaxter:
    def __init__(self, node, render, empty=False, table=True, scene='lift', cameras=False):
        self.node, self.render, self.scene, self.renderers = node, render, scene, {}
        self.env = make_env(render=render)
        self.env.reset()
        if scene == 'cup':
            self.env = scenes.build_cup_scene(self.env)
        self.m, self.d = self.env.sim.model._model, self.env.sim.data._data
        if empty:  # move free objects (the pot) out of reach; the table stays below the hands
            for j in np.flatnonzero(self.m.jnt_type == mujoco.mjtJoint.mjJNT_FREE):
                self.d.qpos[self.m.jnt_qposadr[j] : self.m.jnt_qposadr[j] + 3] = [0.0, -3.0, 0.2]
            mujoco.mj_forward(self.m, self.d)
        if not table:
            remove_table(self.m)
        torso_ignores_arms(self.m)
        self.base = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, 'robot0_base')
        self.arms = {side: Arm(self.m, self.d, side) for side in ARMS}
        self.grippers = {side: Gripper(node, self.m, side) for side in ARMS}
        self.enabled = True
        self.head_pan, self.head_target, self.head_speed = 0.0, 0.0, 0.0
        self.lock = threading.Lock()
        self.M = np.zeros((self.m.nv, self.m.nv))

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        command_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.joint_pub = node.create_publisher(JointState, '/robot/joint_states', 10)
        self.state_pub = node.create_publisher(AssemblyState, '/robot/state', 10)
        self.head_pub = node.create_publisher(HeadState, '/robot/head/head_state', 10)
        node.create_subscription(HeadPanCommand, '/robot/head/command_head_pan', self.on_head, 10)
        self.endpoint_pubs = {}
        for side, arm in self.arms.items():
            ns = f'/robot/limb/{side}/'
            self.endpoint_pubs[side] = node.create_publisher(EndpointState, ns + 'endpoint_state', 10)
            node.create_subscription(
                JointCommand, ns + 'joint_command', lambda msg, a=arm: self.on_command(a, msg), command_qos
            )
            node.create_subscription(
                Float64, ns + 'joint_command_timeout', lambda msg, a=arm: setattr(a, 'timeout', msg.data), latched
            )
            node.create_subscription(
                Float64,
                ns + 'set_speed_ratio',
                lambda msg, a=arm: setattr(a, 'speed_ratio', float(np.clip(msg.data, 0, 1))),
                latched,
            )
        node.create_subscription(Bool, '/robot/set_super_enable', lambda msg: setattr(self, 'enabled', msg.data), 10)
        self.config_pub = node.create_publisher(String, '/sim/config', latched)  # the scene, for recordings
        self.scene_pub = node.create_publisher(MarkerArray, '/sim/scene', 10)
        robot = ('robot0', 'gripper0', 'mount0', 'fixed_mount')

        def body(g):
            return mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, self.m.geom_bodyid[g]) or ''

        self.scene_geoms = [
            g
            for g in range(self.m.ngeom)  # collision shapes the arms can hit
            if self.m.geom_contype[g] and int(self.m.geom_type[g]) in SCENE_SHAPES and not body(g).startswith(robot)
        ]
        scene_name = 'BlueCupRedContainer' if scene == 'cup' else 'TwoArmLift'
        self.config_pub.publish(String(data=json.dumps({'scene': scene_name, 'empty': empty, 'table': table})))
        self.pending_layout = self.pending_lighting = None
        self.cameras = self.make_cameras(node, scene) if cameras else []
        if scene == 'cup':
            self.objects_pub = node.create_publisher(PoseArray, '/sim/objects', 10)
            node.create_subscription(Float64MultiArray, '/sim/layout', self.on_layout, 10)
            node.create_subscription(
                Float64MultiArray, '/sim/lighting', lambda msg: setattr(self, 'pending_lighting', list(msg.data)), 10
            )

    def on_command(self, arm, msg):
        index = {name: i for i, name in enumerate(arm.names)}
        with self.lock:
            if msg.mode != arm.mode and msg.mode != JointCommand.TORQUE_MODE:
                arm.q_ref = self.d.qpos[arm.qpos].copy()  # switching modes: start from where the arm is
            command = arm.command.copy() if msg.mode == arm.mode else np.zeros(7)
            if msg.mode in (JointCommand.POSITION_MODE, JointCommand.RAW_POSITION_MODE) and msg.mode != arm.mode:
                command = self.d.qpos[arm.qpos].copy()  # joints not named in the message hold
            for name, value in zip(msg.names, msg.command):
                if name in index:
                    command[index[name]] = value
            arm.mode, arm.command, arm.stamp = msg.mode, command, time.monotonic()

    def on_layout(self, msg):
        # Applied by the physics loop: only it touches MjData (rendering does not take the lock).
        self.pending_layout = list(msg.data[:5])

    def apply_pending(self):
        layout, self.pending_layout = self.pending_layout, None
        if layout is not None:
            cup_x, cup_y, box_x, box_y, box_yaw = layout
            scenes.place(self.env, (cup_x, cup_y), (box_x, box_y), container_yaw=box_yaw)
        lighting, self.pending_lighting = self.pending_lighting, None
        if lighting is not None:
            scenes.apply_lighting(self.env, lighting)

    def make_cameras(self, node, scene):
        """The robot's hand cameras and, in the cup scene, the lab's third_eye: (MuJoCo camera, publisher,
        height, width, encoding, geoms hidden from it)."""
        m, cameras = self.m, []
        for side in ARMS:
            cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, f'robot0_{side}_eye_in_hand')
            m.cam_pos[cam] = HAND_CAMERA_POS
            m.cam_quat[cam] = Rotation.from_matrix(HAND_CAMERA_AXES).as_quat(scalar_first=True)
            m.cam_fovy[cam] = HAND_CAMERA_FOVY
            # robosuite's gripper housing is wider than the real one, which the camera sees past
            housing = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f'gripper0_{side}_gripper_base')
            w, h = HAND_CAMERA_SIZE
            pub = node.create_publisher(Image, f'/cameras/{side}_hand_camera/image', 5)
            cameras.append((f'robot0_{side}_eye_in_hand', pub, h, w, 'bgra8', np.flatnonzero(m.geom_bodyid == housing)))
        if scene == 'cup':
            eye = scenes.LAB['third_eye']
            pub = node.create_publisher(Image, eye['topic'], 5)
            cameras.append(('third_eye', pub, eye['size'][1], eye['size'][0], eye['encoding'], []))
        return cameras

    def publish_cameras(self):
        """Render and publish the cameras someone subscribes to (renderers made on first use, here)."""
        stamp = self.node.get_clock().now().to_msg()
        visual = mujoco.MjvOption()
        visual.geomgroup[0] = 0  # robosuite's collision geoms, which its viewer hides too
        for cam, pub, h, w, encoding, hidden in self.cameras:
            if not pub.get_subscription_count():
                continue
            if cam not in self.renderers:
                self.renderers[cam] = mujoco.Renderer(self.m, h, w)
            r = self.renderers[cam]
            alpha = self.m.geom_rgba[hidden, 3].copy()
            self.m.geom_rgba[hidden, 3] = 0.0
            r.update_scene(self.d, camera=cam, scene_option=visual)
            self.m.geom_rgba[hidden, 3] = alpha
            rgb = r.render()
            data = np.dstack([rgb[..., ::-1], np.full((h, w), 255, np.uint8)]) if encoding == 'bgra8' else rgb
            msg = Image(height=h, width=w, encoding=encoding, step=w * data.shape[2], data=data.tobytes())
            msg.header.stamp, msg.header.frame_id = stamp, cam
            pub.publish(msg)

    def publish_objects(self):
        msg = PoseArray()
        msg.header.stamp, msg.header.frame_id = self.node.get_clock().now().to_msg(), 'base'
        for name in ('cup', 'container'):
            b = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, name)
            pb, Rb = scenes.base_pose(self.env)
            p = Pose()
            p.position.x, p.position.y, p.position.z = Rb.T @ (self.d.xpos[b] - pb)
            q = Rotation.from_matrix(Rb.T @ self.d.xmat[b].reshape(3, 3)).as_quat()
            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = q
            msg.poses.append(p)
        self.objects_pub.publish(msg)

    def on_head(self, msg):
        self.head_target = float(msg.target)
        self.head_speed = HEAD_MAX_SPEED * float(np.clip(msg.speed_ratio, 0.0, 1.0))

    def step(self):
        m, d = self.m, self.d
        dt = m.opt.timestep
        with self.lock:
            mujoco.mj_fullM(m, self.M, d.qM)
            now = time.monotonic()
            for arm in self.arms.values():
                q, qd = d.qpos[arm.qpos], d.qvel[arm.dof]
                M = self.M[np.ix_(arm.dof, arm.dof)]
                d.ctrl[arm.act] = arm.torque(q, qd, M, d.qfrc_bias[arm.dof], now, dt, self.enabled)
            for gripper in self.grippers.values():
                d.ctrl[gripper.acts] = gripper.ctrl(dt)
        mujoco.mj_step(m, d)

    def publish(self):
        m, d = self.m, self.d
        stamp = self.node.get_clock().now().to_msg()
        step = self.head_speed * DT_PUBLISH
        self.head_pan += float(np.clip(self.head_target - self.head_pan, -step, step))
        fingers = [q for gripper in self.grippers.values() for q in gripper.finger_joints(d)]
        name, position = ['head_pan', *FINGER_JOINTS], [self.head_pan, *fingers]
        velocity, effort = [0.0] * 5, [0.0] * 5
        Rb, pb = d.xmat[self.base].reshape(3, 3), d.xpos[self.base]
        for side, arm in self.arms.items():
            name += arm.names
            position += d.qpos[arm.qpos].tolist()
            velocity += d.qvel[arm.dof].tolist()
            effort += arm.tau.tolist()

            R_hand = d.xmat[arm.hand].reshape(3, 3)
            tcp = d.xpos[arm.hand] + R_hand @ TCP_IN_HAND
            jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            mujoco.mj_jac(m, d, jacp, jacr, tcp, arm.hand)
            ep = EndpointState()
            ep.header.stamp, ep.header.frame_id = stamp, 'base'
            p, o = ep.pose.position, ep.pose.orientation
            p.x, p.y, p.z = Rb.T @ (tcp - pb)
            o.x, o.y, o.z, o.w = Rotation.from_matrix(Rb.T @ R_hand).as_quat()
            v, w = Rb.T @ (jacp @ d.qvel), Rb.T @ (jacr @ d.qvel)
            ep.twist.linear.x, ep.twist.linear.y, ep.twist.linear.z = v
            ep.twist.angular.x, ep.twist.angular.y, ep.twist.angular.z = w
            self.endpoint_pubs[side].publish(ep)
        js = JointState(name=name, position=position, velocity=velocity, effort=effort)
        js.header.stamp = stamp
        self.joint_pub.publish(js)
        self.state_pub.publish(AssemblyState(ready=self.enabled, enabled=self.enabled))
        turning = abs(self.head_target - self.head_pan) > 1e-3
        self.head_pub.publish(HeadState(pan=self.head_pan, is_turning=turning, is_pan_enabled=True))
        for gripper in self.grippers.values():
            gripper.publish(self.d, stamp)

    def publish_scene(self):
        m, d = self.m, self.d
        Rb, pb = d.xmat[self.base].reshape(3, 3), d.xpos[self.base]
        stamp = self.node.get_clock().now().to_msg()
        markers = MarkerArray()
        for g in self.scene_geoms:
            if np.linalg.norm(d.geom_xpos[g] - pb) > 2.0:  # moved out of the workspace (--empty)
                continue
            mk = Marker(type=SCENE_SHAPES[int(m.geom_type[g])], action=Marker.ADD, ns='sim_scene', id=int(g))
            mk.header.frame_id, mk.header.stamp = 'base', stamp
            p, o = mk.pose.position, mk.pose.orientation
            p.x, p.y, p.z = Rb.T @ (d.geom_xpos[g] - pb)
            o.x, o.y, o.z, o.w = Rotation.from_matrix(Rb.T @ d.geom_xmat[g].reshape(3, 3)).as_quat()
            size = m.geom_size[g]
            if mk.type == Marker.CUBE:
                mk.scale.x, mk.scale.y, mk.scale.z = 2 * size[:3]
            elif mk.type == Marker.CYLINDER:
                mk.scale.x = mk.scale.y = 2 * size[0]
                mk.scale.z = 2 * size[1]
            else:
                mk.scale.x = mk.scale.y = mk.scale.z = 2 * size[0]
            free = m.body_jntnum[m.geom_bodyid[g]] > 0  # bodies with a joint can move
            mk.color.r, mk.color.g, mk.color.b, mk.color.a = SCENE_COLORS['free' if free else 'static']
            markers.markers.append(mk)
        self.scene_pub.publish(markers)

    def run(self):
        """Step physics in real time, publishing every DT_PUBLISH and rendering at ~30 Hz."""
        substeps = max(1, round(DT_PUBLISH / self.m.opt.timestep))
        next_tick, frame = time.monotonic(), 0
        while rclpy.ok():
            with self.lock:
                self.apply_pending()
            for _ in range(substeps):
                self.step()
            self.publish()
            frame += 1
            if frame % 10 == 0:
                self.publish_scene()
                if self.scene == 'cup':
                    self.publish_objects()
            if self.cameras and frame % CAMERA_EVERY == 0:
                self.publish_cameras()
            if self.render and frame % 3 == 0:
                # robosuite's render() is a no-op for its mjviewer; update() opens the window, then syncs it
                self.env.viewer.update()
            next_tick += DT_PUBLISH
            time.sleep(max(0.0, next_tick - time.monotonic()))


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Simulated Baxter on the real robot ROS 2 interface')
    parser.add_argument('--no-viewer', action='store_true', help='run without the MuJoCo viewer')
    parser.add_argument('--empty', action='store_true', help='no objects in reach (for controller tests)')
    parser.add_argument('--no-table', action='store_true', help='no table: the arms can reach down freely')
    parser.add_argument(
        '--scene', choices=('lift', 'cup'), default='lift', help='cup: BlueCupRedContainer in the lab of lab.toml'
    )
    args, ros_args = parser.parse_known_args()

    # Cameras render offscreen through EGL, so only without the viewer: GLX offscreen contexts beside a
    # viewer segfault Xorg on the NVIDIA 535 driver.
    cameras = args.no_viewer and os.environ.get('MUJOCO_GL') == 'egl'
    if args.no_viewer and not cameras:
        print('No cameras: set MUJOCO_GL=egl to render them (launch_sim.sh does).', flush=True)

    rclpy.init(args=['baxter_sim', *ros_args])
    node = rclpy.create_node('baxter_sim')
    deadline = time.monotonic() + 2.0  # let discovery find a real robot, if one is up
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    if node.count_publishers('/robot/joint_states'):
        raise SystemExit('/robot/joint_states already has a publisher (a real robot or another sim); not starting.')

    sim = SimBaxter(
        node, render=not args.no_viewer, empty=args.empty, table=not args.no_table, scene=args.scene, cameras=cameras
    )

    def spin():
        try:
            rclpy.spin(node)
        except Exception:  # the context is shut down under it on exit
            pass

    threading.Thread(target=spin, daemon=True).start()
    print('Simulated Baxter up on the real robot topics. Ctrl+C to stop.')
    try:
        sim.run()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:  # Ctrl+C shut ROS down between the loop's check and a publish
        if rclpy.ok():
            raise


if __name__ == '__main__':
    main()

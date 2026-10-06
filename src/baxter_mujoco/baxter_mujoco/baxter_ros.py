"""The MuJoCo Baxter behind the real robot's ROS 2 interface (baxter_core_msgs).

Run `./launch_sim.sh --ros`. bringbackbaxter's baxter_interface (teleop, tuck_arms) and RViz then
drive and show this simulated Baxter exactly as they do the real one, over the same topics:

  publishes   /robot/joint_states, /robot/limb/{arm}/endpoint_state, /robot/state,
              /robot/head/head_state, /sim/config (scene, latched), /sim/scene (RViz markers of
              everything else the arms can hit: the table, objects)
  subscribes  /robot/limb/{arm}/joint_command (position, velocity, torque, raw position modes),
              /robot/limb/{arm}/joint_command_timeout, /robot/limb/{arm}/set_speed_ratio,
              /robot/set_super_enable, /robot/head/command_head_pan

Baxter's own joint controllers are emulated with computed torque plus gravity compensation
(critically damped at OMEGA, about 0.1 s of lag like the real arm) inside the actuators' torque
limits. The head (not in the MuJoCo model) pans kinematically. As on the robot, position targets persist, while velocity and torque commands older than
the command timeout drop back to holding the current position. The endpoint is the URDF
{arm}_gripper frame (hand + 2.5 cm), in the robot base frame.
"""

import json
import threading
import time

from baxter_core_msgs.msg import AssemblyState, EndpointState, HeadPanCommand, HeadState, JointCommand
import mujoco
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Bool, Float64, String

from baxflow.sim import make_env

ARMS = ('left', 'right')
JOINTS = ('s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2')
MAX_SPEED = np.array([1.5, 1.5, 1.5, 1.5, 4.0, 4.0, 4.0])  # rad/s, URDF velocity limits
HEAD_MAX_SPEED = 1.0  # rad/s
OMEGA = 20.0  # rad/s, joint servo bandwidth (critically damped)
TCP_IN_HAND = np.array([0.0, 0.0, 0.025])  # URDF {arm}_hand -> {arm}_gripper
SCENE_SHAPES = {int(mujoco.mjtGeom.mjGEOM_BOX): Marker.CUBE, int(mujoco.mjtGeom.mjGEOM_CYLINDER): Marker.CYLINDER,
                int(mujoco.mjtGeom.mjGEOM_SPHERE): Marker.SPHERE}
SCENE_COLORS = {'static': (0.62, 0.60, 0.55, 0.85), 'free': (0.80, 0.42, 0.22, 0.9)}  # table / movable objects
DT_PUBLISH = 0.01  # s, state publishing period (100 Hz)
FINGER_JOINTS = ('l_gripper_l_finger_joint', 'l_gripper_r_finger_joint', 'r_gripper_l_finger_joint',
                 'r_gripper_r_finger_joint')


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
                step = np.clip(self.command - self.q_ref, -self.speed_ratio * MAX_SPEED * dt, self.speed_ratio * MAX_SPEED * dt)
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
    def __init__(self, node, render, empty=False):
        self.node, self.render = node, render
        self.env = make_env(render=render)
        self.env.reset()
        self.m, self.d = self.env.sim.model._model, self.env.sim.data._data
        if empty:  # move free objects (the pot) out of reach; the table stays below the hands
            for j in np.flatnonzero(self.m.jnt_type == mujoco.mjtJoint.mjJNT_FREE):
                self.d.qpos[self.m.jnt_qposadr[j]:self.m.jnt_qposadr[j] + 3] = [0.0, -3.0, 0.2]
            mujoco.mj_forward(self.m, self.d)
        self.base = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, 'robot0_base')
        self.arms = {side: Arm(self.m, self.d, side) for side in ARMS}
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
            node.create_subscription(JointCommand, ns + 'joint_command', lambda msg, a=arm: self.on_command(a, msg), command_qos)
            node.create_subscription(Float64, ns + 'joint_command_timeout', lambda msg, a=arm: setattr(a, 'timeout', msg.data), latched)
            node.create_subscription(Float64, ns + 'set_speed_ratio', lambda msg, a=arm: setattr(a, 'speed_ratio', float(np.clip(msg.data, 0, 1))), latched)
        node.create_subscription(Bool, '/robot/set_super_enable', lambda msg: setattr(self, 'enabled', msg.data), 10)
        self.config_pub = node.create_publisher(String, '/sim/config', latched)  # the scene, for recordings
        self.scene_pub = node.create_publisher(MarkerArray, '/sim/scene', 10)
        robot = ('robot0', 'gripper0', 'mount0', 'fixed_mount')
        body = lambda g: mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, self.m.geom_bodyid[g]) or ''
        self.scene_geoms = [g for g in range(self.m.ngeom)  # collision shapes the arms can hit
                            if self.m.geom_contype[g] and int(self.m.geom_type[g]) in SCENE_SHAPES
                            and not body(g).startswith(robot)]
        self.config_pub.publish(String(data=json.dumps({'scene': 'TwoArmLift', 'empty': empty})))

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
        mujoco.mj_step(m, d)

    def publish(self):
        m, d = self.m, self.d
        stamp = self.node.get_clock().now().to_msg()
        step = self.head_speed * DT_PUBLISH
        self.head_pan += float(np.clip(self.head_target - self.head_pan, -step, step))
        name, position = ['head_pan', *FINGER_JOINTS], [self.head_pan] + [0.0] * 4
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
            for _ in range(substeps):
                self.step()
            self.publish()
            frame += 1
            if frame % 10 == 0:
                self.publish_scene()
            if self.render and frame % 3 == 0:
                self.env.render()
            next_tick += DT_PUBLISH
            time.sleep(max(0.0, next_tick - time.monotonic()))


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Simulated Baxter on the real robot ROS 2 interface')
    parser.add_argument('--no-viewer', action='store_true', help='run without the MuJoCo viewer')
    parser.add_argument('--empty', action='store_true', help='no objects in reach (for controller tests)')
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=['baxter_sim', *ros_args])
    node = rclpy.create_node('baxter_sim')
    deadline = time.monotonic() + 2.0  # let discovery find a real robot, if one is up
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    if node.count_publishers('/robot/joint_states'):
        raise SystemExit('/robot/joint_states already has a publisher (a real robot or another sim); not starting.')

    sim = SimBaxter(node, render=not args.no_viewer, empty=args.empty)
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


if __name__ == '__main__':
    main()

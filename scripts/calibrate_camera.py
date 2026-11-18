#!/usr/bin/env python3
"""Markerless calibration of a camera fixed relative to Baxter's base (the third-eye camera), from recorded bags.

EasyHeC (Chen et al., RA-L 2023; simple-easyhec, Tao 2025) renders the robot's meshes at its recorded joint angles and
moves the camera pose until the rendered silhouette matches the robot's segmentation mask in each image. This script
only feeds it Baxter's data; the optimizer is easyhec.optim.optimize, unchanged. Each stage runs in the environment
that has its dependencies, and calibrate_camera.sh chains them:

  export OUT BAG..  (ROS Python)       frames where the arms are still and far apart in joint space, their joint
                                       angles, and the intrinsics from camera_info -> OUT/frames.npz
  masks OUT         (rethought_ai)     SAM3 masks of the arms, and of people to drop occluded frames -> OUT/masks.npz,
                                       OUT/masks.jpg (check it: a bad mask is a bad calibration)
  solve OUT         (simple_easyhec)   URDF forward kinematics, EasyHeC from a rough guess -> OUT/extrinsic.yaml
  verify OUT HELD   (simple_easyhec)   the solved pose on another bag's frames: mask IoU per frame -> HELD/verify.*

Frames: the camera pose in Baxter's base frame. The result is given in two conventions: the ROS camera frame (x
forward, y left, z up, as RealSense's *_color_frame) and the optical frame (z forward, x right, y down, as
*_color_optical_frame). Lens distortion is ignored; RealSense color streams report near-zero distortion.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

IMAGE = '/cameras/third_eye/color/image_raw/compressed'
INFO = '/cameras/third_eye/color/camera_info'
JOINTS = '/robot/joint_states'
ARM_JOINTS = [f'{side}_{j}' for side in ('left', 'right') for j in ('s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2')]
ROBOT_PROMPTS = ('robot arm', 'robot')
SCALE = 0.5  # solve at half resolution: 4x less GPU memory, still about 2 mm per pixel at 1 m
OUTLIER = 0.8  # a frame whose silhouette IoU is below this fraction of the median is left out of the second solve
NEAR = 15  # px: how close to the robot mask a person counts as occluding it
STILL = 0.05  # rad/s: max arm joint speed for a frame to be used, so image and joint angles agree despite sync jitter
SYNC = 0.02  # s: max gap between an image and its nearest joint state
# The arms and grippers, not the torso or pedestal: the table hides those, and a rendered part the mask cannot see
# would pull the camera pose toward the occluder.
LINKS = ('left_', 'right_', 'l_gripper', 'r_gripper')
# Rough guess of the camera in base (EasyHeC tolerates about 0.1 m and 15 degrees for a camera this far from the robot,
# per the simple-easyhec README): x y z (m), then pitch (deg the view tips below horizontal, looking along +x) and roll
# (deg about the view axis). The third-eye camera sits on a pole above the head looking steeply down at the table, and
# is turned so the arms enter from the image's left: image right is +x (forward), image up is +y (Baxter's left).
GUESS = (0.1, 0.0, 1.1, 70.0, -90.0)
# The sampled search box around it: x y z (m), pitch, roll (deg), plus yaw about base z within +-SEARCH_YAW.
SEARCH_LO, SEARCH_HI, SEARCH_YAW, SEARCH_SAMPLES = (
    (-0.3, -0.3, 0.5, 30.0, -130.0),
    (0.8, 0.3, 1.6, 90.0, -50.0),
    30.0,
    3000,
)


# --------------------------------------------------------------------------- export (ROS Python)
def export(bags, out, count):
    """Pool the still, synced frames of every bag (a folder of episode_* bags counts as all of them), then pick the
    `count` most different arm poses among them."""
    import cv2

    bags = [e for b in bags for e in (sorted(b.glob('episode_*/')) or [b])]
    pool = [still_frames(b) for b in bags]
    pool = [p for p in pool if p is not None]
    if not pool:
        sys.exit('no bag has the third-eye camera, its camera_info and /robot/joint_states')
    names, K, size = pool[0][1:4]
    jpegs = [j for p in pool for j in p[4]]
    q_all = np.concatenate([p[5] for p in pool])
    stamps = np.concatenate([p[6] for p in pool])
    source = [str(p[0]) for p in pool for _ in p[4]]
    if len(jpegs) < count:
        sys.exit(f'only {len(jpegs)} still, synced frames in {len(bags)} bags; record the arms holding several poses')

    # Farthest-point sampling in arm joint space: poses as different as possible constrain the camera best.
    q = q_all[:, [names.index(j) for j in ARM_JOINTS]]
    chosen = [int(np.argmax(np.linalg.norm(q - q.mean(0), axis=1)))]
    for _ in range(count - 1):
        chosen.append(int(np.argmax(np.min(np.linalg.norm(q[:, None] - q[chosen][None], axis=2), axis=1))))
    images = np.stack([cv2.imdecode(np.frombuffer(jpegs[i], np.uint8), cv2.IMREAD_COLOR)[..., ::-1] for i in chosen])
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / 'frames.npz', images=images, joint_names=np.array(names), joint_positions=q_all[chosen],
        intrinsic=K, size=np.array(size), stamps=stamps[chosen], bag=np.array([source[i] for i in chosen]),
    )  # fmt: skip
    print(f'{count} frames of {len(jpegs)} usable in {len(pool)} bags -> {out / "frames.npz"}')


def still_frames(bag):
    """(bag, joint names, intrinsics, image size, JPEGs, joint angles, stamps) of the frames whose nearest joint state is
    within SYNC and whose arms move slower than STILL; None if the bag lacks the topics."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id='mcap'), rosbag2_py.ConverterOptions('cdr', 'cdr'))
    reader.set_filter(rosbag2_py.StorageFilter(topics=[IMAGE, INFO, JOINTS]))
    types = {t.name: get_message(t.type) for t in reader.get_all_topics_and_types()}
    jpegs, image_t, joint_t, joint_q, names, K, size = [], [], [], [], None, None, None
    while reader.has_next():
        topic, data, t = reader.read_next()
        msg = deserialize_message(data, types[topic])
        if topic == IMAGE:
            jpegs.append(bytes(msg.data))
            image_t.append(t * 1e-9)
        elif topic == INFO and K is None:
            K, size = np.array(msg.k, float).reshape(3, 3), (msg.width, msg.height)
        elif topic == JOINTS and set(ARM_JOINTS) <= set(msg.name):
            names = names or list(msg.name)
            if list(msg.name) == names:  # Baxter also publishes partial joint states (grippers); keep the full ones
                joint_t.append(t * 1e-9)
                joint_q.append(np.array(msg.position, float))
    if K is None or not jpegs or len(joint_q) < 2:
        return None
    joint_t, joint_q, image_t = np.array(joint_t), np.array(joint_q), np.array(image_t)
    speed = np.abs(np.gradient(joint_q[:, [names.index(j) for j in ARM_JOINTS]], joint_t, axis=0)).max(1)
    nearest = np.clip(np.searchsorted(joint_t, image_t), 1, len(joint_t) - 1)
    nearest -= image_t - joint_t[nearest - 1] < joint_t[nearest] - image_t
    usable = np.flatnonzero((np.abs(joint_t[nearest] - image_t) < SYNC) & (speed[nearest] < STILL))
    return bag, names, K, size, [jpegs[i] for i in usable], joint_q[nearest[usable]], image_t[usable]


# --------------------------------------------------------------------------- masks (rethought_ai: SAM3)
def masks(out, rai_dir, threshold, max_person, device):
    import cv2
    import torch

    sys.path.insert(0, str(Path(rai_dir) / 'tools/masking'))
    from sam3_inpaint_dataset import load_sam3, person_union_masks

    images = np.load(out / 'frames.npz')['images']
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    processor, model = load_sam3('cpu')
    if device == 'cuda':  # bf16 weights: about 1.7 GB instead of 3.4, so SAM3 fits beside other jobs on a 4 GB GPU
        model = model.to(device, dtype=torch.bfloat16)
    robot, person = [], []

    def mask(rgb, prompt):  # one frame at a time: the laptop GPU has 4 GB. Returns [(union mask, instance count)].
        return person_union_masks(processor, model, [rgb], prompt, threshold, device, device == 'cuda')[0][0]

    for rgb in images:
        # SAM3 sometimes misses a whole arm for one prompt; the union of two covers it.
        robot.append(np.logical_or.reduce([mask(rgb, p) for p in ROBOT_PROMPTS]))
        person.append(mask(rgb, 'person'))
    robot, person = np.stack(robot).astype(bool), np.stack(person).astype(bool)
    # A person only matters where they hide the robot: a hand on the arm, not legs at the image edge. Drop a frame when
    # the person mask reaches into the robot mask grown by NEAR px.
    near = np.stack([cv2.dilate(r.astype(np.uint8), np.ones((2 * NEAR + 1,) * 2, np.uint8)) > 0 for r in robot])
    keep = (person & near).mean((1, 2)) <= max_person
    np.savez_compressed(out / 'masks.npz', masks=robot, keep=keep)

    tiles = []
    for rgb, m, k in zip(images, robot, keep):
        tile = rgb.copy()
        tile[m] = (0.5 * tile[m] + [0, 127, 0]).astype(np.uint8)
        if not k:
            tile[::8] = 0  # struck through: a person occludes the robot, the frame is dropped
        tiles.append(cv2.resize(tile, (320, 240)))
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 4)
    sheet = np.vstack([np.hstack(tiles[i : i + 4]) for i in range(0, len(tiles), 4)])
    cv2.imwrite(str(out / 'masks.jpg'), sheet[..., ::-1])
    print(f'{keep.sum()} of {len(keep)} frames kept -> {out / "masks.npz"}; check {out / "masks.jpg"}')


# --------------------------------------------------------------------------- solve and verify (simple_easyhec)
def robot_model(ws_dir, out):
    """Baxter's URDF with package:// paths resolved, and the arm and gripper links with their visual meshes placed by
    each visual's origin (simple-easyhec's example ignores it; Baxter's visuals are offset from their links)."""
    from easyhec.utils.utils_3d import merge_meshes
    from urchin import URDF

    text = (ws_dir / 'src/baxter_description/urdf/baxter.urdf').read_text()
    for pkg in ('baxter_description', 'rethink_ee_description'):
        text = text.replace(f'package://{pkg}/', f'{ws_dir / "src" / pkg}/')
    resolved = out / 'baxter_resolved.urdf'
    resolved.write_text(text)
    urdf = URDF.load(str(resolved))
    links, meshes = [], []
    for link in urdf.links:
        if not link.name.startswith(LINKS):
            continue
        parts = []
        for visual in link.visuals:
            if visual.geometry.mesh is None:
                continue
            for m in visual.geometry.mesh.meshes:
                m = m.copy()
                if visual.geometry.mesh.scale is not None:
                    m.apply_scale(visual.geometry.mesh.scale)
                parts.append(m.apply_transform(visual.origin))
        mesh = merge_meshes(parts)
        if mesh is not None:
            links.append(link.name)
            meshes.append(mesh)
    return urdf, links, meshes


def link_poses(urdf, links, frames):
    names, positions = list(frames['joint_names']), frames['joint_positions']
    actuated = {j.name for j in urdf.actuated_joints}
    poses = np.zeros((len(positions), len(links), 4, 4))
    for i, q in enumerate(positions):
        fk = urdf.link_fk(cfg={n: v for n, v in zip(names, q) if n in actuated}, use_names=True)
        poses[i] = [fk[name] for name in links]
    return poses


def guess(values):
    """The rough guess as an OpenCV extrinsic: roll about the view axis first, then pitch (+ about y tips the view, x,
    downward)."""
    from easyhec.utils.camera_conversions import ros2opencv
    from transforms3d.euler import euler2mat

    x, y, z, pitch, roll = values
    pose = np.eye(4)
    pose[:3, :3] = euler2mat(np.radians(roll), np.radians(pitch), 0)
    pose[:3, 3] = x, y, z
    return ros2opencv(pose)


def solve(out, ws_dir, iterations, initial, search):
    """EasyHeC on every kept frame, then once more without the frames its result disagrees with: a mask that misses an
    arm (SAM3 sometimes does) or covers a held tool pulls the pose, and shows as a silhouette IoU far below the rest."""
    import torch
    from easyhec.optim.optimize import optimize
    from easyhec.utils.camera_conversions import opencv2ros

    urdf, links, meshes = robot_model(ws_dir, out)
    poses, images, masks_, K, bags = load(out, urdf, links)
    h, w = images.shape[1:3]
    print(f'{len(images)} frames at {w}x{h}, {len(links)} links')

    def fit(start, use):
        cuda = torch.device('cuda')
        return (
            optimize(
                initial_extrinsic_guess=torch.tensor(start).float().to(cuda),
                camera_intrinsic=torch.from_numpy(K).float().to(cuda),
                masks=torch.from_numpy(masks_[use]).float().to(cuda),
                link_poses_dataset=torch.from_numpy(poses[use]).float().to(cuda),
                meshes=meshes,
                camera_width=w,
                camera_height=h,
                iterations=iterations,
                # All frames each step: simple-easyhec's batch_size path also indexes the shared intrinsic by frame.
                verbose=False,
            )
            .cpu()
            .numpy()
        )

    use = np.ones(len(images), bool)
    extrinsic = fit(coarse(meshes, poses, masks_, K, SEARCH_SAMPLES) if search else guess(initial), use)
    ious = iou(silhouettes(meshes, poses, K, extrinsic, h, w), masks_)
    use = ious >= OUTLIER * np.median(ious)
    if not use.all():
        print(f'dropping {np.flatnonzero(~use).tolist()} (IoU {np.round(ious[~use], 2).tolist()}), solving again')
        extrinsic = fit(extrinsic, use)
    rendered = silhouettes(meshes, poses, K, extrinsic, h, w)
    ious = iou(rendered, masks_)
    np.save(out / 'extrinsic_opencv.npy', extrinsic)
    write_yaml(out / 'extrinsic.yaml', opencv2ros(extrinsic), f'{len(set(bags))} bags')
    report(out / 'solve', images, rendered, masks_, ious, use)


def verify(out, held, ws_dir):
    """Render the robot with the solved pose on frames the solve never saw, and score the silhouette against their
    masks. IoU near the masks' own quality (about 0.9 for clean SAM3 masks) means the pose transfers."""
    extrinsic = np.load(out / 'extrinsic_opencv.npy')
    urdf, links, meshes = robot_model(ws_dir, held)
    poses, images, masks_, K, _ = load(held, urdf, links)
    rendered = silhouettes(meshes, poses, K, extrinsic, *images.shape[1:3])
    report(held / 'verify', images, rendered, masks_, iou(rendered, masks_), np.ones(len(images), bool))


def load(folder, urdf, links):
    """The kept frames at SCALE: link poses, images, masks, intrinsics (scaled to match) and source bags."""
    import cv2

    frames, m = np.load(folder / 'frames.npz'), np.load(folder / 'masks.npz')
    keep = m['keep']
    h, w = (np.array(frames['images'].shape[1:3]) * SCALE).astype(int)
    images = np.stack([cv2.resize(x, (w, h), interpolation=cv2.INTER_AREA) for x in frames['images'][keep]])
    masks_ = (
        np.stack([cv2.resize(x.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) for x in m['masks'][keep]])
        > 0
    )
    K = frames['intrinsic'].copy()
    K[:2] *= SCALE
    return link_poses(urdf, links, frames)[keep], images, masks_, K, frames['bag'][keep].tolist()


def robot_in_base(meshes, poses):
    """Per frame, all links merged into one mesh placed in the base frame (GPU tensors), so rendering a frame at any
    camera pose is a single rasterization."""
    import torch

    faces, n = [], 0
    for x in meshes:
        faces.append(torch.tensor(x.faces).int() + n)
        n += len(x.vertices)
    faces = torch.cat(faces).cuda()
    verts = [torch.tensor(x.vertices).float().cuda() for x in meshes]
    out = []
    for pose in poses:
        P = torch.tensor(pose).float().cuda()
        out.append(torch.cat([v @ T[:3, :3].T + T[:3, 3] for v, T in zip(verts, P)]))
    return out, faces


def silhouettes(meshes, poses, K, extrinsic, h, w):
    """The robot rendered at `extrinsic` (OpenCV, base to camera) for each frame's link poses: (N, H, W) bool."""
    import torch
    from easyhec.optim.nvdiffrast_renderer import NVDiffrastRenderer

    renderer = NVDiffrastRenderer(h, w)
    verts, faces = robot_in_base(meshes, poses)
    K, E = torch.tensor(K).float().cuda(), torch.tensor(extrinsic).float().cuda()
    return np.stack([renderer.render_mask(v, faces, K, E).cpu().numpy() > 0.5 for v in verts])


def coarse(meshes, poses, masks_, K, samples, seed=0):
    """A start for EasyHeC by sampling (as EasyHeC++ does): random camera poses in a wide box around the rough guess,
    each scored by the mean silhouette IoU on a few frames at 1/4 resolution. EasyHeC only refines: from a pose where
    the rendered robot misses the masks there is no gradient to follow."""
    import cv2
    import torch
    from easyhec.optim.nvdiffrast_renderer import NVDiffrastRenderer

    rng = np.random.default_rng(seed)
    sub = rng.choice(len(poses), min(6, len(poses)), replace=False)
    h, w = (np.array(masks_.shape[1:]) // 4).astype(int)
    small = (
        torch.tensor(
            np.stack([cv2.resize(x.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) for x in masks_[sub]])
        )
        .bool()
        .cuda()
    )
    Ks = torch.tensor(K).float().cuda() / 4
    Ks[2, 2] = 1
    renderer = NVDiffrastRenderer(h, w)
    verts, faces = robot_in_base(meshes, poses[sub])
    lo, hi = np.array(SEARCH_LO), np.array(SEARCH_HI)
    best, best_score = None, -1.0
    for _ in range(samples):
        g = lo + rng.random(5) * (hi - lo)
        yaw = rng.uniform(-SEARCH_YAW, SEARCH_YAW)
        E = guess(g)
        R = np.eye(
            4
        )  # yaw about base z, applied to the camera pose: pre-multiply the base-to-camera transform's inverse
        c, s_ = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
        R[:2, :2] = [[c, -s_], [s_, c]]
        E = E @ np.linalg.inv(R)
        Et = torch.tensor(E).float().cuda()
        score = 0.0
        for v, m in zip(verts, small):
            r = renderer.render_mask(v, faces, Ks, Et) > 0.5
            score += ((r & m).sum() / max((r | m).sum(), 1)).item()
        if score > best_score:
            best, best_score = E, score
    print(f'coarse search: best mean IoU {best_score / len(sub):.3f} over {samples} samples')
    return best


def iou(rendered, masks_):
    return (rendered & masks_).sum((1, 2)) / np.maximum((rendered | masks_).sum((1, 2)), 1)


def report(prefix, images, rendered, masks_, ious, used):
    """prefix.jpg: red the rendered robot, green the mask's outline, struck through the frames left out; prefix.txt."""
    import cv2

    tiles = []
    for rgb, r, mask, u in zip(images, rendered, masks_, used):
        tile = rgb.copy()
        tile[r] = (0.5 * tile[r] + [127, 0, 0]).astype(np.uint8)
        tile[cv2.Canny(mask.astype(np.uint8) * 255, 0, 1) > 0] = (0, 255, 0)
        if not u:
            tile[::8] = 0
        tiles.append(cv2.resize(tile, (320, 240)))
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 4)
    cv2.imwrite(f'{prefix}.jpg', np.vstack([np.hstack(tiles[i : i + 4]) for i in range(0, len(tiles), 4)])[..., ::-1])
    text = (
        f'frames {len(ious)} used {int(used.sum())}\nIoU per frame {np.round(ious, 3).tolist()}\n'
        f'IoU median {np.median(ious[used]):.3f} min {ious[used].min():.3f} (frames used)\n'
    )
    Path(f'{prefix}.txt').write_text(text)
    print(text + f'overlays: {prefix}.jpg')


def write_yaml(path, pose_ros, source):
    """Base -> camera, as a static transform in both camera conventions."""
    from transforms3d.quaternions import mat2quat

    optical = pose_ros.copy()
    optical[:3, :3] = pose_ros[:3, :3] @ np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])  # x fwd -> z fwd, x right
    lines = [f'# EasyHeC calibration of the third-eye camera from {source}', 'parent: base']
    for name, T in (('third_eye_color_frame', pose_ros), ('third_eye_color_optical_frame', optical)):
        w, x, y, z = mat2quat(T[:3, :3])
        lines += [
            f'{name}:',
            f'  translation: [{T[0, 3]:.5f}, {T[1, 3]:.5f}, {T[2, 3]:.5f}]  # m',
            f'  rotation_xyzw: [{x:.6f}, {y:.6f}, {z:.6f}, {w:.6f}]',
        ]
    path.write_text('\n'.join(lines) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='stage', required=True)
    e = sub.add_parser('export')
    e.add_argument('out', type=Path)
    e.add_argument('bags', type=Path, nargs='+', help='bags, or folders of episode_* bags')
    e.add_argument('--frames', type=int, default=12)
    k = sub.add_parser('masks')
    k.add_argument('out', type=Path)
    k.add_argument('--rai-dir', type=Path, required=True)
    k.add_argument('--threshold', type=float, default=0.5)
    k.add_argument(
        '--max-person',
        type=float,
        default=0.002,
        help='drop frames where a person covers more of the image next to the robot',
    )
    k.add_argument('--device', help='cuda (default when available, bf16, about 1.7 GB) or cpu')
    s = sub.add_parser('solve')
    s.add_argument('out', type=Path)
    s.add_argument('--iterations', type=int, default=5000)
    s.add_argument('--guess', type=float, nargs=5, default=GUESS, metavar=('X', 'Y', 'Z', 'PITCH', 'ROLL'))
    s.add_argument('--no-search', action='store_true', help='start EasyHeC from --guess instead of the sampled search')
    v = sub.add_parser('verify')
    v.add_argument('out', type=Path)
    v.add_argument('held', type=Path)
    a = p.parse_args()
    ws_dir = Path(__file__).resolve().parents[1]
    if a.stage == 'export':
        export(a.bags, a.out, a.frames)
    elif a.stage == 'masks':
        masks(a.out, a.rai_dir, a.threshold, a.max_person, a.device)
    elif a.stage == 'solve':
        solve(a.out, ws_dir, a.iterations, a.guess, not a.no_search)
    else:
        verify(a.out, a.held, ws_dir)


if __name__ == '__main__':
    main()

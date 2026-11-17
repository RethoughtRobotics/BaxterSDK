#!/usr/bin/env python3
"""Compute fixed image corrections for Baxter wrist cameras.

Point both wrist cameras at a uniform white surface, then run:
    python3 scripts/calibrate_wrist_cameras.py

Saves camera_corrections.yaml (and per-camera vignetting .npy files) next to this script.
To preview corrections on a live frame:
    python3 scripts/calibrate_wrist_cameras.py --preview
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from sensor_msgs.msg import Image

CAMERAS = ('left_hand_camera', 'right_hand_camera')
TOPICS = {c: f'/cameras/{c}/image' for c in CAMERAS}
OUT_DIR = Path(__file__).resolve().parent


def grab_frames(n=5, timeout_s=15):
    rclpy.init()
    node = Node('calibrate_grab')
    counts = {c: 0 for c in CAMERAS}
    stacks = {c: [] for c in CAMERAS}

    def make_cb(cam):
        def cb(msg):
            if counts[cam] >= n:
                return
            h, w = msg.height, msg.width
            raw = np.frombuffer(msg.data, dtype=np.uint8)
            if msg.encoding == 'bgra8':
                img = raw.reshape(h, w, 4)[:, :, :3]
            elif msg.encoding == 'rgb8':
                img = cv2.cvtColor(raw.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
            else:
                img = raw.reshape(h, w, 3)
            stacks[cam].append(img.astype(np.float32))
            counts[cam] += 1

        return cb

    for cam in CAMERAS:
        node.create_subscription(Image, TOPICS[cam], make_cb(cam), 1)

    deadline = time.time() + timeout_s
    while time.time() < deadline and any(counts[c] < n for c in CAMERAS):
        rclpy.spin_once(node, timeout_sec=0.5)

    node.destroy_node()
    rclpy.shutdown()

    frames = {}
    for cam in CAMERAS:
        if stacks[cam]:
            frames[cam] = np.mean(stacks[cam], axis=0).astype(np.uint8)
            print(f'  {cam}: {len(stacks[cam])} frames, {frames[cam].shape[1]}x{frames[cam].shape[0]}')
        else:
            print(f'  {cam}: no frames received')
    return frames


def compute_vignetting(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=gray.shape[1] // 6)
    center_val = blurred[blurred.shape[0] // 2, blurred.shape[1] // 2]
    if center_val < 1:
        return np.ones_like(gray)
    gain = center_val / np.maximum(blurred, 1.0)
    return np.clip(gain, 1.0, 4.0)


def compute_wb_gains(frame):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    r = min(h, w) // 8
    patch = frame[cy - r : cy + r, cx - r : cx + r].astype(np.float32)
    b, g, red = patch[:, :, 0].mean(), patch[:, :, 1].mean(), patch[:, :, 2].mean()
    if g < 1:
        return [1.0, 1.0, 1.0]
    return [g / max(b, 1), 1.0, g / max(red, 1)]


def apply_corrections(frame, vignetting, wb_gains, gamma):
    img = frame.astype(np.float32)
    if vignetting is not None:
        for c in range(3):
            img[:, :, c] *= vignetting
    img[:, :, 0] *= wb_gains[0]
    img[:, :, 2] *= wb_gains[2]
    img = np.clip(img, 0, 255)
    lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img.astype(np.uint8), lut)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--preview', action='store_true', help='apply saved corrections to a live frame')
    p.add_argument('--gamma', type=float, default=1.4)
    p.add_argument('--frames', type=int, default=5, help='frames to average')
    p.add_argument('--output', type=Path, default=OUT_DIR / 'camera_corrections.yaml')
    args = p.parse_args()

    if args.preview:
        if not args.output.exists():
            raise SystemExit(f'No corrections at {args.output}; run without --preview first')
        with open(args.output) as f:
            data = yaml.safe_load(f)
        print('Grabbing frames...')
        frames = grab_frames(n=1)
        for cam in CAMERAS:
            if cam not in frames or cam not in data:
                continue
            entry = data[cam]
            vig = None
            vig_path = args.output.parent / entry.get('vignetting_file', '')
            if vig_path.exists():
                vig = np.load(vig_path)
            corrected = apply_corrections(frames[cam], vig, entry['wb_gains_bgr'], entry['gamma'])
            cv2.imwrite(str(args.output.parent / f'{cam}_raw.png'), frames[cam])
            cv2.imwrite(str(args.output.parent / f'{cam}_corrected.png'), corrected)
            print(f'  {args.output.parent / f"{cam}_raw.png"}')
            print(f'  {args.output.parent / f"{cam}_corrected.png"}')
        return

    print('Grabbing frames for calibration...')
    frames = grab_frames(n=args.frames)
    if not frames:
        raise SystemExit('No frames received from either camera')

    out = {}
    for cam in CAMERAS:
        if cam not in frames:
            continue
        vig = compute_vignetting(frames[cam])
        wb = compute_wb_gains(frames[cam])
        vig_path = args.output.parent / f'{cam}_vignetting.npy'
        np.save(vig_path, vig.astype(np.float32))
        out[cam] = {
            'wb_gains_bgr': [round(float(x), 4) for x in wb],
            'gamma': round(float(args.gamma), 3),
            'vignetting_file': vig_path.name,
        }
        print(f'  {cam}: wb={[round(x, 3) for x in wb]}, gamma={args.gamma}')

    out['_meta'] = {
        'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'description': 'Fixed camera corrections for dataset annotation.',
    }
    with open(args.output, 'w') as f:
        yaml.dump(out, f, default_flow_style=False, sort_keys=False)
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()

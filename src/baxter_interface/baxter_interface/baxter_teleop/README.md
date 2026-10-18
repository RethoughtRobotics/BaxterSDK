# baxter_teleop
This package provides cartesian delta teleoperation for Baxter arms.

## Design intent

The runtime is modality agnostic.
It consumes cartesian deltas and maps them to joint position commands through differential inverse kinematics.
Keyboard is only one adapter.
VR or learned policies can plug into the same runtime path.

## Module layout

- `cartesian_delta_teleop.py`
  - Runtime orchestration.
  - Builds arm specific model from Baxter URDF.
  - Exposes `apply_delta_cartesian` and `run_delta_source`.
- `diff_ik.py`
  - Differential IK solver.
  - Damped least squares with bounded joint step.
- `delta_guard.py`
  - Validation policy for modality delta streams.
  - Enforces velocity, magnitude, and jump constraints.
- `keymap.py`
  - Keyboard to cartesian delta mapping.
- `vr_target.py`
  - VR adapter: servos the TCP toward the libsurvive_ros2 clutched target.
- `urdf_tools.py`
  - Baxter URDF discovery and arm chain extraction.

## Data contract for modalities

A modality source should provide one of:

- `None` when no command is available
- `np.ndarray` with shape `(6,)` as `[dx, dy, dz, droll, dpitch, dyaw]`
- `'quit'` to stop the loop

## Using keyboard mode

From the workspace root:

```bash
source install/setup.bash
ros2 run baxter_interface baxter_teleop --arm right
```

Use `--arm left` for the left arm.
Keyboard gripper control is included in this runtime:

- `[` decreases gripper opening by a small delta
- `]` increases gripper opening by a small delta
- `G` calibrates the gripper before delta control is used

The delta size defaults to 2 percent per keypress and can be changed with `--gripper-step`.

Posture keys:

- `H` moves the arm to the untuck pose, as `tuck_arms -u` (slow, blocking)

## Using VR mode

Needs the libsurvive_ros2 driver running (same Kilted + Zenoh setup). Controller
serials, anchor/target frames and topics live in `libsurvive_ros2/config/vive_devices.yaml`.

```bash
./run_teleop.sh --right --input vr
```

- Touch the trackpad (the clutch) and move; the arm follows. Release it and the arm
  stops where it is; the next touch continues from there. Press A to re-anchor
  (reset the controller axes to how you hold it now).
- The keyboard still handles gripper keys, `H` (rest) and ESC.

The target (`/vive/{arm}/target`, a `PoseStamped` in the anchor frame) drives the
TCP. Rotation is applied in TCP axes. Translation depends on `linear_frame` in the
mapping YAML, chosen with `--vr-config` (a path or a file name in `config/`):

- `vr_axes_base.yaml` (`run_teleop.sh` default): translation in the anchor frame,
  mapped once onto the robot base frame, so rotating the wrist never re-aims it.
  Press A to save the controller's horizontal heading: the driver fixes anchor `z` to
  gravity up and discards pitch/roll, so wrist tilt cannot create vertical drift.
- `vr_axes.yaml` (the node's default): translation in the controller axes at clutch,
  then the TCP axes at clutch. The map follows the wrist, so later strokes drift
  when the TCP orientation changes.

Neither depends on where the base stations are. Both YAMLs hold signed axis
permutations (e.g. `['-y', '+x', '+z']`) for translation (`linear_axes`) and
rotation (`angular_axes`) and scale velocity per TCP axis. `--vr-gain` (1/s) sets
how fast the TCP closes the gap, capped by `--linear-speed` / `--angular-speed`
(VR defaults `2.0` m/s and `5.0` rad/s).

## Control modes

- `--mode velocity` (default): joint velocity commands computed from the measured
  state each cycle. Baxter's own controller handles gravity, and the arm stops
  within `0.1 s` of the last key repeat. Speeds are real: `--linear-speed` (m/s,
  default `0.20`) and `--angular-speed` (rad/s, default `0.8`).
- `--mode position`: accumulated joint position targets, bounded to lead the
  measured arm by at most `0.2 rad` per joint.

A terminal cannot detect key release, so the first keypress stays active for
`--key-hold-initial` seconds (default `0.55`) to bridge the OS auto-repeat delay.
A single tap therefore moves for about half a second. Lowering the OS repeat
delay (e.g. `gsettings set org.gnome.desktop.peripherals.keyboard delay 200`)
lets you lower `--key-hold-initial` to match.

## Command frame and TCP

- `--frame tool`: keys are expressed in the TCP axes. Baxter's hand
  z axis points out through the gripper, so `R`/`F` approach and retract.
- `--frame base` (default): keys are expressed in the robot base axes (x forward, y left, z up).

In both frames rotations pivot about the TCP. The TCP is measured at startup
from the robot's `endpoint_state` (which includes the configured gripper and
fingers) and expressed in the `{arm}_hand` frame. Override it with
`--tcp-offset <meters along hand z>`.

## Redundancy resolution

Baxter has 7 joints for a 6-DOF hand task. The spare DOF is left free: there is
no posture term, so the joints move only to realize the commanded twist

```
qdot = J⁺·v        (damped least squares, minimum norm)
```

as with SNS-IK. A posture pull toward a fixed rest posture leaked into the hand
motion: from the untuck pose, 0.1 m/s key moves came out 38-55 mm off-axis per
100 mm in simulation (BaxFlow-T `launch_sim.sh --ros`), against ~1 mm without it.

## Safety limits

The runtime enforces both velocity and delta limits before each command is applied.

- linear velocity limit in m/s
- angular velocity limit in rad/s
- linear delta limit per cycle in meters
- angular delta limit per cycle in radians
- maximum jump between consecutive deltas

These are configurable from CLI:

```bash
ros2 run baxter_interface baxter_teleop \
  --arm right \
  --max-linear-velocity 0.20 \
  --max-angular-velocity 1.50 \
  --max-linear-delta 0.010 \
  --max-angular-delta 0.200 \
  --max-delta-jump 0.050
```

If a command exceeds any threshold it is rejected and a warning is logged.

## URDF resolution

The runtime resolves the Baxter URDF in this order:

1. `BAXTER_FRAX_URDF` environment variable
2. `baxter_description` package share via ROS index
3. workspace relative discovery from source checkout

## Extending to another modality

Create a callback that returns the modality delta in the contract format and pass it to `run_delta_source`.

```python
teleop = CartesianDeltaTeleop(arm='right')
teleop.run_delta_source(my_delta_source)
```

No changes are needed in the IK solver for new modalities.

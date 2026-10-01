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

- `T` / `Y` swing the elbow (redundant 7th DOF) without moving the hand
- `H` moves the arm to the rest posture (slow, blocking)

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

## Redundancy resolution

Baxter has 7 joints for a 6-DOF hand task. Like the Franka cartesian impedance
controller, the spare DOF is resolved with a nullspace posture attractor:

```
qdot = J⁺·v + (I − J⁺J)·k·(q_rest − q)
```

`q_rest` is Baxter neutral with a more bent elbow (`e1 = 1.2`), which is better
conditioned than neutral. `--null-gain` sets `k` in 1/s (`0` disables). The
elbow keys move `q_rest[e0]`.

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

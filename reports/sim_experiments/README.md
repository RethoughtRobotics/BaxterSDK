# Sim experiments: results

Results of the test-time training plan in RethoughtAI `training/sim_experiments.md` (the TTT plan,
claims H-S3 to H-S24; the plan's former part 1 on the base's failures and on more demonstrations was
dropped). One report per claim in this folder; this page holds the plan's results table and one line
per claim. Rows fill in as conditions are evaluated.

## Setup in short

- Task: the cup task (`BlueCupRedContainer`), MuJoCo sim of `baxter_mujoco`, run closed loop through
  the robot's policy runtime, teleop, IK and gripper; success scored from the sim's object poses.
- Test: 50 layouts (seeds 1000 to 1049), 40 s per attempt, paired across conditions. Differences come
  with a 95% bootstrap interval over layouts; an effect counts only if the interval excludes zero and
  it exceeds the noise floor.
- Bases: VITA trained 30k steps on `andnetdeboer/BlueCupRedContainer_sim_lead640ms` subsets (the sim
  data with the pose labels led by 16 steps, the robot's convention), EMA weights, run with the
  robot's runtime profile.
- Adaptations: three update seeds each; the difference column averages the seeds per layout.

## Results

Closed loop, 50 test layouts. Difference against M0 (or against S0 for bases), in layouts of 50, 95%
interval in brackets; forgetting is the mean change on the layouts S0 solved (minus 1: all lost).
Success of adaptations is the mean of three update seeds.

| # | Success of 50 | Difference | Forgetting | Time to success (s) | Corrected frames | Update compute (s) | Main failure mode |
|---|---|---|---|---|---|---|---|
| S0 | 6 | | | 19.3 | | | missed grasp |
| S0b | 2 | | | | | | stalled |
| S0c | 3 | | | | | | missed grasp |
| M0 | 3.0 (4, 3, 2) | -3.0 [-7.3, +0.7] vs S0 | -0.72 | 16.3 | 363 | 2.1 | missed grasp |
| S3a to S8c | pending | | | | | | |
| S9a | 4.0 (4, 4, 4) | +1.0 [-1.7, +4.0] | -0.72 | 16.7 | 363 | 2.9 | missed grasp |
| S9b | 4.0 (4, 6, 2) | +1.0 [-0.3, +2.7] | -0.78 | 17.2 | 363 | 2.5 | missed grasp |
| S9c | 2.7 (3, 3, 2) | -0.3 [-2.3, +2.0] | -0.78 | 19.6 | 363 | 2.6 | missed grasp |
| S10a to S14 | pending | | | | | | |
| S15 | 0.0 (0, 0, 0) | -3.0 [-6.0, -0.7] | -1.00 | | 363 | 2.4 | stalled |
| S16a to S19b | pending | | | | | | |
| S20 (DiT base) | 0 | -6 vs S0 | | | | | stalled |
| S20-M | pending | | | | | | |
| S21a (5 demonstrations) | 0 | -6 vs S0 | | | | | missed grasp |
| S21a-M | pending | | | | | | |
| S21b (55 demonstrations) | 18 | +12 vs S0 | | 13.9 | | | missed grasp |
| S21b-M | pending | | | | | | |
| S22 to S23b-M | pending | | | | | | |

Update compute is measured on cluster A100 GPUs, not the robot computer.

## Claims

| Claim | Report | Verdict |
|---|---|---|
| H-S3 which weights | | pending |
| H-S4 retraining (DAgger) | | pending |
| H-S5 residual policy | | pending |
| H-S6 latent steering | | pending |
| H-S7 reward | | pending |
| H-S8 where the adapter sits | | pending |
| H-S9 capacity | [H-S9_rank.md](H-S9_rank.md) | inconclusive: no rank beats another, but M0 itself shows no gain over the base to scale |
| H-S10 replay | | pending |
| H-S11 the negative | | pending |
| H-S12 which frames | | pending |
| H-S13 how long | | pending |
| H-S14 blend dial | | pending |
| H-S15 on-policy corrections | [H-S15_onpolicy.md](H-S15_onpolicy.md) | falsified: on-policy planner corrections fall below M0 (0 of 50); a corrector and demonstration mismatch is not yet ruled out |
| H-S16 correction quality | | pending |
| H-S17 how many corrections | | pending |
| H-S18 update schedule | | pending |
| H-S19 unseen conditions | | pending |
| H-S20 other bases | | pending |
| H-S21 weak and strong bases | | pending |
| H-S22 repeated adaptation | | pending |
| H-S23 chunk execution | | pending |
| H-S24 the robot | | pending, needs the robot |

## Threats common to all claims

- **Weak bases.** The 21-demonstration base succeeds on 6 of 50 layouts (training seeds: 6, 2, 3).
  Small gains and forgetting are measured on few solved layouts.
- **Label lead.** Bases first trained on the raw sim labels (the TCP at the same step, no lead)
  stalled at home: every demonstration waits about 1.1 s before moving, and a plan made at home
  repeats the wait (0 to 4 of 50 even with a run-time lead). The robot pipeline met the same problem
  and trains on labels led by 640 ms; the sim bases were retrained the same way. Those zero-lead
  results (S0 4, S0b 4, S0c 3, S2c 15 of 50) are kept as a documented negative and are not used by
  any claim.

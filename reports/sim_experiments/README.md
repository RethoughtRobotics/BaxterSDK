# Sim experiments: results

Results of the test-time training plan in RethoughtAI `training/sim_experiments.md` (the TTT plan,
claims H-S3 to H-S24; the plan's former part 1, H-S1 on the base's failures and H-S2 on more
demonstrations, was dropped from the plan, and its two reports use the plan as written on
2026-10-20). One report per claim in this folder; this page holds the plan's results table and one
line per claim. Rows fill in as conditions are evaluated.

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
| S0b | 2 | -4.0 [-9.0, +0.0] vs S0 | | 22.2 | | | stalled |
| S0c | 3 | -3.0 [-9.0, +3.0] vs S0 | | 16.8 | | | missed grasp |
| S2a (25 demonstrations) | 7 | +4.0 [-1.0, +9.3] vs M0 | -0.67 | 16.9 | | | missed grasp |
| S2b (37 demonstrations) | 11 | +8.0 [+2.0, +14.7] vs M0 | -0.67 | 16.3 | | | missed grasp |
| S2c (55 demonstrations) | 18 | +15.0 [+8.0, +22.0] vs M0 | -0.50 | 13.9 | | | missed grasp |
| M0 | 3.0 (4, 3, 2) | -3.0 [-7.3, +0.7] vs S0 | -0.72 | 16.3 | 363 | 2.1 | missed grasp |
| S3a to S3d | pending | | | | | | |
| S4 (retrained, DAgger) | 6 | +3.0 [-1.3, +7.7] | -0.67 | 16.4 | 363 (714 added) | 4080 (H100) | missed grasp |
| S5 to S8c | pending | | | | | | |
| S9a | 4.0 (4, 4, 4) | +1.0 [-1.7, +4.0] | -0.72 | 16.7 | 363 | 2.9 | missed grasp |
| S9b | 4.0 (4, 6, 2) | +1.0 [-0.3, +2.7] | -0.78 | 17.2 | 363 | 2.5 | missed grasp |
| S9c | 2.7 (3, 3, 2) | -0.3 [-2.3, +2.0] | -0.78 | 19.6 | 363 | 2.6 | missed grasp |
| S10a | 2.3 (1, 3, 3) | -0.7 [-1.7, +0.0] | -0.78 | 18.1 | 363 | 1.7 | missed grasp |
| S10b | 3.3 (2, 3, 5) | +0.3 [-1.7, +2.7] | -0.78 | 15.9 | 363 | 2.1 | missed grasp |
| S10c | 4.0 (4, 4, 4) | +1.0 [-1.3, +3.3] | -0.61 | 16.9 | 363 | 2.3 | missed grasp |
| S11, S12a to S12c | pending | | | | | | |
| S13a | 6.0 (6, 8, 4) | +3.0 [+0.3, +6.0] | -0.61 | 16.1 | 363 | 0.8 | missed grasp |
| S13b | 5.7 (7, 4, 6) | +2.7 [+0.7, +5.0] | -0.56 | 16.8 | 363 | 8.6 | missed grasp |
| S13c | 4.3 (5, 5, 3) | +1.3 [-0.7, +3.7] | -0.67 | 16.6 | 363 | 2.1 | missed grasp |
| S14 | pending | | | | | | |
| S15 | 0.0 (0, 0, 0) | -3.0 [-6.0, -0.7] | -1.00 | | 363 | 2.4 | stalled |
| S16a to S19b | pending | | | | | | |
| S20 (DiT base) | 0 | -6 vs S0 | | | | | stalled |
| S20-M | pending | | | | | | |
| S21a (5 demonstrations) | 0 | -6 vs S0 | | | | | missed grasp |
| S21a-M | 0.3 (0, 0, 1) | +0.3 [+0.0, +1.0] vs S21a | | 34.4 | 1404 | 2.1 | missed grasp |
| S21b (55 demonstrations) | 18 | +12 vs S0 | | 13.9 | | | missed grasp |
| S21b-M (partial: seed 1 at 38 of 50) | 15.3 (19, 12 of 38, 15) | -1.0 [-8.3, +6.3] vs S21b | -0.53 vs S21b | 13.0 | 14 | 2.2 | missed grasp |
| S22 to S23b-M | pending | | | | | | |

Update compute is measured on cluster A100 GPUs (S4's retraining on an H100), not the robot computer.

## Claims

| Claim | Report | Verdict |
|---|---|---|
| H-S1 the base's failures (former part 1) | [H-S1_base.md](H-S1_base.md) | falsified: failed layouts overlap across training seeds no more than chance (40 against 39.7); failure modes per layout do agree above chance |
| H-S2 more demonstrations (former part 1) | [H-S2_demos.md](H-S2_demos.md) | falsified: 25 demonstrations (7 of 50) match M0 (3.0), and 37 and 55 beat it |
| H-S3 which weights | | pending |
| H-S4 retraining (DAgger) | [H-S4_dagger.md](H-S4_dagger.md) | supported: S4 (6 of 50) is within noise of M0 (3.0) at about 1900 times the compute, and forgets no less; neither beats the base |
| H-S5 residual policy | | pending |
| H-S6 latent steering | | pending |
| H-S7 reward | | pending |
| H-S8 where the adapter sits | | pending |
| H-S9 capacity | [H-S9_rank.md](H-S9_rank.md) | inconclusive: no rank beats another, but M0 itself shows no gain over the base to scale |
| H-S10 replay | [H-S10_anchor.md](H-S10_anchor.md) | inconclusive: forgetting falls with lambda (clearly open loop, within noise closed loop); no lambda gains, so the gain part is untestable |
| H-S11 the negative | | pending |
| H-S12 which frames | | pending |
| H-S13 how long | [H-S13_steps.md](H-S13_steps.md) | inconclusive: 800 steps beat M0 (+2.7 [+0.7, +5.0]) but under the noise floor and only to the base's level; longer or faster updates do not forget more |
| H-S14 blend dial | | pending |
| H-S15 on-policy corrections | [H-S15_onpolicy.md](H-S15_onpolicy.md) | falsified: on-policy planner corrections fall below M0 (0 of 50); the corrector matches the demonstrations' convention, the gate selects off-path and pace frames |
| H-S16 correction quality | | pending |
| H-S17 how many corrections | | pending |
| H-S18 update schedule | | pending |
| H-S19 unseen conditions | | pending |
| H-S20 other bases | | pending |
| H-S21 weak and strong bases | [H-S21_strength.md](H-S21_strength.md) | inconclusive: no base gains beyond noise (+0.3, -3.0, -1.0 for 5, 21, 55 demonstrations; 55 partial); the middle base gains least |
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

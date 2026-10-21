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
| M0, update seeds 3 and 4 | 4, 5 (all five seeds: 3.6) | -2.4 [-6.6, +1.2] vs S0 (five seeds) | -0.70 (five seeds) | 14.6, 15.3 | 363 | | missed grasp |
| S3a (everything) | 2.7 (3, 3, 2) | -0.3 [-2.7, +2.0] | -0.67 | 21.4 | 363 | 12.6 | missed grasp |
| S3b (whole flow net) | 6.0 (5, 6, 7) | +3.0 [+0.3, +6.3] | -0.67 | 17.6 | 363 | 2.1 | missed grasp |
| S3c (last layer) | 1.7 (0, 4, 1) | -1.3 [-4.3, +1.0] | -0.94 | 17.5 | 363 | 1.1 | missed grasp |
| S3d (LoRA, no anchor) | 2.0 (2, 2, 2) | -1.0 [-3.7, +0.7] | -0.94 | 21.9 | 363 | 3.2 | missed grasp |
| S4 (retrained, DAgger) | 6 | +3.0 [-1.3, +7.7] | -0.67 | 16.4 | 363 (714 added) | 4080 (H100) | missed grasp |
| S5 (residual MLP) | 6.3 (8, 4, 7) | +3.3 [+0.7, +6.3] | -0.67 | 20.3 | 363 | 1.0 | missed grasp |
| S6 (latent offset) | 0.7 (0, 1, 1) | -2.3 [-5.3, +0.0] | -0.94 | 31.5 | 363 | 1.4 | missed grasp |
| S7a, S7b | pending | | | | | | |
| S8a (encoder) | 6.7 (7, 7, 6) | +3.7 [+0.7, +7.0] | -0.61 | 15.1 | 363 | 8.9 | missed grasp |
| S8b (decoder) | 5.7 (9, 4, 4) | +2.7 [-0.3, +5.7] | -0.56 | 18.7 | 363 | 6.4 | missed grasp |
| S8c (all three) | 2.3 (3, 1, 3) | -0.7 [-3.0, +1.3] | -0.78 | 20.4 | 363 | 14.4 | missed grasp |
| S9a | 4.0 (4, 4, 4) | +1.0 [-1.7, +4.0] | -0.72 | 16.7 | 363 | 2.9 | missed grasp |
| S9b | 4.0 (4, 6, 2) | +1.0 [-0.3, +2.7] | -0.78 | 17.2 | 363 | 2.5 | missed grasp |
| S9c | 2.7 (3, 3, 2) | -0.3 [-2.3, +2.0] | -0.78 | 19.6 | 363 | 2.6 | missed grasp |
| S10a | 2.3 (1, 3, 3) | -0.7 [-1.7, +0.0] | -0.78 | 18.1 | 363 | 1.7 | missed grasp |
| S10b | 3.3 (2, 3, 5) | +0.3 [-1.7, +2.7] | -0.78 | 15.9 | 363 | 2.1 | missed grasp |
| S10c | 4.0 (4, 4, 4) | +1.0 [-1.3, +3.3] | -0.61 | 16.9 | 363 | 2.3 | missed grasp |
| S11 (preference, beta 10) | 2.3 (3, 2, 2) | -0.7 [-2.3, +1.0] | -0.83 | 15.0 | 363 | 4.2 | missed grasp |
| S12a (all frames) | 3.0 (3, 4, 2) | +0.0 [-3.0, +2.7] | -0.83 | 18.2 | 1686 | 2.1 | missed grasp |
| S12b (1 cm gate) | 2.7 (3, 2, 3) | -0.3 [-2.0, +1.3] | -0.78 | 15.9 | 910 | 2.0 | missed grasp |
| S12c (6 cm gate) | 2.0 (2, 3, 1) | -1.0 [-4.0, +1.7] | -0.94 | 15.4 | 165 | 2.2 | missed grasp |
| S13a | 6.0 (6, 8, 4) | +3.0 [+0.3, +6.0] | -0.61 | 16.1 | 363 | 0.8 | missed grasp |
| S13b | 5.7 (7, 4, 6) | +2.7 [+0.7, +5.0] | -0.56 | 16.8 | 363 | 8.6 | missed grasp |
| S13c | 4.3 (5, 5, 3) | +1.3 [-0.7, +3.7] | -0.67 | 16.6 | 363 | 2.1 | missed grasp |
| S14 (alpha 0.25) | 7.0 (6, 10, 5) | +4.0 [+1.3, +7.3] | -0.44 | 16.4 | 363 (M0's) | (M0's) | missed grasp |
| S14 (alpha 0.5) | 5.3 (6, 4, 6) | +2.3 [-0.3, +5.3] | -0.72 | 17.6 | 363 (M0's) | (M0's) | missed grasp |
| S14 (alpha 0.75) | 2.3 (4, 3, 0) | -0.7 [-3.0, +1.7] | -0.78 | 15.1 | 363 (M0's) | (M0's) | missed grasp |
| S15 | 0.0 (0, 0, 0) | -3.0 [-6.0, -0.7] | -1.00 | | 363 | 2.4 | stalled |
| S16a (0.5 s late) | 0.0 (0 of 37, 0, 0) | +0.0 [+0.0, +0.0] vs S15 | -1.00 | | 358 | 2.2 | stalled |
| S16b (half chunk) | 0.0 (0, 0, 0) | +0.0 [+0.0, +0.0] vs S15 | -1.00 | | 363 | 2.2 | stalled |
| S16c (2 cm noise) | 0.0 (0, 0, 0) | +0.0 [+0.0, +0.0] vs S15 | -1.00 | | 363 | 5.5 | stalled |
| S16d (30% dropped) | 0.0 (0, 0, 0) | +0.0 [+0.0, +0.0] vs S15 | -1.00 | | 254 | 2.0 | stalled |
| S17 (1 attempt) | 3.7 (3, 2, 6) | +0.7 [-3.0, +4.0] | -0.89 | 16.7 | 127 | 2.0 | missed grasp |
| S17 (2 attempts) | 2.7 (3, 3, 2) | -0.3 [-4.0, +3.3] | -0.83 | 13.9 | 942 | 2.0 | missed grasp |
| S17 (4 attempts) | 1.7 (1, 2, 2) | -1.3 [-4.7, +1.3] | -0.94 | 21.0 | 2066 | 2.0 | stalled |
| S17 (8 attempts) | 0.0 (0, 0, 0) | -3.0 [-6.0, -0.7] | -1.00 | | 4553 | 2.3 | stalled |
| S17 (16 attempts) | 0.3 (0, 1, 0) | -2.7 [-5.7, -0.3] | -1.00 | 19.0 | 6755 | 2.0 | stalled |
| S18a (one update after 8 attempts) | 0.0 (0, 0, 0) | -3.0 [-6.0, -0.7] | -1.00 | | 4003 | 2.1 | stalled |
| S18b (an update after every attempt) | 1.0 (1, 1, 1) | -2.0 [-5.0, +0.7]; +1.0 [+0.0, +2.3] vs S18a | -1.00 | 28.9 | 3268 (2706, 3584, 3514) | 2.2 per update, 8 updates | stalled |
| S19a (cups below 0.65 m) | 0.0 (0, 0, 0); 0 below, 0 beyond | -3.0 [-6.0, -0.7]; gain vs S0 beyond 0.65 m -2.0 [-5.0, +0.0] of 23 | -1.00 | | 5845 | 2.1 | stalled |
| S19b (lab lighting) | 0.3 (0, 0, 1); 0.3 below, 0 beyond | -2.7 [-5.7, -0.3]; +0.3 [+0.0, +1.0] vs S15 | -1.00 | 32.7 | 363 | 2.1 | stalled |
| S20 (DiT base) | 0 | -6.0 [-11.0, -2.0] vs S0 | | | | | stalled |
| S20-M | 0.7 (1, 0, 1) | +0.7 [+0.0, +1.7] vs S20 | (S20 solved none) | 26.7 | 1263 | 25.0 | timeout |
| S21a (5 demonstrations) | 0 | -6 vs S0 | | | | | missed grasp |
| S21a-M | 0.3 (0, 0, 1) | +0.3 [+0.0, +1.0] vs S21a | | 34.4 | 1404 | 2.1 | missed grasp |
| S21b (55 demonstrations) | 18 | +12 vs S0 | | 13.9 | | | missed grasp |
| S21b-M (partial: seed 1 at 38 of 50) | 15.3 (19, 12 of 38, 15) | -1.0 [-8.3, +6.3] vs S21b | -0.53 vs S21b | 13.0 | 14 | 2.2 | missed grasp |
| S22 | pending | | | | | | |
| S23a (base, re-plan every 8) | 2 | -4.0 [-9.0, +0.0] vs S0 | -0.83 | 15.6 | | | missed grasp |
| S23a-M (M0's adapters, every 8) | 4.7 (6, 4, 4) | +2.7 [-0.3, +6.0] vs S23a; +1.7 [-1.0, +4.3] vs M0 | -0.78 | 16.8 | 363 (M0's) | (M0's) | missed grasp |
| S23b (base, every 31; plan: 32) | 4 | -2.0 [-8.0, +3.0] vs S0 | -0.83 | 18.4 | | | missed grasp |
| S23b-M (M0's adapters, every 31) | 2.0 (2, 3, 1) | -2.0 [-6.0, +1.3] vs S23b; -1.0 [-3.7, +1.7] vs M0 | -0.78 | 14.7 | 363 (M0's) | (M0's) | missed grasp |

Update compute is measured on cluster A100 GPUs (S4's retraining on an H100), not the robot computer.

## Claims

| Claim | Report | Verdict |
|---|---|---|
| H-S1 the base's failures (former part 1) | [H-S1_base.md](H-S1_base.md) | falsified: failed layouts overlap across training seeds no more than chance (40 against 39.7); failure modes per layout do agree above chance |
| H-S2 more demonstrations (former part 1) | [H-S2_demos.md](H-S2_demos.md) | falsified: 25 demonstrations (7 of 50) match M0 (3.0), and 37 and 55 beat it |
| H-S3 which weights | [H-S3_weights.md](H-S3_weights.md) | inconclusive: the whole flow net beats M0 (+3.0 [+0.3, +6.3]) but under the noise floor and only to the base's level; LoRA does not forget less than full fine-tuning |
| H-S4 retraining (DAgger) | [H-S4_dagger.md](H-S4_dagger.md) | supported: S4 (6 of 50) is within noise of M0 (3.0) at about 1900 times the compute, and forgets no less; neither beats the base |
| H-S5 residual policy | [H-S5_residual.md](H-S5_residual.md) | falsified: the residual MLP is not below M0 (6.3 of 50, +3.3 [+0.7, +6.3]), though only at the base's level |
| H-S6 latent steering | [H-S6_latent.md](H-S6_latent.md) | falsified: the offset falls below the base (0.7 of 50) and forgets most (-0.94), not nothing |
| H-S7 reward | | pending |
| H-S8 where the adapter sits | [H-S8_placement.md](H-S8_placement.md) | falsified: encoder (+3.7 [+0.7, +7.0]) and decoder (+2.7 [-0.3, +5.7]) adapters match or beat the flow net |
| H-S9 capacity | [H-S9_rank.md](H-S9_rank.md) | inconclusive: no rank beats another, but M0 itself shows no gain over the base to scale |
| H-S10 replay | [H-S10_anchor.md](H-S10_anchor.md) | inconclusive: forgetting falls with lambda (clearly open loop, within noise closed loop); no lambda gains, so the gain part is untestable |
| H-S11 the negative | [H-S11_negative.md](H-S11_negative.md) | falsified: the preference term adds no gain over M0 (-0.7 [-2.3, +1.0]) |
| H-S12 which frames | [H-S12_frames.md](H-S12_frames.md) | falsified: all frames match the 3 cm gate (+0.0 [-3.0, +2.7]); no gate stands out |
| H-S13 how long | [H-S13_steps.md](H-S13_steps.md) | inconclusive: 800 steps beat M0 (+2.7 [+0.7, +5.0]) but under the noise floor and only to the base's level; longer or faster updates do not forget more |
| H-S14 blend dial | [H-S14_blend.md](H-S14_blend.md) | inconclusive: success falls and forgetting grows with alpha from 0.25 up; alpha 0.75 dips below both ends only within noise (-0.7 [-3.0, +1.7] vs alpha 1) |
| H-S15 on-policy corrections | [H-S15_onpolicy.md](H-S15_onpolicy.md) | falsified: on-policy planner corrections fall below M0 (0 of 50); the corrector matches the demonstrations' convention, the gate selects off-path and pace frames |
| H-S16 correction quality | [H-S16_quality.md](H-S16_quality.md) | falsified, degenerate: every degradation is at 0 of 50 like S15, which has no gain to lose |
| H-S17 how many corrections | [H-S17_attempts.md](H-S17_attempts.md) | inconclusive: no number of attempts gains; success falls with more (3.7 at 1 to 0.3 at 16, -3.3 [-6.0, -1.0]), so saturation cannot be judged |
| H-S18 update schedule | [H-S18_schedule.md](H-S18_schedule.md) | inconclusive: S18b used fewer frames (3268 against 4003) at no lower success (+1.0 [+0.0, +2.3]), but both fail (1.0 and 0.0 of 50, below the base and M0), so there is no gain to reach more cheaply |
| H-S19 unseen conditions | [H-S19_generalise.md](H-S19_generalise.md) | falsified, degenerate: the gain beyond 0.65 m is within noise of zero (-2.0 [-5.0, +0.0] of 23), but S19a scores 0 of 50 on both halves and S19b 0.3, so there is no gain to transfer; lab lighting changes nothing |
| H-S20 other bases | [H-S20_bases.md](H-S20_bases.md) | falsified, degenerate: S20-M's interval against the DiT base includes zero (+0.7 [+0.0, +1.7]), but the DiT base scores 0 of 50 and VITA's own gain is not a gain (M0 -2.4 [-6.6, +1.2] over five seeds), so there is no gain to transfer |
| H-S21 weak and strong bases | [H-S21_strength.md](H-S21_strength.md) | inconclusive: no base gains beyond noise (+0.3, -3.0, -1.0 for 5, 21, 55 demonstrations; 55 partial); the middle base gains least |
| H-S22 repeated adaptation | | pending |
| H-S23 chunk execution | [H-S23_horizon.md](H-S23_horizon.md) | supported, weakly: the gain at 8 (+2.7) exceeds the gain at 16 (-3.0) by +5.7 [+1.3, +10.3], but mostly because the base drops at 8 (2 against 6, single evaluations); 31 was run for 32 (the runtime limit) |
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
- **Lighting.** The evaluator draws a random lighting per layout unless told to use the lab lighting;
  every closed-loop evaluation and on-policy attempt here ran with the random lighting (paired across
  conditions, so differences are unaffected, but absolute success may differ from the lab lighting).
- **Evaluation noise.** A closed-loop run is not deterministic: S3d's adapters are bit-identical to
  S10a's, and their two evaluations scored 2, 2, 2 and 1, 3, 3 of 50 (7 of 150 layout outcomes
  differ). Two repeat evaluations on the same 50 layouts show larger swings: M0's seed 0 adapter,
  identical file, scored 2 of 50 against 4 the first time (1 layout solved in both; paired difference
  -2.0 [-6.0, +2.0]), and the base S0 scored 7 against 6 (3 layouts solved in both; +1.0 [-4.0,
  +6.0]). Most solved layouts are not solved again: success here is a per-attempt chance, not a
  property of the layout. Differences of about 3 to 4 successes of 50 between single evaluations are
  within the evaluator's own repeat noise, on top of the update-seed and training-seed spread.
- **A low M0.** S3b, S5, S8a, S13a and S13b each score about 3 of 50 above M0 (3.0) with intervals
  that exclude zero, yet only reach the base's 6. M0's three update seeds may be a low draw, which
  favours every condition compared against it; claims decided by a difference to M0 near the noise
  floor are marked inconclusive or carry this threat. Two more update seeds of M0 (3 and 4) scored 4
  and 5 of 50, so M0 over five update seeds is 4, 3, 2, 4, 5, mean 3.6, against the 3.0 the reports
  use (three seeds). Every difference against M0 in the table shifts down by 0.6 (gain over S0: -2.4
  [-6.6, +1.2]; forgetting -0.70 against -0.72). No verdict changes: none rests on a difference to M0
  smaller than 0.6. With five seeds, the intervals of S3b (+2.4 [-0.6, +5.7]), S5 (+2.7 [-0.0, +5.8]),
  S13a (+2.4 [-0.2, +5.3]) and S13b (+2.1 [-0.1, +4.6]) reach zero, which strengthens their
  inconclusive or "matches M0" readings; S8a stays above (+3.1 [+0.1, +6.3]). S6's interval against M0
  moves off zero (-2.9 [-5.9, -0.6]), so H-S6's secondary count (S6 matches M0) weakens, but H-S6
  stays falsified on forgetting. H-S23's difference of gains becomes +5.1 [+0.9, +9.5] (from +5.7),
  still above the floor; H-S11 (-1.3 [-3.0, +0.4]) and H-S12 (S12a -0.6 [-3.6, +2.3]) still include
  zero. The table keeps the three-seed figures the reports used.

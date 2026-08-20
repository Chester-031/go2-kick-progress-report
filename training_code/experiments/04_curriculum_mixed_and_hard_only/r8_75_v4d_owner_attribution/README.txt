R8.75 V4D owner-attribution diagnostic

Purpose
  Isolate whether the observed failure belongs to the accepted Delta Pose low
  policy / 62D interface or to the R8.74b high-level policy.

What runs
  - 13 deterministic soccer cases.
  - Every case is instantiated three times in one vectorized Isaac process:
      accepted V4D high + accepted V4D low
      scripted geometric high controller + accepted V4D low
      R8.74b high + accepted V4D low
  - 8 additional direct high-action primitives using the accepted low:
      stand, forward, backward, left, right, turn left, turn right, safe wall approach
  - 47 environments total; 240 high steps by default.

Fairness controls
  Root pose, ball pose, root velocity and joint state are matched between each
  three-environment policy group.  All modes use the same low checkpoint in the
  same process.  The code aborts if root/ball equality fails.

Recorded outputs
  results/r875_summary.json
  results/r875_per_environment.csv
  results/r875_trace.csv.gz
  results/VERDICT.txt

The compressed trace contains every low step, the 17D high observation, raw and
clipped 3D high action, physical Delta Pose, labeled 14D context, full raw and
normalized 62D low observation, raw and clipped 12D low action, posture/action
rate, ball range/bearing, wall-distance proxy and selected skill.

Safety contract
  training=false
  optimizer_step=false
  backward=false
  video=false
  policy state is snapshotted before evaluation and must have exactly zero
  parameter/buffer delta afterward.

Cluster use
  cd /scratch/lj2634/Isaac-Go2-Low-Level-RL
  bash experiments/r8_75_v4d_owner_attribution/preflight_r875.sh
  sbatch experiments/r8_75_v4d_owner_attribution/run_r875_owner_attribution.sbatch

Check
  bash experiments/r8_75_v4d_owner_attribution/check_r875_latest.sh

Do not start PPO from this package.  Do not create a video unless the resulting
metrics show a material A/B difference; any later video must use the existing
verified local-render bundle chain.

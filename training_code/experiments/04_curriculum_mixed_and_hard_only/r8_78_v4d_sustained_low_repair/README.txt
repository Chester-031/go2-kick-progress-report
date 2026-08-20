R8.78 V4D sustained Delta Pose low repair
==========================================

Purpose
-------
R8.77 local telemetry proved that the high policy continuously requested
motion while the accepted low policy moved briefly and then converged to a
near-static attractor. R8.78 repairs the low policy only. It does not train or
select a high policy and it does not render video.

What is different
-----------------
1. Reassesses the existing R8.77 primitive trace using late windows, so the
   original max-displacement primitive gate cannot hide transient-only motion.
2. Warm-starts the exact accepted low checkpoint and freezes the accepted high
   checkpoint byte-for-byte.
3. Trains 300 low-only PPO iterations on continuously repeated symmetric Delta
   Pose primitives: stand, forward/backward, left/right, both turns, diagonal.
4. Adds a penalty for unresolved non-stand pose error with near-zero speed.
5. Requires fresh signed early/late velocity tests. All 8 primitives must pass,
   the response direction must be correct, late response must be retained, the
   robot must remain upright, low weights must change, and high weights must not.

Install and run
---------------
cd /scratch/lj2634/Isaac-Go2-Low-Level-RL

if [[ -e experiments/r8_78_v4d_sustained_low_repair ]]; then
  mv experiments/r8_78_v4d_sustained_low_repair \
     experiments/r8_78_v4d_sustained_low_repair_old_$(date +%Y%m%d_%H%M%S)
fi

tar --no-same-owner -xzf /scratch/lj2634/r8_78_v4d_sustained_low_repair_20260816.tgz \
  -C experiments

bash experiments/r8_78_v4d_sustained_low_repair/preflight_r878.sh
sbatch experiments/r8_78_v4d_sustained_low_repair/run_r878_sustained_low_repair.sbatch

Monitor
-------
bash experiments/r8_78_v4d_sustained_low_repair/check_r878_latest.sh

Expected result
---------------
R878_TRACE_VERDICT=ACCEPTED_LOW_SUSTAINED_FAILURE must appear before training.
At completion, PRE_PASS_COUNT must be below 8, POST_PASS_COUNT must be 8,
high_max_abs must be exactly 0, low_max_abs must be positive, and the verdict
must be R878_LOW_REPAIR_READY_FOR_PAIRED_GATE.

Stop rule
---------
If the post gate rejects the new low checkpoint, do not use it for high-level
training or local video. The next step after acceptance is a paired full-support
metric gate against the old low policy; video still comes only after that gate.

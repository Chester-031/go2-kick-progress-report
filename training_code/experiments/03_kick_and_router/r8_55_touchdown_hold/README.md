# R8.55 touchdown hold

R8.55 warm-starts R8.54 and keeps Kick64 plus Load/Lift/Swing frozen. It first
evaluates a deterministic touchdown latch, then trains Recovery64 with a
low/mid-priority curriculum.

When `all_now` first becomes true, the joint-position target that produced the
valid returned-foot touchdown is replayed for the next three control steps.
This is part of the deployable low-level action state machine. Held steps are
not falsely treated as fresh PPO samples.

```bash
cd /scratch/lj2634/Isaac-Go2-Low-Level-RL
mkdir -p experiments
tar -xzf /scratch/lj2634/r8_55_touchdown_hold_20260811.tgz -C experiments/

bash experiments/r8_55_touchdown_hold/preflight_r855.sh

R855_NUM_ENVS=128 \
R855_UPDATES=32 \
R855_ROLLOUT_STEPS=256 \
R855_BASELINE_EPISODES=256 \
R855_EVAL_EPISODES=1024 \
sbatch experiments/r8_55_touchdown_hold/run_r855_hold.sbatch
```

No-job-ID checker:

```bash
bash experiments/r8_55_touchdown_hold/check_r855_latest.sh
```

The checker reports the pre-training latched baseline, per-strength training
counts, final touchdown/recovery3 rates, contact safety, speed monotonicity,
logs, artifacts, and hashes.

# R8.54 strength-conditioned Recovery64

R8.54 warm-starts the completed R8.53 Recovery64 checkpoint. R8.51 Kick64 and
the accepted Load/Lift/Swing chain remain frozen.

The 64-D observation contract is preserved. Slot 63, originally the legacy
strike-strength slot, is overwritten only for Recovery with the actual
high-level desired strength (0.75, 1.0, or 1.25). The environment's internal
strength remains 1.0, preventing changes to accepted motion timing.

Training uses a high-to-low curriculum and dense recovery rewards for foot
return pose, selected-foot force, four-foot support, touchdown, simultaneous
return+touchdown, and the existing three-consecutive-step recovery hold.

## Install, preflight, submit

```bash
cd /scratch/lj2634/Isaac-Go2-Low-Level-RL
mkdir -p experiments
tar -xzf /scratch/lj2634/r8_54_strength_conditioned_recovery_20260811.tgz -C experiments/

bash experiments/r8_54_strength_conditioned_recovery/preflight_r854.sh

R854_NUM_ENVS=128 \
R854_UPDATES=28 \
R854_ROLLOUT_STEPS=256 \
R854_EVAL_EPISODES=1024 \
sbatch experiments/r8_54_strength_conditioned_recovery/run_r854_recovery.sbatch
```

## Checker (no job ID)

```bash
cd /scratch/lj2634/Isaac-Go2-Low-Level-RL
bash experiments/r8_54_strength_conditioned_recovery/check_r854_latest.sh
```

Acceptance is evaluated independently at all three strengths. Each must retain
at least 95% real strikes, reach 80% touchdown and 60% three-step recovery,
remain below 2% wrong-foot and body contact, and preserve monotonic kick speed.

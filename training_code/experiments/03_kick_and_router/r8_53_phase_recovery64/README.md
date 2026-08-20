# R8.53 phase-aware Recovery64 training

R8.53 reuses the accepted R8.42 Load/Lift/Swing chain and the true-contact
R8.51 Kick64. It intentionally does not use the proxy-trained R8.49 recovery
checkpoint. Recovery64 is initialized by migrating the accepted R8.39 vector
from normalized 70-D observations to `70D[0:63,69]` (64-D).

Only Recovery64 and its critic train. Kick64 and all earlier motion modules are
frozen. Desired kick strength is applied as an explicit Kick64 residual gain:
`0.75 -> 0.5`, `1.0 -> 1.0`, `1.25 -> 2.0`. The legacy environment strength is
kept at `1.0`, so it cannot alter the already accepted Load/Lift/Swing timing.

Recovery control begins only after filtered real contact, accepted Swing, and
`phase >= 0.78`. Existing strict return/touchdown gates remain evaluated from
`phase >= 0.88`.

## Install

From the project root on Torch:

```bash
mkdir -p experiments
tar -xzf /scratch/lj2634/r8_53_phase_recovery64_20260811.tgz -C experiments/
```

## Preflight and submit

The preflight runs on the login node and does not request a GPU:

```bash
cd /scratch/lj2634/Isaac-Go2-Low-Level-RL
bash experiments/r8_53_phase_recovery64/preflight_r853.sh

R853_NUM_ENVS=128 \
R853_UPDATES=20 \
R853_ROLLOUT_STEPS=256 \
R853_EVAL_EPISODES=1024 \
sbatch experiments/r8_53_phase_recovery64/run_r853_recovery64.sbatch
```

## Checker (no job ID required)

Use the same command while pending, running, or completed:

```bash
cd /scratch/lj2634/Isaac-Go2-Low-Level-RL
bash experiments/r8_53_phase_recovery64/check_r853_latest.sh
```

The checker shows Slurm state/account/resources, the full experiment contract,
all progress markers, log tail, fatal scan, artifacts and hashes, PPO losses,
every recovery sub-gate, true-contact/wrong-foot/body rates, kick speed and
displacement at all three requested strengths, and the final router verdict.

The first run is intentionally bounded. If it completes cleanly but is not
accepted, use its per-gate metrics to decide whether to extend Recovery64
training; do not restart the frozen action chain.

# R8.74: fresh V4D high-level training without curriculum

This is the strict experiment for the no-curriculum claim. It starts the V4D
high-level soccer policy from random initialization and does not load a high
checkpoint. The validated delta-pose low-level locomotion checkpoint is loaded
and frozen, so the experiment measures high-level ball/goal behavior rather
than relearning quadruped locomotion at the same time.

The R8.63 stationary full-support reset is used from the first iteration:
there is no progress-dependent difficulty and no targeted-failure replay. The
observation difficulty value is fixed to 1.0 and the sparse reward remains
`goal=+1`, `own_goal=-1`, other=0.

## Submit

```bash
cd /scratch/lj2634/Isaac-Go2-Low-Level-RL

if [[ -e experiments/r8_74b_v4d_fresh_no_curriculum_fix ]]; then
    mv experiments/r8_74b_v4d_fresh_no_curriculum_fix \
       experiments/r8_74b_v4d_fresh_no_curriculum_fix_old_$(date +%Y%m%d_%H%M%S)
fi

tar --no-same-owner -xzf /scratch/lj2634/r8_74b_v4d_fresh_no_curriculum_fix_20260815.tgz \
    -C experiments/

bash experiments/r8_74b_v4d_fresh_no_curriculum_fix/preflight_r874b.sh
sbatch experiments/r8_74b_v4d_fresh_no_curriculum_fix/run_r874b_fresh_no_curriculum.sbatch
```

The preflight only patches and compiles the source; it does not start Isaac or
consume a GPU. It also verifies that the low-level checkpoint exists.

## Check

```bash
cd /scratch/lj2634/Isaac-Go2-Low-Level-RL
bash experiments/r8_74b_v4d_fresh_no_curriculum_fix/check_r874b_latest.sh
```

After training, the fresh high checkpoint must be evaluated on the same 13
R8.71 scenarios and on an independent random full-support set. A successful
no-curriculum result requires learning from random initialization, no own
goals, and no systematic regression across back-facing, diagonal, and wall
states.

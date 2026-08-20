# Go2 Soccer R8.1 — Gated Unload Support Skill Pilot

R8.1 changes the **training strategy**, not the deployment interface.

The preceding R8 pilot completed all 600 PPO updates but converged to a stable
four-foot standing local optimum: final stage-success, strict-success, lift,
and swing fractions were all zero, while upright stability and violation rate
were good.  R8.1 directly addresses that evidence.

## Unchanged deployment contract

- Actor observation: **70-D, teacher-free**.
- Low-level output: **12 joint-position residuals**.
- High-level Router interface: unchanged.
- High-level soccer reward: unchanged (`goal +1`, `own goal -1`, otherwise 0)
  and **not used in this low-level skill pilot**.
- Ball-contact training: deferred until this pilot and its video are reviewed.
- Source warm start: accepted V4D low-level checkpoint by default.

## Strategy changes

### 1. Success-gated curriculum

Stages no longer advance because an iteration counter expired.  The policy
must achieve a recent completed-episode success rate before the next stage is
started:

1. `unload_1cm` — 1 cm micro-clearance target; success at 8 mm.
2. `lift_2cm`.
3. `lift_3cm`.
4. `lift_4cm`.
5. `swing_3cm` — forward displacement counts only while the foot is airborne.
6. `swing_6cm`.
7. `recovery` — unchanged strict recovery geometry.

If a pilot stage reaches its maximum stage budget without passing the gate,
training **stops at that stage**, later stages are not run, the checkpoint and
summary are still saved, the Slurm training job exits successfully, and the
video job renders that failed stage for diagnosis.

### 2. Standing-local-optimum removal

Once the target foot is commanded upward, keeping it planted is explicitly
negative.  Stability rewards remain, but their combined weight is no longer
large enough to make four-foot standing profitable while the lift target is
active.  Successful target tracking, lift progress, and airborne forward
progress dominate the skill reward.

### 3. Airborne-only swing metric

Forward travel is only accumulated into the swing-success metric when the
selected foot is above a stage-dependent airborne threshold.  Ground sliding
or body translation can no longer count as a swing.

### 4. Unload proxy

The current R8 wrapper does not expose a verified per-foot normal-contact-force
tensor.  R8.1 therefore uses selected-foot micro-clearance while the other
three feet remain geometrically low as the first-stage unload proxy.  This is
reward/evaluation information only; it is not added to Actor observation.

### 5. Completion guard repair

`completed_env_fraction` is retained for diagnostics only.  It is not an
acceptance gate because episode boundaries and PPO rollout chunks are not
aligned.  Pilot completion now means the PPO run itself completed cleanly or
stopped intentionally at a curriculum gate.  Full acceptance still requires
the strict physical gates.

## Strict physical target remains unchanged

For the final recovery skill:

- world selected-foot lift >= 0.040 m;
- airborne forward travel >= 0.060 m;
- base displacement <= 0.060 m;
- base-height error <= 0.030 m;
- upright error <= 0.020;
- at least 3 support feet;
- no violation.

## Pilot configuration

- 256 environments.
- Global cap: 900 PPO updates.
- Each curriculum stage has its own minimum and maximum update budget.
- Gate statistics are computed from recent **completed episodes**, not raw
  rollout chunks.
- Pilot gate window: 20 reporting iterations.
- Pilot gate requires at least 256 completed episodes in that window.
- The pilot automatically renders 8 episodes after training because a stage
  timeout is a diagnostic outcome rather than a Slurm failure.

## Install and submit

After extracting the package under `/scratch/lj2634`:

```bash
cd /scratch/lj2634/go2_soccer_r8_1_gated_support_skill
bash install_submit_go2_soccer_r8_support_skill.sh
```

The package installs into:

```text
/scratch/lj2634/Isaac-Go2-Low-Level-RL/experiments/r8_1_gated_support_skill
```

Runs are written under:

```text
/scratch/lj2634/runs/go2_soccer_r8_1_gated_support_skill
```

## Compact result check

```bash
cd /scratch/lj2634/go2_soccer_r8_1_gated_support_skill
bash check_go2_soccer_r8_support_skill_result.sh
```

The checker intentionally prints a short report: Slurm status, curriculum
gates, final metrics, a compact per-stage tail summary, and video status.

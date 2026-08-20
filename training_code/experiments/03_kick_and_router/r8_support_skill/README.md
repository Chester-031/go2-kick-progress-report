# R8 isolated support-transfer low-level skill

R8 stops tuning the deterministic support controller and trains the missing
capability directly as a low-level PPO skill.

## Interfaces preserved

- Actor observation: 70-D and teacher-free.
- Actor action: 12 joint-position residuals.
- High-level Router interface: unchanged.
- High-level soccer reward: unchanged (`goal +1`, `own goal -1`, otherwise 0).
- This package does not train or modify the high-level policy.
- Dense rewards exist only inside this isolated low-level skill task.

## Pilot curriculum

The 600-iteration pilot uses 256 environments and four stages:

1. `support_unload` — learn coordinated full-body unloading and a small FL lift.
2. `stable_lift` — reach at least 4 cm world-frame lift while remaining stable.
3. `forward_swing` — retain lift and produce forward foot travel.
4. `recovery` — return the foot and body to a stable stance.

The ball is parked away and contributes no reward or collision to this pilot.
Ball contact training is deliberately deferred until the numerical summary and
8-episode video are reviewed.

## Why this is the next method

R7.2 proved that coordinated support can make FL leave the ground, but the
frozen V4D stand/pose policy cannot maintain the required three-leg support.
That is a learned capability gap, not another Jacobian or threshold bug. R8
lets all 12 low-level action outputs learn the coordination while preserving
the exact downstream interface.

## Submit

```bash
cd /scratch/lj2634/go2_soccer_r8_support_skill
bash install_submit_go2_soccer_r8_support_skill.sh
```

The installer submits:

- `r8sup-pilot`: 256 envs, 600 iterations, 3-hour limit.
- `r8sup-video`: 1 env, 8 episodes, after the pilot completes.

## Monitor

```bash
RUN_ROOT=/scratch/lj2634/runs/go2_soccer_r8_support_skill
ROOT="$(tr -d '\r\n' < "${RUN_ROOT}/latest_r8_support_skill.txt")"
PILOT_JOB="$(tr -d '\r\n' < "${ROOT}/pilot/job_id.txt")"
VIDEO_JOB="$(tr -d '\r\n' < "${ROOT}/video/job_id.txt")"

squeue -j "${PILOT_JOB},${VIDEO_JOB}" \
  -o "%.18i %.20P %.24j %.3t %.12M %.12l %R"

grep -nE \
'R8_STAGE_START|R8_STAGE_GATE|R8_ITER|R8_FINAL_GUARD|R8_SUPPORT_PILOT_COMPLETE|R8_FATAL|RUN_RC=' \
  "${ROOT}/pilot/worker.log" | tail -n 220
```

## Pilot outputs

- `pilot/logs/support_model_final.pt`
- `pilot/logs/r8_support_summary.json`
- `video/r8_support_pilot.mp4`
- `video/r8_support_video_summary.json`

The pilot is diagnostic and does not automatically modify V4D or the high-level
Router. A full continuation and ball-contact stage should only follow after the
video shows genuine support transfer, lift, swing, and recovery.

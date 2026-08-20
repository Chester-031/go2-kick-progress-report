#!/usr/bin/env bash
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

set -Eeuo pipefail

: "${RUN_DIR:?RUN_DIR required}"
: "${R85_SOURCE:?R85_SOURCE required}"

PROJECT="/scratch/lj2634/Isaac-Go2-Low-Level-RL"
ISAACLAB_ROOT="/scratch/lj2634/IsaacLab_232_src_37ddf62"
ENV_ROOT="/scratch/lj2634/envs/isaaclab232_v2"
PY="${ENV_ROOT}/bin/python"
GLU_ROOT="/scratch/lj2634/vendor/libglu-9.0.0"

R83="/scratch/lj2634/runs/go2_soccer_r8_3_real_load_transfer_skill/r8_3_real_load_transfer_skill_20260807_100811/pilot/logs/support_model_final.pt"

R84_105929="/scratch/lj2634/runs/go2_soccer_r8_4_load_preserving_progressive_lift/r8_4_load_preserving_progressive_lift_20260807_105929/pilot/logs/support_model_final.pt"

R84_112445="/scratch/lj2634/runs/go2_soccer_r8_4_load_preserving_progressive_lift/r8_4_load_preserving_progressive_lift_20260807_112445/pilot/logs/support_model_final.pt"

for path in \
    "${PY}" \
    "${R85_SOURCE}" \
    "${R83}" \
    "${R84_105929}" \
    "${R84_112445}"
do
    test -e "${path}"
done

export LD_LIBRARY_PATH="${GLU_ROOT}/lib:${ENV_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="${ENV_ROOT}/lib/libstdc++.so.6${LD_PRELOAD:+:${LD_PRELOAD}}"

export PYTHONPATH="${PROJECT}/source/isaac_go2_low_level_rl:${ISAACLAB_ROOT}/source/isaaclab:${ISAACLAB_ROOT}/source/isaaclab_tasks:${PYTHONPATH:-}"

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export GIT_PYTHON_REFRESH=quiet

RUNTIME_ROOT="/scratch/lj2634/job_runtime/${SLURM_JOB_ID}_r85_probe"

export HOME="${RUNTIME_ROOT}/home"
export XDG_CACHE_HOME="${RUNTIME_ROOT}/cache"
export XDG_CONFIG_HOME="${RUNTIME_ROOT}/config"
export XDG_DATA_HOME="${RUNTIME_ROOT}/data"
export TMPDIR="${RUNTIME_ROOT}/tmp"
export MPLCONFIGDIR="${RUNTIME_ROOT}/matplotlib"

mkdir -p \
    "${HOME}" \
    "${XDG_CACHE_HOME}" \
    "${XDG_CONFIG_HOME}" \
    "${XDG_DATA_HOME}" \
    "${TMPDIR}" \
    "${MPLCONFIGDIR}" \
    "${RUN_DIR}"

export V4K1_SETTLE_MAX_STEPS=150
export V4K1_SETTLE_MIN_STEPS=25
export V4K1_SETTLE_REQUIRED_STEPS=10
export V4K1_MIN_STABLE_ACTIVATION_RATE=1.0
export V4K1_BALL_ARM_STEPS=4

# New physical contract:
# 3 consecutive steps of simultaneous unloading + lift.
export R85_SIMULTANEOUS_HOLD_STEPS=3

# Runner construction only. act_inference() is used; no PPO update occurs.
export R8_INIT_NOISE_STD=0.26
export R8_ENTROPY_COEF=0.008
export R8_EPOCHS=3
export R8_MINI_BATCHES=4
export R8_GAMMA=0.995
export R8_DESIRED_KL=0.012

echo "============================================================"
echo "R8.5 SIMULTANEOUS UNLOAD+LIFT CHECKPOINT PROBE"
echo "============================================================"
echo "JOB=${SLURM_JOB_ID}"
echo "HOST=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "SOURCE=${R85_SOURCE}"

run_probe() {
    local name="$1"
    local checkpoint="$2"

    local out="${RUN_DIR}/${name}"

    mkdir -p \
        "${out}/logs"

    echo
    echo "============================================================"
    echo "R85_PROBE_START=${name}"
    echo "CHECKPOINT=${checkpoint}"
    echo "============================================================"

    set +e

    srun \
        --export=ALL \
        --unbuffered \
        "${PY}" \
        "${R85_SOURCE}" \
        --task Isaac-Go2-Soccer-Milestone2-v0 \
        --low_level_checkpoint "${checkpoint}" \
        --trained_low_checkpoint "${checkpoint}" \
        --log_dir "${out}/logs" \
        --run_name "${name}" \
        --num_envs 256 \
        --max_iterations 1 \
        --kick_episode_steps 80 \
        --kick_fixed_leg left \
        --r8_profile pilot \
        --seed 43 \
        --device cuda:0 \
        --headless \
        --r85_probe_only \
        --r85_probe_stage lift_5mm \
        --r85_probe_episodes 512 \
        --r85_probe_summary "${out}/summary.json" \
        2>&1 |
    tee "${out}/worker.log"

    rc=${PIPESTATUS[0]}

    set -e

    echo "R85_PROBE_RC_${name}=${rc}"

    if [ "${rc}" -ne 0 ]; then
        touch "${out}/FAILED"
        return "${rc}"
    fi

    test -s "${out}/summary.json"
    grep -qF \
        'R8_5_PROBE_OK' \
        "${out}/worker.log"

    touch "${out}/COMPLETE"
}


run_probe \
    r83_final \
    "${R83}"

run_probe \
    r84_105929_final \
    "${R84_105929}"

run_probe \
    r84_112445_final \
    "${R84_112445}"


"${PY}" - \
    "${RUN_DIR}/r83_final/summary.json" \
    "${RUN_DIR}/r84_105929_final/summary.json" \
    "${RUN_DIR}/r84_112445_final/summary.json" \
    "${RUN_DIR}" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

summary_files = [
    Path(value)
    for value in sys.argv[1:4]
]

run_dir = Path(sys.argv[4])

rows = []

for path in summary_files:
    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    rows.append(
        {
            "name": path.parent.name,
            "checkpoint": payload[
                "checkpoint"
            ],
            "stage_success_rate": float(
                payload[
                    "stage_success_rate"
                ]
                or 0.0
            ),
            "strict_success_rate": float(
                payload[
                    "strict_success_rate"
                ]
                or 0.0
            ),
            "violation_rate": float(
                payload[
                    "violation_rate"
                ]
                or 0.0
            ),
            "completed_episodes": int(
                payload[
                    "completed_episodes"
                ]
            ),
            "means": payload["means"],
        }
    )

rows.sort(
    key=lambda row: (
        row["stage_success_rate"],
        -row["violation_rate"],
    ),
    reverse=True,
)

best = rows[0]

comparison = {
    "contract": {
        "stage": "lift_5mm",
        "success_lift_m": 0.004,
        "max_fl_load_ratio": 0.120,
        "min_support3_fz_N": 80.0,
        "simultaneous_hold_steps": 3,
        "requires_prelift_unload": True,
        "video": False,
        "training": False,
    },
    "ranking": rows,
    "best_candidate": best,
}

(run_dir / "probe_comparison.json").write_text(
    json.dumps(
        comparison,
        indent=2,
        sort_keys=True,
    ),
    encoding="utf-8",
)

(run_dir / "best_parent.txt").write_text(
    best["checkpoint"] + "\n",
    encoding="utf-8",
)

print(
    "[R8_5_PARENT_RANKING]"
)

for index, row in enumerate(
    rows,
    start=1,
):
    print(
        f"rank={index} "
        f"name={row['name']} "
        f"rate={row['stage_success_rate']:.6f} "
        f"violation={row['violation_rate']:.6f} "
        f"checkpoint={row['checkpoint']}"
    )

print(
    "R8_5_BEST_PARENT="
    + best["checkpoint"]
)

print(
    "R8_5_COMPARISON="
    + str(
        run_dir
        / "probe_comparison.json"
    )
)
PY

touch "${RUN_DIR}/R85_PROBE_COMPLETE"

echo
echo "R85_SIMULTANEOUS_PARENT_PROBE_COMPLETE"

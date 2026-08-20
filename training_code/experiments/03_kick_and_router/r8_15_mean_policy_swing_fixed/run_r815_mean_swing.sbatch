#!/usr/bin/env bash
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G

set -Eeuo pipefail

: "${RUN_DIR:?}"
: "${TRAIN_SOURCE:?}"
: "${AUDIT_SOURCE:?}"
: "${PARENT:?}"
: "${PARENT_AUDIT:?}"

PROJECT="/scratch/lj2634/Isaac-Go2-Low-Level-RL"

ISAACLAB_ROOT="/scratch/lj2634/IsaacLab_232_src_37ddf62"
ENV_ROOT="/scratch/lj2634/envs/isaaclab232_v2"
GLU_ROOT="/scratch/lj2634/vendor/libglu-9.0.0"

PY="${ENV_ROOT}/bin/python"

export LD_LIBRARY_PATH="${GLU_ROOT}/lib:${ENV_ROOT}/lib:${LD_LIBRARY_PATH:-}"

export LD_PRELOAD="${ENV_ROOT}/lib/libstdc++.so.6${LD_PRELOAD:+:${LD_PRELOAD}}"

export PYTHONPATH="${PROJECT}/source/isaac_go2_low_level_rl:${ISAACLAB_ROOT}/source/isaaclab:${ISAACLAB_ROOT}/source/isaaclab_tasks:${PYTHONPATH:-}"

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export GIT_PYTHON_REFRESH=quiet


RUNTIME="/scratch/lj2634/job_runtime/${SLURM_JOB_ID}_r815"

export HOME="${RUNTIME}/home"
export XDG_CACHE_HOME="${RUNTIME}/cache"
export XDG_CONFIG_HOME="${RUNTIME}/config"
export XDG_DATA_HOME="${RUNTIME}/data"
export TMPDIR="${RUNTIME}/tmp"
export MPLCONFIGDIR="${RUNTIME}/mpl"

mkdir -p \
    "${HOME}" \
    "${XDG_CACHE_HOME}" \
    "${XDG_CONFIG_HOME}" \
    "${XDG_DATA_HOME}" \
    "${TMPDIR}" \
    "${MPLCONFIGDIR}" \
    "${RUN_DIR}"


# ============================================================
# Fixed physical/runtime contract.
# ============================================================

export V4K1_SETTLE_MAX_STEPS=150
export V4K1_SETTLE_MIN_STEPS=25
export V4K1_SETTLE_REQUIRED_STEPS=10
export V4K1_MIN_STABLE_ACTIVATION_RATE=1.0
export V4K1_BALL_ARM_STEPS=4

export R852_BRIDGE_HOLD_STEPS=3
export R89_SWING_HOLD_STEPS=2


# ============================================================
# PPO structure unchanged except entropy disabled because
# std is intentionally frozen.
# ============================================================

export R815_LR=0.000004

export R8_ENTROPY_COEF=0.0
export R8_ROLLOUT_STEPS=48
export R8_EPOCHS=3
export R8_MINI_BATCHES=4
export R8_GAMMA=0.995
export R8_DESIRED_KL=0.012

export R8_CHECKPOINT_INTERVAL=5

export R84_GATE_WINDOW=20
export R84_GATE_MIN_EPISODES=256


train_one() {

    local name="$1"
    local parent="$2"
    local std="$3"
    local iterations="$4"

    local dst="${RUN_DIR}/${name}"

    mkdir -p \
        "${dst}/logs"

    export R815_FORCE_STD="${std}"
    export R815_ITERATIONS="${iterations}"

    echo
    echo "============================================================"
    echo "R815_TRAIN_START=${name}"
    echo "PARENT=${parent}"
    echo "STD=${std}"
    echo "ITERATIONS=${iterations}"
    echo "============================================================"

    srun \
        --export=ALL \
        --unbuffered \
        "${PY}" \
        "${TRAIN_SOURCE}" \
        --task Isaac-Go2-Soccer-Milestone2-v0 \
        --low_level_checkpoint "${parent}" \
        --log_dir "${dst}/logs" \
        --run_name "${name}" \
        --num_envs 512 \
        --max_iterations "${iterations}" \
        --kick_episode_steps 80 \
        --kick_fixed_leg left \
        --r8_profile pilot \
        --seed 43 \
        --device cuda:0 \
        --headless \
        > "${dst}/worker.log" \
        2>&1

    test -s \
        "${dst}/logs/support_model_final.pt"

    grep -qF \
        '[R815_STD_APPLIED]' \
        "${dst}/worker.log"

    echo "R815_TRAIN_DONE=${name}"
}


audit_one() {

    local name="$1"
    local checkpoint="$2"

    local dst="${RUN_DIR}/${name}"

    mkdir -p "${dst}"

    export R814_AUDIT_ONLY=1
    export R814_EPISODES=256
    export R814_CHECKPOINT_NAME="${name}"
    export R814_SUMMARY="${dst}/audit.json"

    echo
    echo "R815_AUDIT_START=${name}"

    srun \
        --export=ALL \
        --unbuffered \
        "${PY}" \
        "${AUDIT_SOURCE}" \
        --task Isaac-Go2-Soccer-Milestone2-v0 \
        --low_level_checkpoint "${checkpoint}" \
        --log_dir "${dst}/logs" \
        --run_name "${name}" \
        --num_envs 128 \
        --max_iterations 1 \
        --kick_episode_steps 80 \
        --kick_fixed_leg left \
        --r8_profile pilot \
        --seed 43 \
        --device cuda:0 \
        --headless \
        > "${dst}/worker.log" \
        2>&1

    test -s \
        "${dst}/audit.json"

    grep -qF \
        'R814_AUDIT_OK' \
        "${dst}/worker.log"

    echo "R815_AUDIT_DONE=${name}"
}


# ============================================================
# Phase A: std 0.20
# ============================================================

train_one \
    phaseA_std020 \
    "${PARENT}" \
    "0.20" \
    "30"


A_MODEL="${RUN_DIR}/phaseA_std020/logs/support_model_final.pt"


audit_one \
    audit_phaseA \
    "${A_MODEL}"


# ============================================================
# Decide whether Phase B is justified.
#
# Continue only if deterministic airborne behavior improved.
# ============================================================

"${PY}" - \
    "${PARENT_AUDIT}" \
    "${RUN_DIR}/audit_phaseA/audit.json" \
    "${RUN_DIR}/phaseA_decision.json" <<'PY'
import json
import sys
from pathlib import Path

parent_path = Path(sys.argv[1])
a_path = Path(sys.argv[2])
out = Path(sys.argv[3])

parent = json.loads(
    parent_path.read_text()
)

phase_a = json.loads(
    a_path.read_text()
)


def get_case(payload):
    for case in payload["cases"]:
        if (
            case["stage"] == "swing_6cm"
            and case["mode"] == "inference"
        ):
            return case

    raise RuntimeError(
        "swing_6cm inference case missing"
    )


p = get_case(parent)
a = get_case(phase_a)

pn = p["episodes"]
an = a["episodes"]

p_lift = (
    p["episode"]["max_lift_sum"]
    / pn
)

a_lift = (
    a["episode"]["max_lift_sum"]
    / an
)

a_lift4 = (
    a["episode"]["lift_ge_4cm"]
    / an
)

a_bridge = (
    a["episode"]["bridge_seen"]
    / an
)

a_swing = (
    a["episode"]["swing_seen"]
    / an
)

lift_gain = (
    a_lift - p_lift
)

# Continue if any deterministic physical capability appeared
# OR mean maximum lift improved by >= 3 mm.
continue_b = (
    a_lift4 > 0.0
    or a_bridge > 0.0
    or a_swing > 0.0
    or lift_gain >= 0.003
)

payload = {
    "parent_mean_lift_m":
        p_lift,

    "phaseA_mean_lift_m":
        a_lift,

    "mean_lift_gain_m":
        lift_gain,

    "phaseA_lift4_rate":
        a_lift4,

    "phaseA_bridge_rate":
        a_bridge,

    "phaseA_swing_rate":
        a_swing,

    "continue_phaseB":
        continue_b,
}

out.write_text(
    json.dumps(
        payload,
        indent=2,
        sort_keys=True,
    )
)

print(
    "R815_PHASE_A_DECISION "
    + json.dumps(payload)
)
PY


CONTINUE="$(
    "${PY}" - \
        "${RUN_DIR}/phaseA_decision.json" <<'PY'
import json
import sys

p = json.load(open(sys.argv[1]))

print(
    "1"
    if p["continue_phaseB"]
    else "0"
)
PY
)"


if [ "${CONTINUE}" = "1" ]; then

    echo
    echo "R815_PHASE_B_ENABLED"

    # ========================================================
    # Phase B: std 0.14
    # ========================================================

    train_one \
        phaseB_std014 \
        "${A_MODEL}" \
        "0.14" \
        "25"


    B_MODEL="${RUN_DIR}/phaseB_std014/logs/support_model_final.pt"


    audit_one \
        audit_phaseB \
        "${B_MODEL}"

else

    echo
    echo "R815_PHASE_B_SKIPPED_STD_ANNEAL_ALONE_INSUFFICIENT"

fi


touch "${RUN_DIR}/R815_COMPLETE"

echo
echo "R815_MEAN_POLICY_CONSOLIDATION_COMPLETE"

#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G

set -Eeuo pipefail

PROJECT_DIR="/scratch/lj2634/Isaac-Go2-Low-Level-RL"
ISAACLAB_ROOT="/scratch/lj2634/IsaacLab_232_src_37ddf62"
ENV_ROOT="/scratch/lj2634/envs/isaaclab232_v2"
GLU_ROOT="/scratch/lj2634/vendor/libglu-9.0.0"
PY="${ENV_ROOT}/bin/python"
TRAIN_SCRIPT="${PROJECT_DIR}/experiments/v4k1_r5_reset_diagnostic/train_go2_soccer_v4k1_r5_reset_diagnostic.py"

for name in RUN_DIR LOW_CHECKPOINT TASK; do
  [ -n "${!name:-}" ] || { echo "ERROR: ${name} required" >&2; exit 20; }
done

mkdir -p "${RUN_DIR}"
exec > >(tee -a "${RUN_DIR}/worker.log") 2>&1

echo "============================================================"
echo "V4K1-R5 RESET/ARMING COMPREHENSIVE DIAGNOSTIC"
echo "============================================================"
echo "JOB_ID=${SLURM_JOB_ID}"
echo "HOST=$(hostname)"
echo "RUN_DIR=${RUN_DIR}"
echo "LOW_CHECKPOINT=${LOW_CHECKPOINT}"

if [ -e "${GLU_ROOT}/lib/libGLU.so.1" ]; then
  export LD_LIBRARY_PATH="${GLU_ROOT}/lib:${ENV_ROOT}/lib:${LD_LIBRARY_PATH:-}"
else
  export LD_LIBRARY_PATH="${ENV_ROOT}/lib:${LD_LIBRARY_PATH:-}"
fi
export LD_PRELOAD="${ENV_ROOT}/lib/libstdc++.so.6${LD_PRELOAD:+:${LD_PRELOAD}}"
export PYTHONPATH="${PROJECT_DIR}/source/isaac_go2_low_level_rl:${ISAACLAB_ROOT}/source/isaaclab:${ISAACLAB_ROOT}/source/isaaclab_tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export GIT_PYTHON_REFRESH=quiet

RUNTIME_ROOT="/scratch/lj2634/job_runtime/${SLURM_JOB_ID}_v4k1_r5_reset_diag"
export HOME="${RUNTIME_ROOT}/home"
export XDG_CACHE_HOME="${RUNTIME_ROOT}/cache"
export XDG_CONFIG_HOME="${RUNTIME_ROOT}/config"
export XDG_DATA_HOME="${RUNTIME_ROOT}/data"
export TMPDIR="${RUNTIME_ROOT}/tmp"
export MPLCONFIGDIR="${RUNTIME_ROOT}/matplotlib"
mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${XDG_DATA_HOME}" "${TMPDIR}" "${MPLCONFIGDIR}"

sha256sum "${TRAIN_SCRIPT}" "${LOW_CHECKPOINT}" > "${RUN_DIR}/input_sha256.txt"

# case | seed | settle_max | soft_arm | diagnostic_steps
CASES=(
  "baseline_seed42|42|75|0|320"
  "baseline_seed43|43|75|0|320"
  "extended_settle_seed42|42|150|0|480"
  "soft_arm_seed42|42|75|1|320"
  "extended_soft_seed42|42|150|1|480"
)

for spec in "${CASES[@]}"; do
  IFS='|' read -r CASE_ID SEED SETTLE_MAX SOFT_ARM DIAG_STEPS <<< "${spec}"
  CASE_DIR="${RUN_DIR}/${CASE_ID}"
  mkdir -p "${CASE_DIR}/logs"

  export V4K1_SETTLE_MIN_STEPS=25
  export V4K1_SETTLE_MAX_STEPS="${SETTLE_MAX}"
  export V4K1_SETTLE_REQUIRED_STEPS=10
  export V4K1_BALL_ARM_STEPS=6
  export V4K1_DIAG_SOFT_ARM="${SOFT_ARM}"

  cat > "${CASE_DIR}/case_config.txt" <<EOF
case_id=${CASE_ID}
seed=${SEED}
num_envs=32
settle_min=25
settle_max=${SETTLE_MAX}
settle_required=10
ball_arm_steps=6
soft_arm=${SOFT_ARM}
diagnostic_steps=${DIAG_STEPS}
EOF

  echo
  echo "===== CASE ${CASE_ID} ====="
  set +e
  /usr/bin/timeout --signal=TERM --kill-after=30s 25m \
    srun --export=ALL --unbuffered "${PY}" "${TRAIN_SCRIPT}" \
      --task "${TASK}" \
      --low_level_checkpoint "${LOW_CHECKPOINT}" \
      --log_dir "${CASE_DIR}/logs" \
      --run_name "${CASE_ID}" \
      --num_envs 32 \
      --max_iterations 1 \
      --kick_episode_steps 60 \
      --kick_fixed_leg left \
      --seed "${SEED}" \
      --device cuda:0 \
      --headless \
      --reset_diagnostic_only \
      --reset_diagnostic_steps "${DIAG_STEPS}" \
      --reset_diagnostic_json "${CASE_DIR}/reset_diagnostic.json" \
      2>&1 | tee "${CASE_DIR}/diagnostic.log"
  RC="${PIPESTATUS[0]}"
  set -e
  echo "CASE_RC=${RC}"

  if [ "${RC}" -ne 0 ] \
     || [ ! -s "${CASE_DIR}/reset_diagnostic.json" ] \
     || ! grep -qF 'V4K1_R5_RESET_DIAGNOSTIC_COMPLETE' "${CASE_DIR}/diagnostic.log"
  then
    echo "ERROR: diagnostic case failed: ${CASE_ID}" >&2
    touch "${RUN_DIR}/V4K1_R5_RESET_DIAGNOSTIC_FAILED"
    exit 30
  fi
done

"${PY}" - "${RUN_DIR}" <<'PY_AGG'
from __future__ import annotations

from pathlib import Path
import json
import statistics
import sys

root = Path(sys.argv[1])
rows = []
capabilities = {}
for path in sorted(root.glob("*/reset_diagnostic.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = dict(data["summary"])
    summary["case_id"] = path.parent.name
    rows.append(summary)
    capabilities[path.parent.name] = data.get("capabilities", {})

expected = {
    "baseline_seed42",
    "baseline_seed43",
    "extended_settle_seed42",
    "soft_arm_seed42",
    "extended_soft_seed42",
}
found = {row["case_id"] for row in rows}
if found != expected:
    raise RuntimeError(f"case set mismatch: expected={sorted(expected)} found={sorted(found)}")

by_id = {row["case_id"]: row for row in rows}
baseline_rates = [
    by_id["baseline_seed42"]["activation_any_rate"],
    by_id["baseline_seed43"]["activation_any_rate"],
]
baseline_mean = statistics.fmean(baseline_rates)
exact = by_id["baseline_seed42"]
extended = by_id["extended_settle_seed42"]
soft = by_id["soft_arm_seed42"]
both = by_id["extended_soft_seed42"]

settle_gain = extended["activation_any_rate"] - exact["activation_any_rate"]
soft_gain = soft["activation_any_rate"] - exact["activation_any_rate"]
both_gain = both["activation_any_rate"] - exact["activation_any_rate"]

if soft_gain >= 0.05 and soft["activation_any_rate"] >= 0.96875:
    primary = "transient_arming_violation_full_reset"
elif settle_gain >= 0.05 and extended["activation_any_rate"] >= 0.96875:
    primary = "settle_budget_or_slow_stabilization"
elif both_gain >= 0.05 and both["activation_any_rate"] >= 0.96875:
    primary = "combined_slow_settle_and_transient_arming"
elif exact["settle_timeout_total"] > 0:
    primary = "unresolved_settle_failure"
elif exact["arming_unstable_total"] > 0 or exact["arming_speed_total"] > 0 or exact["arming_displacement_total"] > 0:
    primary = "unresolved_arming_failure"
else:
    primary = "not_reproduced_or_other"

aggregate = {
    "cases": rows,
    "capabilities": capabilities,
    "comparison": {
        "baseline_activation_mean": baseline_mean,
        "settle_extension_gain_seed42": settle_gain,
        "soft_arm_gain_seed42": soft_gain,
        "extended_soft_gain_seed42": both_gain,
        "primary_hypothesis": primary,
    },
}
(root / "aggregate_reset_diagnostic.json").write_text(
    json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8"
)

print("case_id                         activation  settle_to  arm_unstable  arm_speed  arm_disp  never")
print("-" * 102)
for row in rows:
    print(
        f"{row['case_id']:31s} "
        f"{row['activation_any_rate']:10.5f} "
        f"{row['settle_timeout_total']:10d} "
        f"{row['arming_unstable_total']:13d} "
        f"{row['arming_speed_total']:9d} "
        f"{row['arming_displacement_total']:8d} "
        f"{str(row['never_activated_envs'])}"
    )
print(f"baseline_activation_mean={baseline_mean}")
print(f"settle_extension_gain_seed42={settle_gain}")
print(f"soft_arm_gain_seed42={soft_gain}")
print(f"extended_soft_gain_seed42={both_gain}")
print(f"primary_hypothesis={primary}")
print(f"aggregate={root / 'aggregate_reset_diagnostic.json'}")
print("V4K1_R5_RESET_DIAGNOSTIC_AGGREGATE_COMPLETE")
PY_AGG

touch "${RUN_DIR}/V4K1_R5_RESET_DIAGNOSTIC_ACCEPTED"
echo "V4K1_R5_RESET_DIAGNOSTIC_ACCEPTED"

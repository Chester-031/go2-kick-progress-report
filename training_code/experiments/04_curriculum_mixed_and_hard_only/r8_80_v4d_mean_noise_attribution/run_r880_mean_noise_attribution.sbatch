#!/usr/bin/env bash
#SBATCH --account=torch_pr_1066_general
#SBATCH --partition=l40s_public
#SBATCH --job-name=r880-attr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-r880-%j.out

set -Eeuo pipefail
PROJECT="/scratch/lj2634/Isaac-Go2-Low-Level-RL"
EXP="$PROJECT/experiments/r8_80_v4d_mean_noise_attribution"
R879_RUN="/scratch/lj2634/runs/go2_soccer_r8_79_v4d_directional_low_repair/directional_15874488_20260816_201731"
SOURCE="$R879_RUN/source/train_r879.py"
LOW="$R879_RUN/logs/low_model_final.pt"
ROOT="/scratch/lj2634/runs/go2_soccer_r8_80_v4d_mean_noise_attribution"
PY="${ISAAC_PYTHON:-/scratch/lj2634/envs/isaaclab232_v2/bin/python}"
HIGH="/scratch/lj2634/runs/go2_soccer_v4d_joint/v4d_staged_joint_20260803_223518/full/logs/high_model_final.pt"
EXPECTED_HIGH="1a05964f6330b532fe944d24028c67b880664eee3218a984cf17db6bf8010397"

cd "$PROJECT"
for FILE in "$EXP/patch_r880_mean_noise_eval.py" "$SOURCE" "$LOW" "$HIGH"; do
  [[ -s "$FILE" ]] || { echo "R880_FATAL=missing:$FILE" >&2; exit 21; }
done
[[ "$(sha256sum "$HIGH" | awk '{print $1}')" == "$EXPECTED_HIGH" ]] || exit 22

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN="$ROOT/attribution_${SLURM_JOB_ID:-manual}_${STAMP}"
mkdir -p "$RUN"/{source,logs,runtime/home,runtime/cache,runtime/config,runtime/data,runtime/tmp,runtime/mpl}
printf '%s\n' "$RUN" > "$ROOT/latest_r880.txt"
"$PY" "$EXP/patch_r880_mean_noise_eval.py" --source "$SOURCE" --output "$RUN/source/train_r880.py"
"$PY" -m py_compile "$RUN/source/train_r880.py"

export PYTHONPATH="$PROJECT/source/isaac_go2_low_level_rl:$PROJECT/source:${PYTHONPATH:-}"
export HOME="$RUN/runtime/home"
export XDG_CACHE_HOME="$RUN/runtime/cache"
export XDG_CONFIG_HOME="$RUN/runtime/config"
export XDG_DATA_HOME="$RUN/runtime/data"
export TMPDIR="$RUN/runtime/tmp"
export MPLCONFIGDIR="$RUN/runtime/mpl"
export PYTHONUNBUFFERED=1 QT_QPA_PLATFORM=offscreen R873_HEADLESS_STRIP=1
export R876_CURRICULUM_HIGH_REPAIR=1
export R879_FORCE_LOCOMOTION_PRIMITIVES=1
export R880_MEAN_NOISE_ATTRIBUTION=1
export V4D_HIGH_CHECKPOINT="$HIGH"
unset R879_SUSTAINED_LOW_REPAIR R878_SUSTAINED_LOW_REPAIR R874B_FREEZE_LOW R876_FREEZE_LOW

set +e
setsid timeout --foreground --kill-after=30s 15m \
  "$PY" "$RUN/source/train_r880.py" --headless --num_envs 512 \
    --max_iterations 1 --seed 88001 --low_level_checkpoint "$LOW" \
    --log_dir "$RUN/logs" --run_name r880_mean_noise_attribution \
  > "$RUN/worker.log" 2>&1
RC=$?
set -e
printf 'PROCESS_RC=%s\n' "$RC" >> "$RUN/worker.log"

VALIDATION_RC=0
SUMMARY="$RUN/logs/r880_mean_noise_attribution.json"
if [[ "$RC" -eq 0 && -s "$SUMMARY" ]]; then
  "$PY" - "$SUMMARY" <<'PY' > "$RUN/validation.txt" 2>&1 || VALIDATION_RC=$?
import json, sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text())
assert d["training"] is False
assert float(d["high_parameter_delta"]["max_abs"]) == 0.0
assert float(d["low_parameter_delta"]["max_abs"]) == 0.0
assert d["verdict"] in {
 "R880_MEAN_POLICY_COLLAPSE_CONFIRMED",
 "R880_MECHANICAL_OR_OBJECTIVE_PERSISTENCE_FAILURE",
}
print("R880_ATTRIBUTION_VALIDATION_OK")
print("VERDICT=" + d["verdict"])
print("STD=" + json.dumps(d["std"], sort_keys=True))
print("MATERIALLY_HELPED=" + json.dumps(d["materially_helped_cases"]))
PY
else
  VALIDATION_RC=1
fi
(cd "$RUN" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
cat "$RUN/validation.txt" 2>/dev/null || true
echo "R880_RUN=$RUN"
exit "$(( RC != 0 ? RC : VALIDATION_RC ))"

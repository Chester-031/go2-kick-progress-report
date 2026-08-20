#!/usr/bin/env bash
#SBATCH --account=torch_pr_1066_general
#SBATCH --partition=l40s_public
#SBATCH --job-name=r877-gate
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=slurm-r877-%j.out

set -Eeuo pipefail

PROJECT="/scratch/lj2634/Isaac-Go2-Low-Level-RL"
EXP="$PROJECT/experiments/r8_77_v4d_checkpoint_gate_local_video"
R875_EXP="$PROJECT/experiments/r8_75_v4d_owner_attribution"
R876_ROOT="/scratch/lj2634/runs/go2_soccer_r8_76_v4d_curriculum_high_repair"
R876_RUN="$(tr -d '\r\n' < "$R876_ROOT/latest_r876.txt")"
ROOT="/scratch/lj2634/runs/go2_soccer_r8_77_v4d_checkpoint_gate"
PY="${ISAAC_PYTHON:-/scratch/lj2634/envs/isaaclab232_v2/bin/python}"

ACCEPTED="/scratch/lj2634/runs/go2_soccer_v4d_joint/v4d_staged_joint_20260803_223518/full"
ACCEPTED_HIGH="$ACCEPTED/logs/high_model_final.pt"
ACCEPTED_LOW="$ACCEPTED/logs/low_model_final.pt"
DIAGNOSE="$R875_EXP/diagnose_r875.py"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN="$ROOT/gate_${SLURM_JOB_ID:-manual}_${STAMP}"
mkdir -p "$RUN"/{source,results,logs,runtime/home,runtime/cache,runtime/config,runtime/data,runtime/tmp,runtime/mpl}
printf '%s\n' "$RUN" > "$ROOT/latest_r877.txt"

declare -A CANDIDATES=(
  [intermediate_199]="$R876_RUN/logs/high_model_199.pt"
  [hard_mix_449]="$R876_RUN/logs/high_model_449_joint_finetune.pt"
  [full_support_final]="$R876_RUN/logs/high_model_final.pt"
)

for FILE in "$DIAGNOSE" "$EXP/select_r877.py" "$ACCEPTED_HIGH" "$ACCEPTED_LOW" \
  "$R876_RUN/R876_READY_FOR_FULL_SUPPORT_EVALUATION" \
  "${CANDIDATES[intermediate_199]}" "${CANDIDATES[hard_mix_449]}" \
  "${CANDIDATES[full_support_final]}"; do
  [[ -s "$FILE" || -f "$FILE" ]] || {
    echo "R877_FATAL=missing:$FILE" >&2
    exit 21
  }
done

install -m 0644 "$DIAGNOSE" "$RUN/source/diagnose_r875.py"
install -m 0644 "$EXP/select_r877.py" "$RUN/source/select_r877.py"
"$PY" -m py_compile "$RUN/source/diagnose_r875.py" "$RUN/source/select_r877.py"

"$PY" - "$RUN/candidate_paths.json" \
  "${CANDIDATES[intermediate_199]}" \
  "${CANDIDATES[hard_mix_449]}" \
  "${CANDIDATES[full_support_final]}" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "intermediate_199": sys.argv[2],
    "hard_mix_449": sys.argv[3],
    "full_support_final": sys.argv[4],
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

cat > "$RUN/contract.txt" <<EOF
version=r8_77_v4d_checkpoint_gate_v1
training=false
optimizer_step=false
backward=false
video=false
cases=13
candidate_count=3
accepted_high=$ACCEPTED_HIGH
accepted_low=$ACCEPTED_LOW
r876_run=$R876_RUN
selection_before_video=true
EOF

export PYTHONPATH="$PROJECT/source/isaac_go2_low_level_rl:$PROJECT/source:${PYTHONPATH:-}"
export HOME="$RUN/runtime/home"
export XDG_CACHE_HOME="$RUN/runtime/cache"
export XDG_CONFIG_HOME="$RUN/runtime/config"
export XDG_DATA_HOME="$RUN/runtime/data"
export TMPDIR="$RUN/runtime/tmp"
export MPLCONFIGDIR="$RUN/runtime/mpl"
export PYTHONUNBUFFERED=1
export QT_QPA_PLATFORM=offscreen
export R873_HEADLESS_STRIP=1
export V4D_HIGH_CHECKPOINT="$ACCEPTED_HIGH"
unset R874B_FRESH_NO_CURRICULUM R874_FRESH_NO_CURRICULUM R874B_FREEZE_LOW

cd "$PROJECT"
for LABEL in intermediate_199 hard_mix_449 full_support_final; do
  OUT="$RUN/results/$LABEL"
  mkdir -p "$OUT" "$RUN/logs/$LABEL"
  echo "[R877_CASE_START] label=$LABEL checkpoint=${CANDIDATES[$LABEL]}"
  set +e
  setsid timeout --foreground --kill-after=30s 12m \
    "$PY" "$RUN/source/diagnose_r875.py" \
      --headless \
      --num_envs 47 \
      --seed 87501 \
      --high_level_episode_steps 300 \
      --low_level_steps_per_action 5 \
      --low_level_checkpoint "$ACCEPTED_LOW" \
      --trained_low_checkpoint "$ACCEPTED_LOW" \
      --log_dir "$RUN/logs/$LABEL/isaac" \
      --run_name "r877_${LABEL}" \
      --r875_candidate_high_checkpoint "${CANDIDATES[$LABEL]}" \
      --r875_output_dir "$OUT" \
      --r875_steps 240 \
    > "$RUN/logs/$LABEL/worker.log" 2>&1
  RC=$?
  set -e
  echo "[R877_CASE_END] label=$LABEL rc=$RC"
  [[ "$RC" -eq 0 && -s "$OUT/r875_summary.json" ]] || {
    tail -n 120 "$RUN/logs/$LABEL/worker.log" || true
    exit "$(( RC == 0 ? 31 : RC ))"
  }
done

"$PY" "$RUN/source/select_r877.py" \
  --results-root "$RUN/results" \
  --candidate-paths "$RUN/candidate_paths.json" \
  --output-dir "$RUN/results" \
  > "$RUN/selection.log" 2>&1

cat "$RUN/selection.log"
touch "$RUN/R877_CHECKPOINT_GATE_OK"
(cd "$RUN" && find . -type f ! -name SHA256SUMS -print0 \
  | sort -z | xargs -0 sha256sum > SHA256SUMS)
echo "R877_RUN=$RUN"
echo "VERDICT=R877_CHECKPOINT_SELECTED_FOR_LOCAL_VIDEO"

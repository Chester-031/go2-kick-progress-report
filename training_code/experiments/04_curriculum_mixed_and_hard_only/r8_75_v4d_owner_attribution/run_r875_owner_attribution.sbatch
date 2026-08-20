#!/usr/bin/env bash
#SBATCH --account=torch_pr_1066_general
#SBATCH --partition=l40s_public
#SBATCH --job-name=r875-owner
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=slurm-r875-%j.out

set -Eeuo pipefail

PROJECT="/scratch/lj2634/Isaac-Go2-Low-Level-RL"
EXP="$PROJECT/experiments/r8_75_v4d_owner_attribution"
ROOT="/scratch/lj2634/runs/go2_soccer_r8_75_v4d_owner_attribution"
PY="${ISAAC_PYTHON:-/scratch/lj2634/envs/isaaclab232_v2/bin/python}"

ACCEPTED_RUN="/scratch/lj2634/runs/go2_soccer_v4d_joint/v4d_staged_joint_20260803_223518/full"
ACCEPTED_HIGH="$ACCEPTED_RUN/logs/high_model_final.pt"
ACCEPTED_LOW="$ACCEPTED_RUN/logs/low_model_final.pt"
CANDIDATE_RUN="/scratch/lj2634/runs/go2_soccer_r8_74b_v4d_fresh_no_curriculum_fix/fresh_15834260_20260816_083308"
CANDIDATE_HIGH="$CANDIDATE_RUN/logs/high_model_final.pt"

EXPECTED_ACCEPTED_HIGH="1a05964f6330b532fe944d24028c67b880664eee3218a984cf17db6bf8010397"
EXPECTED_ACCEPTED_LOW="52acb55160f4d4db0ac5c3da41aadbc5383a90555c64ab41851ba7383989996a"
EXPECTED_CANDIDATE_HIGH="12b596f7627feaa4a5b557499ba681c042daa9323cce54ffa1339d4b9db4df3f"

STEPS="${R875_STEPS:-240}"
NUM_ENVS=47
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN="$ROOT/owner_${SLURM_JOB_ID:-manual}_${STAMP}"

cd "$PROJECT"
mkdir -p "$RUN/source" "$RUN/results" "$RUN/logs" \
  "$RUN/runtime"/{home,cache,config,data,tmp,mpl}
printf '%s\n' "$RUN" > "$ROOT/latest_r875.txt"

for FILE in "$EXP/diagnose_r875.py" "$ACCEPTED_HIGH" "$ACCEPTED_LOW" "$CANDIDATE_HIGH"; do
  [[ -s "$FILE" ]] || {
    echo "R875_FATAL=missing:$FILE" >&2
    exit 21
  }
done

verify_sha() {
  local file="$1"
  local expected="$2"
  local actual
  actual="$(sha256sum "$file" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || {
    echo "R875_FATAL=sha_mismatch:$file expected=$expected actual=$actual" >&2
    exit 22
  }
}

verify_sha "$ACCEPTED_HIGH" "$EXPECTED_ACCEPTED_HIGH"
verify_sha "$ACCEPTED_LOW" "$EXPECTED_ACCEPTED_LOW"
verify_sha "$CANDIDATE_HIGH" "$EXPECTED_CANDIDATE_HIGH"

install -m 0644 "$EXP/diagnose_r875.py" "$RUN/source/diagnose_r875.py"
"$PY" -m py_compile "$RUN/source/diagnose_r875.py"

cat > "$RUN/contract.txt" <<EOF
version=r8_75_v4d_owner_attribution_v1
training=false
optimizer_step=false
backward=false
video=false
num_envs=$NUM_ENVS
policy_cases=13
policy_modes=accepted_high,scripted,r874b_high
primitive_cases=8
same_initial_root_state=true
same_initial_joint_state=true
steps=$STEPS
accepted_high=$ACCEPTED_HIGH
accepted_high_sha256=$EXPECTED_ACCEPTED_HIGH
accepted_low=$ACCEPTED_LOW
accepted_low_sha256=$EXPECTED_ACCEPTED_LOW
candidate_high=$CANDIDATE_HIGH
candidate_high_sha256=$EXPECTED_CANDIDATE_HIGH
source_parent_sha256=5565dcc09dcc0283525201bf4a42dbcf1f63a50d6e5cc516f2cc02a525d00d0c
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

{
  echo "R875_RUN=$RUN"
  echo "R875_GPU_ALLOCATION=${CUDA_VISIBLE_DEVICES:-unset}"
  nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader || true
} > "$RUN/logs/resource_start.txt"

set +e
setsid timeout --foreground --kill-after=30s 35m \
  "$PY" "$RUN/source/diagnose_r875.py" \
    --headless \
    --num_envs "$NUM_ENVS" \
    --seed 87501 \
    --high_level_episode_steps 300 \
    --low_level_steps_per_action 5 \
    --low_level_checkpoint "$ACCEPTED_LOW" \
    --trained_low_checkpoint "$ACCEPTED_LOW" \
    --log_dir "$RUN/logs/isaac" \
    --run_name r875_owner_attribution \
    --r875_candidate_high_checkpoint "$CANDIDATE_HIGH" \
    --r875_output_dir "$RUN/results" \
    --r875_steps "$STEPS" \
  > "$RUN/worker.log" 2>&1
RC=$?
set -e

printf 'PROCESS_RC=%s\n' "$RC" >> "$RUN/worker.log"

{
  echo "process_rc=$RC"
  nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader || true
} > "$RUN/logs/resource_end.txt"

(cd "$RUN" && find . -type f ! -name SHA256SUMS -print0 \
  | sort -z | xargs -0 sha256sum > SHA256SUMS)

echo "R875_PROCESS_RC=$RC"
echo "R875_RUN=$RUN"
if [[ "$RC" -eq 0 && -s "$RUN/results/r875_summary.json" ]]; then
  echo "VERDICT=R875_METRICS_READY"
else
  echo "VERDICT=R875_DIAGNOSTIC_FAILED"
fi
exit "$RC"

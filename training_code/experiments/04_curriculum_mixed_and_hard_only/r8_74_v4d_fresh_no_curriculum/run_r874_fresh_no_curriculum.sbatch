#!/usr/bin/env bash
#SBATCH --account=torch_pr_1066_general
#SBATCH --partition=l40s_public
#SBATCH --job-name=r874-fresh
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-r874-%j.out

set -Eeuo pipefail
PROJECT="/scratch/lj2634/Isaac-Go2-Low-Level-RL"
R863_ROOT="/scratch/lj2634/runs/go2_soccer_r8_63_full_support_v4d"
ROOT="/scratch/lj2634/runs/go2_soccer_r8_74_v4d_fresh_no_curriculum"
PY="${ISAAC_PYTHON:-/scratch/lj2634/envs/isaaclab232_v2/bin/python}"
LOW="${R874_LOW_CHECKPOINT:-/scratch/lj2634/runs/go2_soccer_v4d_joint/v4d_staged_joint_20260803_223518/full/logs/low_model_final.pt}"
MAX_ITERS="${R874_MAX_ITERATIONS:-1500}"
NUM_ENVS="${R874_NUM_ENVS:-512}"

cd "$PROJECT"
export PYTHONPATH="$PROJECT/source/isaac_go2_low_level_rl:$PROJECT/source:${PYTHONPATH:-}"
[[ -s "$R863_ROOT/latest_r863_candidate.txt" ]] || { echo "R874_PARENT_POINTER_MISSING" >&2; exit 2; }
[[ -f "$LOW" ]] || { echo "R874_LOW_CHECKPOINT_MISSING=$LOW" >&2; exit 2; }
BASE="$(tr -d '\r\n' < "$R863_ROOT/latest_r863_candidate.txt")"
SRC="$BASE/source/train_v4d_full_support.py"
[[ -f "$SRC" ]] || { echo "R874_SOURCE_MISSING=$SRC" >&2; exit 2; }

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN="$ROOT/fresh_${SLURM_JOB_ID:-manual}_${STAMP}"
mkdir -p "$RUN/source" "$RUN/logs" "$RUN/checkpoints" "$RUN/runtime"/{home,cache,config,data,tmp,mpl}
export HOME="$RUN/runtime/home" XDG_CACHE_HOME="$RUN/runtime/cache" \
    XDG_CONFIG_HOME="$RUN/runtime/config" XDG_DATA_HOME="$RUN/runtime/data" \
    TMPDIR="$RUN/runtime/tmp" MPLCONFIGDIR="$RUN/runtime/mpl" \
    R874_FRESH_NO_CURRICULUM=1 R874_FREEZE_LOW=1 R873_HEADLESS_STRIP=1 \
    V4D_HIGH_CHECKPOINT= PYTHONUNBUFFERED=1

cp -a "$SRC" "$RUN/source/source.py"
cp -a "$PROJECT/experiments/r8_74_v4d_fresh_no_curriculum/patch_headless_train.py" "$RUN/source/"
cp -a "$PROJECT/experiments/r8_74_v4d_fresh_no_curriculum/patch_fresh_no_curriculum.py" "$RUN/source/"
"$PY" "$RUN/source/patch_headless_train.py" --source "$RUN/source/source.py" --out "$RUN/source/headless.py"
"$PY" "$RUN/source/patch_fresh_no_curriculum.py" --source "$RUN/source/headless.py" --out "$RUN/source/train.py"
"$PY" -m py_compile "$RUN/source/train.py"

cat > "$RUN/contract.txt" <<EOF
version=r8_74_v4d_fresh_no_curriculum
training=true
parent_mutation=false
fresh_high_policy=true
high_checkpoint=NONE_RANDOM_INIT
low_checkpoint=$LOW
low_policy=frozen_warm_started_delta_pose
progressive_curriculum=false
difficulty_observation=1.0
reset_distribution=stationary_full_support_random
targeted_failure_replay=false
reward_contract=goal=1,own_goal=-1,other=0
max_iterations=$MAX_ITERS
num_envs=$NUM_ENVS
EOF

set +e
setsid timeout --foreground --kill-after=30s 7h "$PY" "$RUN/source/train.py" \
    --headless --num_envs "$NUM_ENVS" --max_iterations "$MAX_ITERS" --seed 87401 \
    --low_level_checkpoint "$LOW" --log_dir "$RUN/logs" --run_name r874_fresh_no_curriculum \
    > "$RUN/worker.log" 2>&1
RC=$?
set -e
printf 'PROCESS_RC=%s\n' "$RC" >> "$RUN/worker.log"
printf '%s\n' "$RUN" > "$ROOT/latest_r874.txt"
for f in "$RUN"/logs/high_model_final.pt "$RUN"/logs/low_model_final.pt "$RUN"/logs/model_final.pt; do
    [[ -f "$f" ]] && cp -a "$f" "$RUN/checkpoints/"
done
(cd "$RUN" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
echo "R874_FRESH_NO_CURRICULUM_DONE rc=$RC"
echo "RUN=$RUN"
if [[ "$RC" == 124 || "$RC" == 137 ]]; then echo "VERDICT=R874_TRAINING_TIMEOUT";
elif [[ "$RC" != 0 ]]; then echo "VERDICT=R874_TRAINING_FAILED";
else echo "VERDICT=R874_TRAINING_FINISHED_CHECKPOINTS_REQUIRED"; fi

#!/usr/bin/env bash
#SBATCH --account=torch_pr_1066_general
#SBATCH --partition=l40s_public
#SBATCH --job-name=v56-task-progress
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --chdir=/scratch/lj2634/Isaac-Go2-Low-Level-RL/experiments/56_sparse_high_task_progress_canary_v1
#SBATCH --output=/scratch/lj2634/runs/go2_sparse_high_task_progress_canary_v1/slurm-v56-%j.out
#SBATCH --error=/scratch/lj2634/runs/go2_sparse_high_task_progress_canary_v1/slurm-v56-%j.err
#SBATCH --open-mode=truncate

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo 'ERROR=V56_SBATCH_MUST_BE_EXECUTED_NOT_SOURCED' >&2
  return 2
fi
set -Eeuo pipefail
umask 077
FAILURE_REASON=

PROJECT=${V56_PROJECT:-/scratch/lj2634/Isaac-Go2-Low-Level-RL}
SOURCE=${V56_SOURCE:-$PROJECT/experiments/56_sparse_high_task_progress_canary_v1}
ENV_ROOT=${V56_ENV_ROOT:-/scratch/lj2634/envs/isaaclab232_v2}
PY=${V56_CLUSTER_PY:-$ENV_ROOT/bin/python}
ISAAC_LAB=${V56_ISAAC_LAB:-/scratch/lj2634/IsaacLab_232_src_37ddf62}
GLU_ROOT=${V56_GLU_ROOT:-/scratch/lj2634/vendor/libglu-9.0.0}
RUN_ROOT=${V56_RUN_ROOT:-/scratch/lj2634/runs/go2_sparse_high_task_progress_canary_v1}
CURRENT_LOW=${V56_CURRENT_LOW_CHECKPOINT:-/scratch/lj2634/runs/go2_native_low_lateral_deadzone_repair_v1/native_low_lateral_deadzone_repair_16543750_1_symmetric_lateral_10_20260829_012702/evidence/training/model_final.pt}
V14_SOURCE_ROOT=$SOURCE/vendor/v14
CURRENT_SUPPORT_ROOT=$SOURCE/vendor/current_qualified

PARENT_RUN_LABEL=${V56_PARENT_V55_RUN_LABEL:-}
PARENT_RESULT_SHA256=${V56_PARENT_V55_RESULT_SHA256:-}
PARENT_TEACHER_FIXED_SUMMARY_SHA256=${V56_PARENT_V55_TEACHER_FIXED_SUMMARY_SHA256:-}
PARENT_TEACHER_FIXED_ROWS_SHA256=${V56_PARENT_V55_TEACHER_FIXED_ROWS_SHA256:-}
PARENT_TRANSACTION_STATUS_SHA256=${V56_PARENT_V55_TRANSACTION_STATUS_SHA256:-}
PARENT_TASK_MANIFEST_SHA256=${V56_PARENT_V55_TASK_MANIFEST_SHA256:-}
BOUND_SOURCE_MANIFEST_SHA256=${V56_BOUND_SOURCE_MANIFEST_SHA256:-}
SUBMITTER_FULL_STATIC_SUITE=${V56_SUBMITTER_FULL_STATIC_SUITE:-}
SUBMITTER_RSL_RL_SMOKE=${V56_SUBMITTER_RSL_RL_SMOKE:-}

CONTRACT=$SOURCE/SPARSE_HIGH_TASK_PROGRESS_CANARY_V1_CONTRACT.json
PARENT_VERIFY=$SOURCE/validate_parent_v55.py
LIVE_TASK_VERIFY=$SOURCE/validate_live_task_package.py
CONTACT_RESET_PROBE_CLI=$SOURCE/probe_contact_reset_lifecycle.py
EVALUATE_CLI=$SOURCE/evaluate_task_progress_policy.py
TRAIN_CLI=$SOURCE/train_task_progress_canary.py
FINALIZE_CLI=$SOURCE/finalize_task_progress_canary.py
CHECKER=$SOURCE/check_sparse_high_task_progress_canary.sh
CHECKPOINT_VERIFY=$SOURCE/validate_checkpoint_payload.py
SOURCE_MANIFEST=$SOURCE/SPARSE_HIGH_TASK_PROGRESS_CANARY_V1_SOURCE.sha256
SOURCE_MANIFEST_VALIDATOR=$SOURCE/validate_source_manifest.py
SOURCE_MANIFEST_RECEIPT=$SOURCE.source-manifest.sha256
PARENT_SOURCE=$PROJECT/experiments/55_sparse_high_offline_bc_warm_start_v1
PARENT_CHECKER=$PARENT_SOURCE/check_sparse_high_offline_bc_warm_start.sh
PARENT_PY_VERIFY=$PARENT_SOURCE/check_sparse_high_offline_bc_warm_start.py
PARENT_SOURCE_MANIFEST=$PARENT_SOURCE/SPARSE_HIGH_OFFLINE_BC_WARM_START_V1_SOURCE.sha256
PARENT_SOURCE_VALIDATOR=$PARENT_SOURCE/validate_source_manifest.py
PARENT_SOURCE_RECEIPT=$PARENT_SOURCE.source-manifest.sha256

CURRENT_LOW_BYTES=4584447
CURRENT_LOW_SHA256=f46d3b760c43664350aac8099ae5bbd32bc1207f97c8814f5d21c9b531c3d96c
PARENT_CHECKER_BYTES=4696
PARENT_CHECKER_SHA256=5423b5130a45632223addc78a557237917eb255746deb6b15d8239c22f4ae6c1
PARENT_PY_VERIFY_BYTES=110489
PARENT_PY_VERIFY_SHA256=5c63a0f85f86d0103c38fae0d968a05a5f8ac969a3f8e21e756c296a4dd8ca86
PARENT_SOURCE_MANIFEST_BYTES=4348
PARENT_SOURCE_MANIFEST_SHA256=a16b3cfe5e723369010e4a4ba2b5d0583a6acb5a969003fac2f5656a46ea0a84
PARENT_SOURCE_VALIDATOR_BYTES=5715
PARENT_SOURCE_VALIDATOR_SHA256=a726b537b736b4253ec2995def96d45f0c058270a5dde8015c04e88fc76bac76

fail() {
  FAILURE_REASON=$1
  echo "ERROR=$FAILURE_REASON" >&2
  exit 2
}

exact_regular_file() {
  local path=$1 expected_bytes=$2 expected_sha=$3 label=$4 actual_bytes actual_sha
  [[ -f "$path" && ! -L "$path" ]] || fail "${label}_MISSING_OR_NOT_REGULAR"
  [[ "$(stat -Lc '%h' -- "$path")" == 1 ]] || fail "${label}_HARDLINK_COUNT_CHANGED"
  actual_bytes=$(stat -Lc '%s' -- "$path")
  [[ "$actual_bytes" == "$expected_bytes" ]] || fail "${label}_BYTES_CHANGED"
  actual_sha=$(sha256sum -- "$path" | awk '{print $1}')
  [[ "$actual_sha" == "$expected_sha" ]] || fail "${label}_SHA256_CHANGED"
}

[[ -x "$PY" ]] || fail CLUSTER_PYTHON_INVALID
mkdir -p -- "$RUN_ROOT"
RUN_ROOT=$(cd -- "$RUN_ROOT" && pwd -P)
STAMP=$(date -u +%Y%m%dT%H%M%S%NZ)
RUN=$RUN_ROOT/sparse_high_task_progress_canary_v1_${SLURM_JOB_ID:?}_${STAMP}
mkdir -- "$RUN"
EVIDENCE=$RUN/evidence
LOGS=$RUN/logs
RUNTIME=$RUN/runtime
INITIAL=$EVIDENCE/initial
TRAINING=$EVIDENCE/training
FINAL=$EVIDENCE/final
RESULT=$EVIDENCE/result
CONTACT_RESET_PROBE=$EVIDENCE/contact_reset_probe
TASK_RUNTIME_SOURCE=$RUN/source/runtime_project/source
TASK_PACKAGE_SNAPSHOT=$TASK_RUNTIME_SOURCE/isaac_go2_low_level_rl
TASK_PACKAGE_MANIFEST=$RUN/source/TASK_PACKAGE_RUNTIME.sha256
mkdir -p -- "$EVIDENCE" "$LOGS" "$CONTACT_RESET_PROBE" "$INITIAL" "$TRAINING" "$FINAL" "$RESULT" \
  "$RUNTIME"/{cache,config,data,tmp,matplotlib,omni,torch,torch-extensions,cuda,warp,pycache} \
  "$TASK_PACKAGE_SNAPSHOT"

ACTIVE_PHASE=wrapper_setup
FINALIZED=0
PHASE_FAILURE_RECORDED=0
durable_line() {
  local path=$1 value=$2
  "$PY" -B -I -S -c 'import os,pathlib,sys
path=pathlib.Path(sys.argv[1]); payload=(sys.argv[2]+"\n").encode(); tmp=path.with_name(path.name+f".tmp.{os.getpid()}")
fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
try:
 offset=0
 while offset<len(payload): offset+=os.write(fd,payload[offset:])
 os.fsync(fd)
finally: os.close(fd)
os.replace(tmp,path); dfd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
try: os.fsync(dfd)
finally: os.close(dfd)
' "$path" "$value"
}

durable_sync() {
  local path=$1
  "$PY" -B -I -S -c 'import os,pathlib,sys
path=pathlib.Path(sys.argv[1]); fd=os.open(path,os.O_RDONLY)
try: os.fsync(fd)
finally: os.close(fd)
dfd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
try: os.fsync(dfd)
finally: os.close(dfd)
' "$path"
}

durable_line "$RUN_ROOT/latest_run_${SLURM_JOB_ID}.txt" "$RUN"

phase() {
  local name=$1 state=$2 rc=${3:-}
  ACTIVE_PHASE=$name
  "$PY" -B -I -S -c 'import os,pathlib,sys
path=pathlib.Path(sys.argv[1]); payload=("\t".join(sys.argv[2:])+"\n").encode()
fd=os.open(path,os.O_WRONLY|os.O_APPEND)
try:
 offset=0
 while offset<len(payload): offset+=os.write(fd,payload[offset:])
 os.fsync(fd)
finally: os.close(fd)
dfd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
try: os.fsync(dfd)
finally: os.close(dfd)
' "$EVIDENCE/phase_events.tsv" "$(date --iso-8601=seconds)" "$name" "$state" "$rc"
  if [[ "$state" == FAIL ]]; then
    PHASE_FAILURE_RECORDED=1
  else
    PHASE_FAILURE_RECORDED=0
  fi
}

write_transaction_status() {
  local status=$1 process_rc=$2 failure=$3
  "$PY" -B -I -S -c 'import json,os,pathlib,sys
path=pathlib.Path(sys.argv[1]); evidence=path.parent
status=sys.argv[2]; rc=int(sys.argv[3]); failure=sys.argv[4]; phase=sys.argv[5]
def read(path): return json.loads(path.read_text()) if path.is_file() else None
result=read(evidence/"result/task_progress_canary_result.json")
support=read(evidence/"training/first_rollout_reward_support_summary.json")
support_pass=isinstance(support,dict) and support.get("gate_pass") is True
progress=read(evidence/"training/iteration_progress.json")
progress_valid=(
 isinstance(progress,dict)
 and set(progress)=={"status","completed_iterations","rollout_steps","optimizer_steps","last_iteration"}
 and type(progress.get("completed_iterations")) is int
 and 1<=progress["completed_iterations"]<=12
 and progress.get("rollout_steps")==progress["completed_iterations"]*200
 and progress.get("optimizer_steps")==progress["completed_iterations"]*20
 and progress.get("status")==("COMPLETE" if progress["completed_iterations"]==12 else "ACTIVE")
 and isinstance(progress.get("last_iteration"),dict)
)
progress_optimizer_steps=progress["optimizer_steps"] if progress_valid else 0
value={
 "schema":"sparse_high_task_progress_canary_v1/transaction_status/v1",
 "status":status,"process_rc":rc,
 "failed_phase":None if status=="COMPLETE" else phase,
 "failure":None if status=="COMPLETE" else failure,
 "parent_v55_run_label":os.environ.get("V56_PARENT_V55_RUN_LABEL", ""),
 "decision":None if result is None else result.get("decision"),
 "initial_evaluation_executed":(evidence/"initial/initial_evaluation_summary.json").is_file(),
 "reward_support_executed":support is not None,
 "reward_support_gate_pass":support_pass,
 "first_update_executed":(evidence/"training/first_update_summary.json").is_file(),
 "training_executed":(evidence/"training/training_summary.json").is_file(),
 "final_evaluation_executed":(evidence/"final/final_evaluation_summary.json").is_file(),
 "ppo_optimizer_steps":progress_optimizer_steps if result is None else result.get("ppo_optimizer_steps"),
 "long_training_authorized":False,
}
payload=(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
tmp=path.with_name(path.name+".tmp"); fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
try:
 offset=0
 while offset<len(payload): offset+=os.write(fd,payload[offset:])
 os.fsync(fd)
finally: os.close(fd)
os.replace(tmp,path); dfd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
try: os.fsync(dfd)
finally: os.close(dfd)
' "$EVIDENCE/transaction_status.json" "$status" "$process_rc" "$failure" "$ACTIVE_PHASE"
}

on_exit() {
  local rc=$?
  local reason
  trap - EXIT
  if [[ "$FINALIZED" != 1 ]]; then
    [[ "$rc" -ne 0 ]] || rc=97
    reason=${FAILURE_REASON:-transaction failed before trusted completion}
    if [[ "$PHASE_FAILURE_RECORDED" != 1 ]]; then
      phase "$ACTIVE_PHASE" FAIL "$rc" || true
    fi
    durable_line "$EVIDENCE/process_rc.txt" "$rc" || true
    durable_line "$EVIDENCE/SPARSE_HIGH_TASK_PROGRESS_CANARY_TRANSACTION_FINAL_GUARD" \
      "FAIL phase=$ACTIVE_PHASE process_rc=$rc" || true
    write_transaction_status ERROR "$rc" "$reason" || true
    echo 'SPARSE_HIGH_TASK_PROGRESS_CANARY_TRANSACTION=FAIL'
    echo "FAILED_PHASE=$ACTIVE_PHASE"
    echo "PROCESS_RC=$rc"
    echo 'LONG_TRAINING_AUTHORIZED=NO'
  fi
  exit "$rc"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

durable_line "$EVIDENCE/phase_events.tsv" $'timestamp\tphase\tstate\trc'
phase wrapper_static_preflight START
[[ "${V56_DELIVERY_SCHEMA_STATE:-}" == CORE_STABLE_RELEASE ]] \
  || fail V56_DELIVERY_SCHEMA_NOT_RELEASED
for path in "$CONTRACT" "$PARENT_VERIFY" "$LIVE_TASK_VERIFY" "$CONTACT_RESET_PROBE_CLI" "$EVALUATE_CLI" \
  "$TRAIN_CLI" "$FINALIZE_CLI" "$CHECKER" "$CHECKPOINT_VERIFY" \
  "$SOURCE_MANIFEST" "$SOURCE_MANIFEST_VALIDATOR" "$SOURCE_MANIFEST_RECEIPT"; do
  [[ -f "$path" && ! -L "$path" ]] || fail "REQUIRED_SOURCE_FILE_MISSING_$(basename -- "$path")"
done
[[ "$BOUND_SOURCE_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || fail V56_BOUND_SOURCE_MANIFEST_SHA256_INVALID
[[ "$SUBMITTER_FULL_STATIC_SUITE" == PASS ]] \
  || fail V56_SUBMITTER_FULL_STATIC_SUITE_NOT_PASS
[[ "$SUBMITTER_RSL_RL_SMOKE" == PASS ]] \
  || fail V56_SUBMITTER_RSL_RL_SMOKE_NOT_PASS
[[ "$(sha256sum -- "$SOURCE_MANIFEST" | awk '{print $1}')" == "$BOUND_SOURCE_MANIFEST_SHA256" ]] \
  || fail V56_SOURCE_MANIFEST_CHANGED_AFTER_SUBMIT
[[ -f "$GLU_ROOT/lib/libGLU.so.1" ]] || fail GLU_RUNTIME_MISSING
[[ -f "$ENV_ROOT/lib/libstdc++.so.6" ]] || fail LIBSTDCXX_RUNTIME_MISSING
for distribution in isaaclab isaaclab_tasks isaaclab_assets isaaclab_rl; do
  [[ -d "$ISAAC_LAB/source/$distribution" && ! -L "$ISAAC_LAB/source/$distribution" ]] \
    || fail "ISAAC_DISTRIBUTION_MISSING_${distribution}"
done
for support_root in "$V14_SOURCE_ROOT" "$CURRENT_SUPPORT_ROOT"; do
  [[ -d "$support_root" && ! -L "$support_root" ]] \
    || fail PACKAGE_LOCAL_SUPPORT_ROOT_MISSING_OR_UNSAFE
done
exact_regular_file "$CURRENT_LOW" "$CURRENT_LOW_BYTES" "$CURRENT_LOW_SHA256" CURRENT_QUALIFIED_LOW_CHECKPOINT

"$PY" -B -I -S "$PARENT_VERIFY" --contract "$CONTRACT" --pins-only \
  || fail PARENT_V55_RELEASE_PIN_GATE_FAILED
"$PY" -B -I -S -c 'import json,os,pathlib,sys
value=json.loads(pathlib.Path(sys.argv[1]).read_text()); parent=value.get("parent_v55",{})
expected={
 "run_label":os.environ.get("V56_PARENT_V55_RUN_LABEL", ""),
 "result_sha256":os.environ.get("V56_PARENT_V55_RESULT_SHA256", ""),
 "teacher_fixed_summary_sha256":os.environ.get("V56_PARENT_V55_TEACHER_FIXED_SUMMARY_SHA256", ""),
 "teacher_fixed_rows_sha256":os.environ.get("V56_PARENT_V55_TEACHER_FIXED_ROWS_SHA256", ""),
 "transaction_status_sha256":os.environ.get("V56_PARENT_V55_TRANSACTION_STATUS_SHA256", ""),
 "task_manifest_sha256":os.environ.get("V56_PARENT_V55_TASK_MANIFEST_SHA256", ""),
}
if any(parent.get(k)!=v for k,v in expected.items()): raise SystemExit("exported V55 identity differs")
' "$CONTRACT" || fail EXPORTED_PARENT_V55_BINDING_INVALID

export LD_LIBRARY_PATH="$GLU_ROOT/lib:$ENV_ROOT/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export LD_PRELOAD="$ENV_ROOT/lib/libstdc++.so.6${LD_PRELOAD:+:${LD_PRELOAD}}"
export XDG_CACHE_HOME=$RUNTIME/cache
export XDG_CONFIG_HOME=$RUNTIME/config
export XDG_DATA_HOME=$RUNTIME/data
export TMPDIR=$RUNTIME/tmp
export TMP=$RUNTIME/tmp
export TEMP=$RUNTIME/tmp
export MPLCONFIGDIR=$RUNTIME/matplotlib
export OMNI_USER_HOME=$RUNTIME/omni
export TORCH_HOME=$RUNTIME/torch
export TORCH_EXTENSIONS_DIR=$RUNTIME/torch-extensions
export CUDA_CACHE_PATH=$RUNTIME/cuda
export WARP_CACHE_PATH=$RUNTIME/warp
export PYTHONPYCACHEPREFIX=$RUNTIME/pycache
export QT_QPA_PLATFORM=offscreen
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export PYTHONOPTIMIZE=0
export HYDRA_FULL_ERROR=1
export GIT_PYTHON_REFRESH=quiet
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

"$PY" -B -I -S "$SOURCE_MANIFEST_VALIDATOR" \
  --root "$SOURCE" --manifest "$SOURCE_MANIFEST" --receipt "$SOURCE_MANIFEST_RECEIPT" \
  > "$LOGS/source_manifest_gate.json" || fail SOURCE_MANIFEST_GATE_FAILED
[[ "$(sha256sum -- "$SOURCE_MANIFEST" | awk '{print $1}')" == "$BOUND_SOURCE_MANIFEST_SHA256" ]] \
  || fail V56_SOURCE_MANIFEST_CHANGED_DURING_JOB_PREFLIGHT
bash -n "$SOURCE/run_sparse_high_task_progress_canary.sbatch" \
  "$SOURCE/submit_and_check_sparse_high_task_progress_canary.sh" "$CHECKER" \
  || fail SOURCE_SHELL_SYNTAX_GATE_FAILED
mapfile -t source_python_files < <(find "$SOURCE" -type f -name '*.py' -print | sort)
[[ "${#source_python_files[@]}" -gt 0 ]] || fail SOURCE_PYTHON_SET_EMPTY
"$PY" -B -I -S -c 'import ast,pathlib,sys
for name in sys.argv[1:]:
 path=pathlib.Path(name); ast.parse(path.read_bytes(),filename=str(path))
' "${source_python_files[@]}" || fail SOURCE_PYTHON_AST_GATE_FAILED
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$SOURCE" \
  "$PY" -B -m unittest -v tests.test_contact_reset_probe_delivery \
  > "$LOGS/contact_reset_probe_delivery_tests.log" 2>&1 \
  || { tail -n 160 "$LOGS/contact_reset_probe_delivery_tests.log" >&2; fail CONTACT_RESET_PROBE_DELIVERY_TEST_GATE_FAILED; }

env -u PYTHONHOME -u PYTHONSTARTUP -u PYTHONPATH \
  V56_V14_SOURCE_ROOT="$V14_SOURCE_ROOT" V56_CURRENT_SUPPORT_ROOT="$CURRENT_SUPPORT_ROOT" \
  "$PY" -B -I -S -c 'import pathlib,sys
sys.path.insert(0,sys.argv[1]); from source_bootstrap import configure_source_roots
value=configure_source_roots()
if pathlib.Path(value["roots"]["v14"]).resolve()!=pathlib.Path(sys.argv[2]).resolve(): raise SystemExit("V14 root escaped")
if pathlib.Path(value["roots"]["current"]).resolve()!=pathlib.Path(sys.argv[3]).resolve(): raise SystemExit("current root escaped")
' "$SOURCE" "$V14_SOURCE_ROOT" "$CURRENT_SUPPORT_ROOT" \
  || fail PACKAGE_LOCAL_SUPPORT_IDENTITY_GATE_FAILED

RSL_RL_IDENTITY=$("$PY" -B -c 'import importlib.metadata,importlib.util,pathlib
try: version=importlib.metadata.version("rsl-rl-lib")
except importlib.metadata.PackageNotFoundError: version=importlib.metadata.version("rsl-rl")
spec=importlib.util.find_spec("rsl_rl")
if spec is None or spec.origin is None: raise SystemExit("rsl_rl missing")
package=pathlib.Path(spec.origin).resolve(strict=True).parent
if package.name!="rsl_rl" or not (package/"__init__.py").is_file(): raise SystemExit("rsl_rl layout changed")
print(version+"\t"+str(package.parent))
') || fail RSL_RL_IDENTITY_UNRESOLVED
IFS=$'\t' read -r RSL_RL_VERSION RSL_RL_DISTRIBUTION_ROOT <<< "$RSL_RL_IDENTITY"
[[ "$RSL_RL_VERSION" == 3.1.2 && "$RSL_RL_DISTRIBUTION_ROOT" == /* \
   && -d "$RSL_RL_DISTRIBUTION_ROOT/rsl_rl" && ! -L "$RSL_RL_DISTRIBUTION_ROOT/rsl_rl" ]] \
  || fail RSL_RL_DISTRIBUTION_IDENTITY_CHANGED
set +e
rsl_smoke_output=$("$PY" -B -I "$SOURCE/tests/test_recurrent_hidden_guard.py" 2>&1)
rsl_smoke_rc=$?
set -e
printf '%s\n' "$rsl_smoke_output" > "$LOGS/rsl_rl_no_sim_lifecycle_smoke.log"
[[ "$rsl_smoke_rc" == 0 && "$rsl_smoke_output" == *OK* && "$rsl_smoke_output" != *skipped* ]] \
  || fail RSL_RL_3_1_2_NO_SIM_RECURRENT_LIFECYCLE_SMOKE_FAILED

exact_regular_file "$PARENT_CHECKER" "$PARENT_CHECKER_BYTES" "$PARENT_CHECKER_SHA256" PARENT_V55_CHECKER
exact_regular_file "$PARENT_PY_VERIFY" "$PARENT_PY_VERIFY_BYTES" "$PARENT_PY_VERIFY_SHA256" PARENT_V55_PYTHON_VERIFIER
exact_regular_file "$PARENT_SOURCE_MANIFEST" "$PARENT_SOURCE_MANIFEST_BYTES" "$PARENT_SOURCE_MANIFEST_SHA256" PARENT_V55_SOURCE_MANIFEST
exact_regular_file "$PARENT_SOURCE_VALIDATOR" "$PARENT_SOURCE_VALIDATOR_BYTES" "$PARENT_SOURCE_VALIDATOR_SHA256" PARENT_V55_SOURCE_VALIDATOR
[[ -f "$PARENT_SOURCE_RECEIPT" && ! -L "$PARENT_SOURCE_RECEIPT" \
   && "$(stat -Lc '%h' -- "$PARENT_SOURCE_RECEIPT")" == 1 ]] || fail PARENT_V55_SOURCE_RECEIPT_MISSING
"$PY" -B -I -S "$PARENT_SOURCE_VALIDATOR" \
  --root "$PARENT_SOURCE" --manifest "$PARENT_SOURCE_MANIFEST" --receipt "$PARENT_SOURCE_RECEIPT" \
  > "$LOGS/parent_source_manifest_gate.json" || fail PARENT_V55_SOURCE_GATE_FAILED
PARENT_RUN=/scratch/lj2634/runs/go2_sparse_high_offline_bc_warm_start_v1/$PARENT_RUN_LABEL
[[ -d "$PARENT_RUN" && ! -L "$PARENT_RUN" ]] || fail EXACT_PARENT_V55_RUN_MISSING
set +e
parent_check_output=$(V55_CHECK_PY="$PY" timeout --foreground --signal=TERM --kill-after=10s 180s \
  bash "$PARENT_CHECKER" "$PARENT_RUN" 2>&1)
parent_check_rc=$?
set -e
printf '%s\n' "$parent_check_output" > "$LOGS/parent_v55_checker.log"
[[ "$parent_check_rc" == 0 ]] || { tail -n 240 "$LOGS/parent_v55_checker.log" >&2; fail PARENT_V55_CHECKER_FAILED; }
for required_line in \
  'V55_CHECK=PASS' 'ENGINEERING_INTEGRITY=PASS' \
  'DECISION=V55_GEOMETRIC_TEACHER_FIXED_GATE_REJECTED_STOP_PRE_BC' \
  'TEACHER_FIXED_GATE_PASS=NO' 'PPO_OPTIMIZER_STEPS=0' \
  'TEACHER_FIXED_CONTACTS=91/288' 'TEACHER_FIXED_HARD_CONTACTS=9/36' \
  'TEACHER_FIXED_GOALS=0/288' 'TEACHER_FIXED_UNSAFE=1/288' \
  'LONG_TRAINING_AUTHORIZED=NO'; do
  [[ $'\n'"$parent_check_output"$'\n' == *$'\n'"$required_line"$'\n'* ]] \
    || fail PARENT_V55_CHECKER_OUTPUT_FACT_MISSING
done
"$PY" -B -I -S "$PARENT_VERIFY" --contract "$CONTRACT" --run "$PARENT_RUN" \
  > "$LOGS/parent_v55_own_gate.log" || fail PARENT_V55_OWN_FACT_AND_HASH_GATE_FAILED
"$PY" -B -I -S "$LIVE_TASK_VERIFY" \
  --distribution "$PROJECT/source/isaac_go2_low_level_rl" \
  --expected-sha256 "$PARENT_TASK_MANIFEST_SHA256" \
  > "$LOGS/live_task_package_gate.json" || fail LIVE_TASK_PACKAGE_DIFFERS_FROM_PARENT

{
  echo "SOURCE=$SOURCE"
  echo "PROJECT=$PROJECT"
  echo "PYTHON=$PY"
  echo "ISAAC_LAB=$ISAAC_LAB"
  echo "V56_ISAACLAB_DISTRIBUTION_ROOT=$ISAAC_LAB/source/isaaclab"
  echo "V56_ISAACLAB_TASKS_DISTRIBUTION_ROOT=$ISAAC_LAB/source/isaaclab_tasks"
  echo "V56_ISAACLAB_ASSETS_DISTRIBUTION_ROOT=$ISAAC_LAB/source/isaaclab_assets"
  echo "V56_ISAACLAB_RL_DISTRIBUTION_ROOT=$ISAAC_LAB/source/isaaclab_rl"
  echo "V56_RSL_RL_DISTRIBUTION_ROOT=$RSL_RL_DISTRIBUTION_ROOT"
  echo "GLU_ROOT=$GLU_ROOT"
  echo "CURRENT_LOW_SHA256=$CURRENT_LOW_SHA256"
  echo "BOUND_SOURCE_MANIFEST_SHA256=$BOUND_SOURCE_MANIFEST_SHA256"
  echo "PARENT_V55_RUN_LABEL=$PARENT_RUN_LABEL"
  echo "PARENT_V55_RESULT_SHA256=$PARENT_RESULT_SHA256"
  echo "PARENT_V55_TEACHER_FIXED_SUMMARY_SHA256=$PARENT_TEACHER_FIXED_SUMMARY_SHA256"
  echo "PARENT_V55_TEACHER_FIXED_ROWS_SHA256=$PARENT_TEACHER_FIXED_ROWS_SHA256"
  echo "PARENT_V55_TRANSACTION_STATUS_SHA256=$PARENT_TRANSACTION_STATUS_SHA256"
  echo "PARENT_V55_TASK_MANIFEST_SHA256=$PARENT_TASK_MANIFEST_SHA256"
  echo "RSL_RL_VERSION=$RSL_RL_VERSION"
  echo "V56_SUBMITTER_FULL_STATIC_SUITE=$SUBMITTER_FULL_STATIC_SUITE"
  echo "V56_SUBMITTER_RSL_RL_SMOKE=$SUBMITTER_RSL_RL_SMOKE"
  echo 'EXTERNAL_HIGH_CHECKPOINT=NONE'
  echo 'V4D_WEIGHTS_LOADED=NO'
  echo 'TEACHER_LOADED=NO'
  echo 'LONG_TRAINING_AUTHORIZED=NO'
  echo 'RSL_RL_NO_SIM_RECURRENT_LIFECYCLE_SMOKE=PASS'
} > "$EVIDENCE/runtime_environment.txt"
durable_sync "$EVIDENCE/runtime_environment.txt"
phase wrapper_static_preflight COMPLETE 0

phase task_package_snapshot START
timeout --foreground --signal=TERM --kill-after=10s 90s \
  bash -c 'set -Eeuo pipefail
source_root=$1; snapshot_root=$2
tar --exclude="__pycache__" --exclude="*/__pycache__" --exclude="*.pyc" --exclude="*.pyo" \
 --exclude=".pytest_cache" --exclude="*/.pytest_cache" --exclude=".git" --exclude="*/.git" \
 -C "$source_root" -cf - . | tar -C "$snapshot_root" -xf -
' _ "$PROJECT/source/isaac_go2_low_level_rl" "$TASK_PACKAGE_SNAPSHOT" \
  > "$LOGS/task_package_snapshot_copy.log" 2>&1 || fail TASK_PACKAGE_SNAPSHOT_COPY_FAILED

"$PY" -B -I -S -c 'import hashlib,json,os,pathlib,stat,sys
root=pathlib.Path(sys.argv[1]).resolve(strict=True); distribution=(root/"isaac_go2_low_level_rl").resolve(strict=True)
manifest=pathlib.Path(sys.argv[2]); receipt=pathlib.Path(sys.argv[3]); expected=sys.argv[4]
if not ((distribution/"setup.py").is_file() and (distribution/"isaac_go2_low_level_rl/__init__.py").is_file()): raise SystemExit("task snapshot layout changed")
rows=[]
for path in distribution.rglob("*"):
 info=path.lstat(); relative=path.relative_to(root).as_posix()
 if stat.S_ISLNK(info.st_mode): raise SystemExit(f"task snapshot symlink: {relative}")
 if path.name in {"sitecustomize.py","usercustomize.py"}: raise SystemExit(f"startup hook: {relative}")
 if stat.S_ISDIR(info.st_mode):
  if path.name in {"__pycache__",".pytest_cache",".git"}: raise SystemExit(f"task cache: {relative}")
  continue
 if not stat.S_ISREG(info.st_mode) or path.suffix in {".pyc",".pyo"}: raise SystemExit(f"task special/bytecode: {relative}")
 rows.append((relative,hashlib.sha256(path.read_bytes()).hexdigest()))
rows.sort(); payload="".join(f"{digest}  {name}\n" for name,digest in rows).encode(); digest=hashlib.sha256(payload).hexdigest()
if not rows or digest!=expected: raise SystemExit(f"task snapshot differs: {digest}")
def atomic(path,data):
 tmp=path.with_name(path.name+".tmp"); fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 try:
  offset=0
  while offset<len(data): offset+=os.write(fd,data[offset:])
  os.fsync(fd)
 finally: os.close(fd)
 os.replace(tmp,path); dfd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
 try: os.fsync(dfd)
 finally: os.close(dfd)
atomic(manifest,payload)
report={"schema":"sparse_high_task_progress_canary_v1/task_snapshot/v1","status":"PASS","manifest_sha256":digest,"regular_file_count":len(rows),"symlink_count":0,"cache_or_bytecode_count":0,"failures":[]}
atomic(receipt,(json.dumps(report,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode())
' "$TASK_RUNTIME_SOURCE" "$TASK_PACKAGE_MANIFEST" \
  "$EVIDENCE/task_package_snapshot_receipt.json" "$PARENT_TASK_MANIFEST_SHA256" \
  > "$LOGS/task_package_snapshot_manifest.log" 2>&1 || fail TASK_PACKAGE_SNAPSHOT_MANIFEST_FAILED
(cd -- "$TASK_RUNTIME_SOURCE" && sha256sum -c --strict "$TASK_PACKAGE_MANIFEST" >/dev/null) \
  || fail TASK_PACKAGE_SNAPSHOT_HASH_GATE_FAILED
find "$TASK_PACKAGE_SNAPSHOT" -type d -exec chmod 0555 {} +
find "$TASK_PACKAGE_SNAPSHOT" -type f -exec chmod 0444 {} +
chmod 0444 "$TASK_PACKAGE_MANIFEST"
CONTROLLED_PYTHONPATH="$SOURCE:$TASK_PACKAGE_SNAPSHOT:$ISAAC_LAB/source/isaaclab:$ISAAC_LAB/source/isaaclab_tasks:$ISAAC_LAB/source/isaaclab_assets:$ISAAC_LAB/source/isaaclab_rl"
printf 'TASK_PACKAGE_SNAPSHOT=%s\nTASK_PACKAGE_MANIFEST_SHA256=%s\nPYTHONPATH=%s\n' \
  "$TASK_PACKAGE_SNAPSHOT" "$PARENT_TASK_MANIFEST_SHA256" "$CONTROLLED_PYTHONPATH" \
  >> "$EVIDENCE/runtime_environment.txt"
durable_sync "$EVIDENCE/runtime_environment.txt"
phase task_package_snapshot COMPLETE 0

run_isaac_child() {
  local timeout_value=$1 log_path=$2
  shift 2
  timeout --signal=TERM --kill-after=15s "$timeout_value" \
    env -u PYTHONHOME -u PYTHONSTARTUP -u PYTHONPATH \
    PYTHONPATH="$CONTROLLED_PYTHONPATH" \
    V56_TASK_PACKAGE_SNAPSHOT="$TASK_PACKAGE_SNAPSHOT" \
    V56_ISAACLAB_DISTRIBUTION_ROOT="$ISAAC_LAB/source/isaaclab" \
    V56_ISAACLAB_TASKS_DISTRIBUTION_ROOT="$ISAAC_LAB/source/isaaclab_tasks" \
    V56_ISAACLAB_ASSETS_DISTRIBUTION_ROOT="$ISAAC_LAB/source/isaaclab_assets" \
    V56_ISAACLAB_RL_DISTRIBUTION_ROOT="$ISAAC_LAB/source/isaaclab_rl" \
    V56_RSL_RL_DISTRIBUTION_ROOT="$RSL_RL_DISTRIBUTION_ROOT" \
    V56_V14_SOURCE_ROOT="$V14_SOURCE_ROOT" \
    V56_CURRENT_SUPPORT_ROOT="$CURRENT_SUPPORT_ROOT" \
    V56_PARENT_V55_RUN_LABEL="$PARENT_RUN_LABEL" \
    V56_PARENT_V55_RESULT_SHA256="$PARENT_RESULT_SHA256" \
    V56_PARENT_V55_TEACHER_FIXED_SUMMARY_SHA256="$PARENT_TEACHER_FIXED_SUMMARY_SHA256" \
    V56_PARENT_V55_TEACHER_FIXED_ROWS_SHA256="$PARENT_TEACHER_FIXED_ROWS_SHA256" \
    V56_PARENT_V55_TRANSACTION_STATUS_SHA256="$PARENT_TRANSACTION_STATUS_SHA256" \
    V56_PARENT_V55_TASK_MANIFEST_SHA256="$PARENT_TASK_MANIFEST_SHA256" \
    "$@" > "$log_path" 2>&1
}

protocol_python() {
  env -u PYTHONHOME -u PYTHONSTARTUP -u PYTHONPATH PYTHONPATH="$SOURCE" "$PY" -B -P "$@"
}

sha_file() {
  local path=$1
  [[ -f "$path" && ! -L "$path" ]] || fail "HASH_INPUT_MISSING_OR_UNSAFE_$(basename -- "$path")"
  sha256sum -- "$path" | awk '{print $1}'
}

guard_pass() {
  local path=$1 label=$2
  [[ -f "$path" && ! -L "$path" && "$(cat -- "$path")" == PASS ]] || fail "${label}_FINAL_GUARD_NOT_PASS"
}

checkpoint_receipt() {
  local summary=$1 key=$2 checkpoint=$3 kind=$4 iteration=$5 optimizer=$6 label=$7 values
  values=$("$PY" -B -I -S -c 'import hashlib,json,pathlib,sys
summary=json.loads(pathlib.Path(sys.argv[1]).read_text()); path=pathlib.Path(sys.argv[2]); item=summary.get(sys.argv[3],{})
if item.get("relative_path")!=path.name or item.get("bytes")!=path.stat().st_size: raise SystemExit("receipt path/bytes differ")
h=hashlib.sha256();
with path.open("rb") as stream:
 for block in iter(lambda:stream.read(1048576),b""): h.update(block)
if h.hexdigest()!=item.get("sha256"): raise SystemExit("checkpoint bytes differ")
sha=item.get("sha256"); state=item.get("policy_state_sha256")
if not all(isinstance(v,str) and len(v)==64 and set(v)<=set("0123456789abcdef") for v in (sha,state)): raise SystemExit("identity invalid")
print(sha,state)
' "$summary" "$checkpoint" "$key") || fail "${label}_RECEIPT_INVALID"
  local file_sha policy_sha extra
  read -r file_sha policy_sha extra <<< "$values"
  [[ -n "$file_sha" && -n "$policy_sha" && -z "${extra:-}" ]] || fail "${label}_RECEIPT_OUTPUT_INVALID"
  env -u PYTHONHOME -u PYTHONSTARTUP -u PYTHONPATH \
    "$PY" -B -I "$CHECKPOINT_VERIFY" --checkpoint "$checkpoint" \
    --kind "$kind" --iteration "$iteration" --sha256 "$file_sha" \
    --policy-state-sha256 "$policy_sha" --optimizer "$optimizer" \
    > "$LOGS/${label,,}_payload_gate.log" || fail "${label}_PAYLOAD_INVALID"
  printf '%s %s\n' "$file_sha" "$policy_sha"
}

require_absent() {
  local label=$1
  shift
  local path
  for path in "$@"; do
    [[ ! -e "$path" && ! -L "$path" ]] || fail "${label}_FORBIDDEN_ARTIFACT_$(basename -- "$path")"
  done
}

finish_transaction() {
  phase terminal_aggregate START
  local initial_sha support_sha
  initial_sha=$(sha_file "$INITIAL/initial_evaluation_summary.json")
  support_sha=$(sha_file "$TRAINING/first_rollout_reward_support_summary.json")
  local finalize_args=(
    --contract "$CONTRACT" --evidence-root "$EVIDENCE" --output-dir "$RESULT"
    --initial-evaluation "$INITIAL/initial_evaluation_summary.json"
    --initial-evaluation-sha256 "$initial_sha"
    --reward-support "$TRAINING/first_rollout_reward_support_summary.json"
    --reward-support-sha256 "$support_sha"
  )
  if [[ "$SUPPORT_GATE" == PASS ]]; then
    finalize_args+=(
      --first-update "$TRAINING/first_update_summary.json"
      --first-update-sha256 "$(sha_file "$TRAINING/first_update_summary.json")"
      --training "$TRAINING/training_summary.json"
      --training-sha256 "$(sha_file "$TRAINING/training_summary.json")"
      --final-evaluation "$FINAL/final_evaluation_summary.json"
      --final-evaluation-sha256 "$(sha_file "$FINAL/final_evaluation_summary.json")"
    )
  fi
  protocol_python "$FINALIZE_CLI" "${finalize_args[@]}" \
    > "$LOGS/finalization.log" 2>&1 || { tail -n 200 "$LOGS/finalization.log" >&2; fail FINALIZER_FAILED; }
  [[ -s "$RESULT/task_progress_canary_result.json" ]] || fail RESULT_JSON_MISSING
  guard_pass "$RESULT/V56_RESULT_FINAL_GUARD" RESULT
  phase terminal_aggregate COMPLETE 0

  phase independent_result_check START
  durable_line "$EVIDENCE/process_rc.txt" 0
  write_transaction_status COMPLETE 0 ''
  durable_line "$EVIDENCE/SPARSE_HIGH_TASK_PROGRESS_CANARY_TRANSACTION_FINAL_GUARD" PASS
  V56_CHECK_IN_JOB=1 V56_CHECK_PY="$PY" bash "$CHECKER" "$RUN" \
    > "$LOGS/in_job_checker.log" 2>&1 \
    || { tail -n 260 "$LOGS/in_job_checker.log" >&2; fail IN_JOB_CHECKER_FAILED; }
  phase independent_result_check COMPLETE 0
  write_transaction_status COMPLETE 0 ''
  FINALIZED=1
  echo 'SPARSE_HIGH_TASK_PROGRESS_CANARY_TRANSACTION=COMPLETE'
  echo "SPARSE_HIGH_TASK_PROGRESS_RUN=$RUN"
  echo 'LONG_TRAINING_AUTHORIZED=NO'
  cat "$LOGS/in_job_checker.log"
  exit 0
}

phase contact_reset_lifecycle_probe START
set +e
run_isaac_child 180s "$LOGS/contact_reset_lifecycle_probe.log" \
  "$PY" -B -P "$CONTACT_RESET_PROBE_CLI" --contract "$CONTRACT" \
  --output-dir "$CONTACT_RESET_PROBE" --task Isaac-Go2-Soccer-Milestone2-v0 \
  --headless --device cuda:0
CONTACT_RESET_PROBE_RC=$?
set -e
durable_line "$EVIDENCE/contact_reset_probe_child_rc.txt" "$CONTACT_RESET_PROBE_RC"
[[ "$CONTACT_RESET_PROBE_RC" == 0 ]] \
  || { phase contact_reset_lifecycle_probe FAIL "$CONTACT_RESET_PROBE_RC"; tail -n 200 "$LOGS/contact_reset_lifecycle_probe.log" >&2; fail "CONTACT_RESET_PROBE_CHILD_RC_${CONTACT_RESET_PROBE_RC}"; }
guard_pass "$CONTACT_RESET_PROBE/CONTACT_RESET_LIFECYCLE_PROBE_FINAL_GUARD" CONTACT_RESET_LIFECYCLE_PROBE
[[ -s "$CONTACT_RESET_PROBE/contact_reset_lifecycle_probe.json" ]] \
  || fail CONTACT_RESET_LIFECYCLE_PROBE_EVIDENCE_MISSING
phase contact_reset_lifecycle_probe COMPLETE 0

phase initial_fixed_evaluation START
set +e
run_isaac_child 35m "$LOGS/initial_evaluation.log" \
  "$PY" -B -P "$EVALUATE_CLI" --phase initial --contract "$CONTRACT" \
  --output-dir "$INITIAL" --task Isaac-Go2-Soccer-Milestone2-v0 --headless --device cuda:0
INITIAL_RC=$?
set -e
durable_line "$EVIDENCE/initial_child_rc.txt" "$INITIAL_RC"
[[ "$INITIAL_RC" == 0 ]] || { phase initial_fixed_evaluation FAIL "$INITIAL_RC"; tail -n 200 "$LOGS/initial_evaluation.log" >&2; fail "INITIAL_CHILD_RC_${INITIAL_RC}"; }
guard_pass "$INITIAL/INITIAL_EVALUATION_FINAL_GUARD" INITIAL_EVALUATION
[[ -s "$INITIAL/initial_evaluation_summary.json" && -s "$INITIAL/initial_episode_results.jsonl" && -s "$INITIAL/model_initial.pt" ]] \
  || fail INITIAL_TERMINAL_EVIDENCE_MISSING
read -r INITIAL_CHECKPOINT_SHA INITIAL_POLICY_SHA < <(
  checkpoint_receipt "$INITIAL/initial_evaluation_summary.json" initial_checkpoint \
    "$INITIAL/model_initial.pt" initial_before_any_act 0 absent INITIAL_CHECKPOINT
) || fail INITIAL_CHECKPOINT_IDENTITY_INVALID
phase initial_fixed_evaluation COMPLETE 0

phase bounded_task_progress_ppo START
set +e
run_isaac_child 215m "$LOGS/training.log" \
  "$PY" -B -P "$TRAIN_CLI" --contract "$CONTRACT" --output-dir "$TRAINING" \
  --initial-checkpoint "$INITIAL/model_initial.pt" \
  --initial-checkpoint-sha256 "$INITIAL_CHECKPOINT_SHA" \
  --expected-initial-policy-state-sha256 "$INITIAL_POLICY_SHA" \
  --task Isaac-Go2-Soccer-Milestone2-v0 --headless --device cuda:0
TRAINING_RC=$?
set -e
durable_line "$EVIDENCE/training_child_rc.txt" "$TRAINING_RC"
[[ "$TRAINING_RC" == 0 ]] || { phase bounded_task_progress_ppo FAIL "$TRAINING_RC"; tail -n 240 "$LOGS/training.log" >&2; fail "TRAINING_CHILD_RC_${TRAINING_RC}"; }
guard_pass "$TRAINING/FIRST_ROLLOUT_REWARD_SUPPORT_FINAL_GUARD" REWARD_SUPPORT
[[ -s "$TRAINING/first_rollout_transitions.jsonl" && -s "$TRAINING/first_rollout_reward_support_summary.json" ]] \
  || fail REWARD_SUPPORT_TERMINAL_EVIDENCE_MISSING
SUPPORT_GATE=$("$PY" -B -I -S -c 'import pathlib,sys
sys.path.insert(0,sys.argv[1]); from check_sparse_high_task_progress_canary import strict_json,validate_transition_file
summary=strict_json(pathlib.Path(sys.argv[2]),"reward support"); validate_transition_file(pathlib.Path(sys.argv[3]),summary)
print("PASS" if summary.get("gate_pass") is True else "REJECT")
' "$SOURCE" "$TRAINING/first_rollout_reward_support_summary.json" "$TRAINING/first_rollout_transitions.jsonl") \
  || fail INDEPENDENT_REWARD_SUPPORT_RECONSTRUCTION_FAILED
[[ "$SUPPORT_GATE" == PASS || "$SUPPORT_GATE" == REJECT ]] || fail REWARD_SUPPORT_GATE_TOKEN_INVALID

if [[ "$SUPPORT_GATE" == REJECT ]]; then
  guard_pass "$TRAINING/REWARD_SUPPORT_REJECTED_FINAL_GUARD" REWARD_SUPPORT_REJECTION
  require_absent REWARD_SUPPORT_REJECT \
    "$TRAINING/model_first_update.pt" "$TRAINING/first_update_summary.json" \
    "$TRAINING/FIRST_UPDATE_FINAL_GUARD" "$TRAINING/iteration_progress.json" \
    "$TRAINING/model_final.pt" "$TRAINING/training_summary.json" \
    "$TRAINING/TRAINING_FINAL_GUARD" "$EVIDENCE/final_child_rc.txt" \
    "$FINAL/final_evaluation_summary.json" "$FINAL/final_episode_results.jsonl" \
    "$FINAL/FINAL_EVALUATION_FINAL_GUARD"
  phase bounded_task_progress_ppo REJECT 0
  finish_transaction
fi

guard_pass "$TRAINING/FIRST_UPDATE_FINAL_GUARD" FIRST_UPDATE
guard_pass "$TRAINING/TRAINING_FINAL_GUARD" TRAINING
[[ -s "$TRAINING/first_update_summary.json" && -s "$TRAINING/model_first_update.pt" \
   && -s "$TRAINING/training_summary.json" && -s "$TRAINING/model_final.pt" \
   && -s "$TRAINING/iteration_progress.json" ]] || fail TRAINING_TERMINAL_EVIDENCE_MISSING
read -r FIRST_CHECKPOINT_SHA FIRST_POLICY_SHA < <(
  checkpoint_receipt "$TRAINING/first_update_summary.json" first_update_checkpoint \
    "$TRAINING/model_first_update.pt" first_update 1 present FIRST_UPDATE_CHECKPOINT
) || fail FIRST_UPDATE_CHECKPOINT_IDENTITY_INVALID
read -r FINAL_CHECKPOINT_SHA FINAL_POLICY_SHA < <(
  checkpoint_receipt "$TRAINING/training_summary.json" final_checkpoint \
    "$TRAINING/model_final.pt" bounded_training_final 12 present FINAL_CHECKPOINT
) || fail FINAL_CHECKPOINT_IDENTITY_INVALID
[[ "$INITIAL_POLICY_SHA" != "$FIRST_POLICY_SHA" && "$INITIAL_POLICY_SHA" != "$FINAL_POLICY_SHA" \
   && "$FIRST_POLICY_SHA" != "$FINAL_POLICY_SHA" ]] || fail ACTUAL_CHECKPOINT_POLICY_STATES_DID_NOT_CHANGE
phase bounded_task_progress_ppo COMPLETE 0

phase final_fixed_evaluation START
set +e
run_isaac_child 35m "$LOGS/final_evaluation.log" \
  "$PY" -B -P "$EVALUATE_CLI" --phase final --contract "$CONTRACT" \
  --output-dir "$FINAL" --checkpoint "$TRAINING/model_final.pt" \
  --checkpoint-sha256 "$FINAL_CHECKPOINT_SHA" \
  --task Isaac-Go2-Soccer-Milestone2-v0 --headless --device cuda:0
FINAL_RC=$?
set -e
durable_line "$EVIDENCE/final_child_rc.txt" "$FINAL_RC"
[[ "$FINAL_RC" == 0 ]] || { phase final_fixed_evaluation FAIL "$FINAL_RC"; tail -n 200 "$LOGS/final_evaluation.log" >&2; fail "FINAL_CHILD_RC_${FINAL_RC}"; }
guard_pass "$FINAL/FINAL_EVALUATION_FINAL_GUARD" FINAL_EVALUATION
[[ -s "$FINAL/final_evaluation_summary.json" && -s "$FINAL/final_episode_results.jsonl" ]] \
  || fail FINAL_TERMINAL_EVIDENCE_MISSING
phase final_fixed_evaluation COMPLETE 0
finish_transaction

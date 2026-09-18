#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo 'ERROR=V56_SUBMITTER_MUST_BE_EXECUTED_NOT_SOURCED' >&2
  return 2
fi
set -Eeuo pipefail
umask 077

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

CONTRACT=$SOURCE/SPARSE_HIGH_TASK_PROGRESS_CANARY_V1_CONTRACT.json
PARENT_VERIFY=$SOURCE/validate_parent_v55.py
LIVE_TASK_VERIFY=$SOURCE/validate_live_task_package.py
SBATCH=$SOURCE/run_sparse_high_task_progress_canary.sbatch
CHECKER=$SOURCE/check_sparse_high_task_progress_canary.sh
SOURCE_MANIFEST=$SOURCE/SPARSE_HIGH_TASK_PROGRESS_CANARY_V1_SOURCE.sha256
SOURCE_MANIFEST_VALIDATOR=$SOURCE/validate_source_manifest.py
SOURCE_MANIFEST_RECEIPT=$SOURCE.source-manifest.sha256
PARENT_SOURCE=$PROJECT/experiments/55_sparse_high_offline_bc_warm_start_v1
PARENT_CHECKER=$PARENT_SOURCE/check_sparse_high_offline_bc_warm_start.sh
PARENT_PY_VERIFY=$PARENT_SOURCE/check_sparse_high_offline_bc_warm_start.py
PARENT_SOURCE_MANIFEST=$PARENT_SOURCE/SPARSE_HIGH_OFFLINE_BC_WARM_START_V1_SOURCE.sha256
PARENT_SOURCE_VALIDATOR=$PARENT_SOURCE/validate_source_manifest.py
PARENT_SOURCE_RECEIPT=$PARENT_SOURCE.source-manifest.sha256

EXPECTED_PARENT_RUN_LABEL=sparse_high_offline_bc_warm_start_v1_17938567_20260918T045821635312091Z
EXPECTED_PARENT_RESULT_SHA256=30b1369ff05a084733d0278e7fb31dbae0261a243f9a678b3d595e8fe3f7b4e3
EXPECTED_PARENT_TEACHER_FIXED_SUMMARY_SHA256=00b54c399abb05cd34fc7f86a0f5caf62dc4dc3b6eb238aa80d8e0ca07444356
EXPECTED_PARENT_TEACHER_FIXED_ROWS_SHA256=248d7e158becb03ec8480cf687d5b9b368380b5c20bca86abfb4ae25fc15a496
EXPECTED_PARENT_TRANSACTION_STATUS_SHA256=09a842bc23895489113f104ec63b640087115619583a7cc08f1eea24680ebfcc
EXPECTED_PARENT_TASK_MANIFEST_SHA256=b3af08bf243fa793257ebf625012ef05a43eb38df8e533293be68dfda9b36e3f

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
  echo "ERROR=$1" >&2
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

[[ "${V56_PREFLIGHT_ONLY:-0}" == 0 || "${V56_PREFLIGHT_ONLY:-0}" == 1 ]] \
  || fail INVALID_V56_PREFLIGHT_ONLY
[[ -d "$SOURCE" && ! -L "$SOURCE" ]] || fail SOURCE_DIRECTORY_INVALID
[[ -x "$PY" ]] || fail CLUSTER_PYTHON_INVALID
for path in "$CONTRACT" "$PARENT_VERIFY" "$LIVE_TASK_VERIFY" "$SBATCH" "$CHECKER" \
  "$SOURCE_MANIFEST" "$SOURCE_MANIFEST_VALIDATOR" "$SOURCE_MANIFEST_RECEIPT"; do
  [[ -f "$path" && ! -L "$path" ]] || fail "REQUIRED_SOURCE_FILE_MISSING_$(basename -- "$path")"
done

# Release pins are the first semantic gate and run before parent traversal or
# any scheduler request.
pins_output=$("$PY" -B -I -S "$PARENT_VERIFY" --contract "$CONTRACT" --pins-only) \
  || fail V55_OWN_RELEASE_PINS_NOT_READY
printf '%s\n' "$pins_output"
PARENT_RUN_LABEL=$(sed -n 's/^V56_PARENT_V55_RUN_LABEL=//p' <<< "$pins_output")
PARENT_RESULT_SHA256=$(sed -n 's/^V56_PARENT_V55_RESULT_SHA256=//p' <<< "$pins_output")
PARENT_TEACHER_FIXED_SUMMARY_SHA256=$(sed -n 's/^V56_PARENT_V55_TEACHER_FIXED_SUMMARY_SHA256=//p' <<< "$pins_output")
PARENT_TEACHER_FIXED_ROWS_SHA256=$(sed -n 's/^V56_PARENT_V55_TEACHER_FIXED_ROWS_SHA256=//p' <<< "$pins_output")
PARENT_TRANSACTION_STATUS_SHA256=$(sed -n 's/^V56_PARENT_V55_TRANSACTION_STATUS_SHA256=//p' <<< "$pins_output")
PARENT_TASK_MANIFEST_SHA256=$(sed -n 's/^V56_PARENT_V55_TASK_MANIFEST_SHA256=//p' <<< "$pins_output")
for value in "$PARENT_RESULT_SHA256" "$PARENT_TEACHER_FIXED_SUMMARY_SHA256" \
  "$PARENT_TEACHER_FIXED_ROWS_SHA256" "$PARENT_TRANSACTION_STATUS_SHA256" \
  "$PARENT_TASK_MANIFEST_SHA256"; do
  [[ "$value" =~ ^[0-9a-f]{64}$ ]] || fail PARENT_V55_PIN_OUTPUT_INVALID
done
[[ "$PARENT_RUN_LABEL" == "$EXPECTED_PARENT_RUN_LABEL" \
   && "$PARENT_RESULT_SHA256" == "$EXPECTED_PARENT_RESULT_SHA256" \
   && "$PARENT_TEACHER_FIXED_SUMMARY_SHA256" == "$EXPECTED_PARENT_TEACHER_FIXED_SUMMARY_SHA256" \
   && "$PARENT_TEACHER_FIXED_ROWS_SHA256" == "$EXPECTED_PARENT_TEACHER_FIXED_ROWS_SHA256" \
   && "$PARENT_TRANSACTION_STATUS_SHA256" == "$EXPECTED_PARENT_TRANSACTION_STATUS_SHA256" \
   && "$PARENT_TASK_MANIFEST_SHA256" == "$EXPECTED_PARENT_TASK_MANIFEST_SHA256" ]] \
  || fail PARENT_V55_IDENTITY_DIFFERS_FROM_EXACT_RELEASE
PARENT_RUN=/scratch/lj2634/runs/go2_sparse_high_offline_bc_warm_start_v1/$PARENT_RUN_LABEL

"$PY" -B -I -S "$SOURCE_MANIFEST_VALIDATOR" \
  --root "$SOURCE" --manifest "$SOURCE_MANIFEST" --receipt "$SOURCE_MANIFEST_RECEIPT" \
  || fail V56_SOURCE_EXACT_MEMBERSHIP_GATE_FAILED
V56_BOUND_SOURCE_MANIFEST_SHA256=$(sha256sum -- "$SOURCE_MANIFEST" | awk '{print $1}')
[[ "$V56_BOUND_SOURCE_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]] || fail V56_SOURCE_MANIFEST_SHA256_UNREADABLE
for support_root in "$V14_SOURCE_ROOT" "$CURRENT_SUPPORT_ROOT"; do
  [[ -d "$support_root" && ! -L "$support_root" ]] || fail PACKAGE_LOCAL_SUPPORT_ROOT_MISSING_OR_UNSAFE
done
env -u PYTHONHOME -u PYTHONSTARTUP -u PYTHONPATH \
  V56_V14_SOURCE_ROOT="$V14_SOURCE_ROOT" V56_CURRENT_SUPPORT_ROOT="$CURRENT_SUPPORT_ROOT" \
  "$PY" -B -I -S -c 'import pathlib,sys
sys.path.insert(0,sys.argv[1]); from source_bootstrap import configure_source_roots
value=configure_source_roots()
if pathlib.Path(value["roots"]["v14"]).resolve()!=pathlib.Path(sys.argv[2]).resolve(): raise SystemExit("V14 root escaped")
if pathlib.Path(value["roots"]["current"]).resolve()!=pathlib.Path(sys.argv[3]).resolve(): raise SystemExit("current root escaped")
' "$SOURCE" "$V14_SOURCE_ROOT" "$CURRENT_SUPPORT_ROOT" \
  || fail PACKAGE_LOCAL_SUPPORT_IDENTITY_GATE_FAILED

[[ -d "$PROJECT/source/isaac_go2_low_level_rl" ]] || fail TASK_PACKAGE_ROOT_MISSING
[[ -f "$GLU_ROOT/lib/libGLU.so.1" ]] || fail GLU_RUNTIME_MISSING
[[ -f "$ENV_ROOT/lib/libstdc++.so.6" ]] || fail LIBSTDCXX_RUNTIME_MISSING
for distribution in isaaclab isaaclab_tasks isaaclab_assets isaaclab_rl; do
  [[ -d "$ISAAC_LAB/source/$distribution" && ! -L "$ISAAC_LAB/source/$distribution" ]] \
    || fail "ISAAC_DISTRIBUTION_MISSING_${distribution}"
done
exact_regular_file "$CURRENT_LOW" "$CURRENT_LOW_BYTES" "$CURRENT_LOW_SHA256" CURRENT_QUALIFIED_LOW_CHECKPOINT
exact_regular_file "$PARENT_CHECKER" "$PARENT_CHECKER_BYTES" "$PARENT_CHECKER_SHA256" PARENT_V55_CHECKER
exact_regular_file "$PARENT_PY_VERIFY" "$PARENT_PY_VERIFY_BYTES" "$PARENT_PY_VERIFY_SHA256" PARENT_V55_PYTHON_VERIFIER
exact_regular_file "$PARENT_SOURCE_MANIFEST" "$PARENT_SOURCE_MANIFEST_BYTES" "$PARENT_SOURCE_MANIFEST_SHA256" PARENT_V55_SOURCE_MANIFEST
exact_regular_file "$PARENT_SOURCE_VALIDATOR" "$PARENT_SOURCE_VALIDATOR_BYTES" "$PARENT_SOURCE_VALIDATOR_SHA256" PARENT_V55_SOURCE_VALIDATOR
[[ -f "$PARENT_SOURCE_RECEIPT" && ! -L "$PARENT_SOURCE_RECEIPT" \
   && "$(stat -Lc '%h' -- "$PARENT_SOURCE_RECEIPT")" == 1 ]] || fail PARENT_V55_SOURCE_RECEIPT_MISSING
"$PY" -B -I -S "$PARENT_SOURCE_VALIDATOR" \
  --root "$PARENT_SOURCE" --manifest "$PARENT_SOURCE_MANIFEST" --receipt "$PARENT_SOURCE_RECEIPT" \
  || fail PARENT_V55_SOURCE_EXACT_MEMBERSHIP_GATE_FAILED

bash -n "$0" "$SBATCH" "$CHECKER" || fail SHELL_SYNTAX_GATE_FAILED
mapfile -t python_sources < <(find "$SOURCE" -type f -name '*.py' -print | sort)
[[ "${#python_sources[@]}" -gt 0 ]] || fail PYTHON_SOURCE_SET_EMPTY
"$PY" -B -I -S -c 'import ast,pathlib,sys
for name in sys.argv[1:]:
 path=pathlib.Path(name); ast.parse(path.read_bytes(),filename=str(path))
' "${python_sources[@]}" || fail PYTHON_AST_GATE_FAILED
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$SOURCE" \
  "$PY" -B -m unittest discover -v -s "$SOURCE/tests" \
  || fail STATIC_TEST_GATE_FAILED
V56_SUBMITTER_FULL_STATIC_SUITE=PASS

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
printf '%s\n' "$rsl_smoke_output"
[[ "$rsl_smoke_rc" == 0 && "$rsl_smoke_output" == *OK* && "$rsl_smoke_output" != *skipped* ]] \
  || fail RSL_RL_3_1_2_NO_SIM_RECURRENT_LIFECYCLE_SMOKE_FAILED
V56_SUBMITTER_RSL_RL_SMOKE=PASS

[[ -d "$PARENT_RUN" && ! -L "$PARENT_RUN" ]] || fail EXACT_PARENT_V55_RUN_MISSING
set +e
parent_check_output=$(V55_CHECK_PY="$PY" timeout --foreground --signal=TERM --kill-after=10s 180s \
  bash "$PARENT_CHECKER" "$PARENT_RUN" 2>&1)
parent_check_rc=$?
set -e
[[ "$parent_check_rc" == 0 ]] || { printf '%s\n' "$parent_check_output" >&2; fail PARENT_V55_CHECKER_FAILED; }
printf '%s\n' "$parent_check_output"
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
  || fail PARENT_V55_OWN_FACT_AND_HASH_GATE_FAILED
"$PY" -B -I -S "$LIVE_TASK_VERIFY" \
  --distribution "$PROJECT/source/isaac_go2_low_level_rl" \
  --expected-sha256 "$PARENT_TASK_MANIFEST_SHA256" \
  || fail PRE_GPU_LIVE_TASK_PACKAGE_DIFFERS_FROM_EXACT_PARENT

echo 'V56_PRE_GPU_SOURCE_LAYOUT_GATE=PASS'
echo 'V56_PRE_GPU_STATIC_TEST_GATE=PASS'
echo 'V56_PRE_GPU_CHECKPOINT_IDENTITY_GATE=PASS'
echo 'V56_PRE_GPU_EXACT_PARENT_V55_GATE=PASS'
echo 'V56_PRE_GPU_LIVE_TASK_PACKAGE_MANIFEST_GATE=PASS'
echo 'V56_PRE_GPU_RSL_RL_3_1_2_NO_SIM_LIFECYCLE_GATE=PASS'
echo "V56_SUBMITTER_FULL_STATIC_SUITE=$V56_SUBMITTER_FULL_STATIC_SUITE"
echo "V56_SUBMITTER_RSL_RL_SMOKE=$V56_SUBMITTER_RSL_RL_SMOKE"
echo 'LONG_TRAINING_AUTHORIZED=NO'
if [[ "${V56_PREFLIGHT_ONLY:-0}" == 1 ]]; then
  echo 'V56_PREFLIGHT_MODE=OPTIONAL_STANDALONE_NOT_A_SUBMISSION_PREREQUISITE'
  echo 'V56_PREFLIGHT_ONLY=PASS'
  echo 'V56_PREFLIGHT_ONLY_NEXT_ACTION=STOP_NO_IMMEDIATE_FORMAL_SUBMISSION'
  exit 0
fi
echo 'V56_STANDARD_SUBMISSION_FLOW=ONE_FORMAL_COMMAND'
echo 'V56_IN_JOB_FULL_STATIC_SUITE=NOT_REPEATED'

mkdir -p -- "$RUN_ROOT"
set +e
job_output=$(sbatch --parsable --wait \
  --export=ALL,V56_DELIVERY_SCHEMA_STATE=CORE_STABLE_RELEASE,V56_SOURCE="$SOURCE",V56_BOUND_SOURCE_MANIFEST_SHA256="$V56_BOUND_SOURCE_MANIFEST_SHA256",V56_SUBMITTER_FULL_STATIC_SUITE="$V56_SUBMITTER_FULL_STATIC_SUITE",V56_SUBMITTER_RSL_RL_SMOKE="$V56_SUBMITTER_RSL_RL_SMOKE",V56_PROJECT="$PROJECT",V56_ENV_ROOT="$ENV_ROOT",V56_CLUSTER_PY="$PY",V56_ISAAC_LAB="$ISAAC_LAB",V56_GLU_ROOT="$GLU_ROOT",V56_RUN_ROOT="$RUN_ROOT",V56_CURRENT_LOW_CHECKPOINT="$CURRENT_LOW",V56_PARENT_V55_RUN_LABEL="$PARENT_RUN_LABEL",V56_PARENT_V55_RESULT_SHA256="$PARENT_RESULT_SHA256",V56_PARENT_V55_TEACHER_FIXED_SUMMARY_SHA256="$PARENT_TEACHER_FIXED_SUMMARY_SHA256",V56_PARENT_V55_TEACHER_FIXED_ROWS_SHA256="$PARENT_TEACHER_FIXED_ROWS_SHA256",V56_PARENT_V55_TRANSACTION_STATUS_SHA256="$PARENT_TRANSACTION_STATUS_SHA256",V56_PARENT_V55_TASK_MANIFEST_SHA256="$PARENT_TASK_MANIFEST_SHA256" \
  "$SBATCH" 2>&1)
sbatch_rc=$?
set -e
job_id=${job_output%%;*}
job_id=${job_id%%$'\n'*}
echo "SBATCH_JOB_ID=$job_id"
echo "SBATCH_WAIT_RC=$sbatch_rc"
if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
  echo "SBATCH_OUTPUT=$job_output" >&2
  fail SBATCH_JOB_ID_UNREADABLE
fi

POINTER=$RUN_ROOT/latest_run_${job_id}.txt
[[ -s "$POINTER" ]] || { echo "SBATCH_OUTPUT=$job_output" >&2; fail RUN_POINTER_MISSING; }
RUN=$(sed -n '1p' "$POINTER")
[[ "$RUN" == "$RUN_ROOT"/sparse_high_task_progress_canary_v1_"$job_id"_* \
   && -d "$RUN" && ! -L "$RUN" ]] || fail RUN_POINTER_INVALID

set +e
V56_CHECK_PY="$PY" timeout --foreground --signal=TERM --kill-after=10s 300s \
  bash "$CHECKER" "$RUN"
check_rc=$?
set -e
echo "V56_CHECK_RC=$check_rc"
echo "V56_RUN=$RUN"
[[ "$check_rc" == 0 ]] || exit "$check_rc"
exit "$sbatch_rc"

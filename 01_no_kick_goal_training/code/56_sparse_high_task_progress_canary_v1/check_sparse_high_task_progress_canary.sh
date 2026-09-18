#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo 'ERROR=V56_CHECKER_MUST_BE_EXECUTED_NOT_SOURCED' >&2
  return 2
fi
set -Eeuo pipefail

RUN=${1:-}
CHECK_PY=${V56_CHECK_PY:-${V56_CLUSTER_PY:-/scratch/lj2634/envs/isaaclab232_v2/bin/python}}
SOURCE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
VERIFY=$SOURCE_DIR/check_sparse_high_task_progress_canary.py
CHECKPOINT_VERIFY=$SOURCE_DIR/validate_checkpoint_payload.py
SOURCE_MANIFEST=$SOURCE_DIR/SPARSE_HIGH_TASK_PROGRESS_CANARY_V1_SOURCE.sha256
SOURCE_MANIFEST_VALIDATOR=$SOURCE_DIR/validate_source_manifest.py
SOURCE_MANIFEST_RECEIPT=$SOURCE_DIR.source-manifest.sha256

fail() {
  echo "V56_CHECK=FAIL $1" >&2
  exit 2
}

[[ -n "$RUN" && "$RUN" == /* && -d "$RUN" && ! -L "$RUN" ]] || fail RUN_DIRECTORY_INVALID
[[ -x "$CHECK_PY" ]] || fail CHECK_PYTHON_INVALID
for path in "$VERIFY" "$CHECKPOINT_VERIFY" "$SOURCE_MANIFEST" \
  "$SOURCE_MANIFEST_VALIDATOR" "$SOURCE_MANIFEST_RECEIPT"; do
  [[ -f "$path" && ! -L "$path" ]] || fail "SOURCE_INPUT_MISSING_OR_UNSAFE=$path"
done
"$CHECK_PY" -B -I -S "$SOURCE_MANIFEST_VALIDATOR" \
  --root "$SOURCE_DIR" --manifest "$SOURCE_MANIFEST" \
  --receipt "$SOURCE_MANIFEST_RECEIPT" \
  || fail SOURCE_EXACT_MEMBERSHIP_GATE_FAILED

RUN=$(cd -- "$RUN" && pwd -P)
EVIDENCE=$RUN/evidence
IN_JOB=${V56_CHECK_IN_JOB:-0}
[[ "$IN_JOB" == 0 || "$IN_JOB" == 1 ]] || fail INVALID_V56_CHECK_IN_JOB

if [[ -f "$EVIDENCE/transaction_status.json" && ! -L "$EVIDENCE/transaction_status.json" ]]; then
  transaction_state=$("$CHECK_PY" -B -I -S -c 'import json,pathlib,sys
value=json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
status=value.get("status") if isinstance(value,dict) else None
if status not in {"COMPLETE","ERROR"}: raise SystemExit("transaction status invalid")
print(status)
' "$EVIDENCE/transaction_status.json") || fail TRANSACTION_STATUS_DISPATCH_INVALID
  if [[ "$transaction_state" == ERROR ]]; then
    checker_args=("$RUN")
    [[ "$IN_JOB" == 1 ]] && checker_args+=(--in-job)
    "$CHECK_PY" -B -I -S "$VERIFY" "${checker_args[@]}"
    exit $?
  fi
fi

for path in "$EVIDENCE/runtime_environment.txt" "$EVIDENCE/phase_events.tsv" \
  "$EVIDENCE/task_package_snapshot_receipt.json" \
  "$RUN/source/TASK_PACKAGE_RUNTIME.sha256" \
  "$EVIDENCE/contact_reset_probe/contact_reset_lifecycle_probe.json" \
  "$EVIDENCE/contact_reset_probe/CONTACT_RESET_LIFECYCLE_PROBE_FINAL_GUARD" \
  "$EVIDENCE/contact_reset_probe_child_rc.txt" \
  "$EVIDENCE/initial/initial_evaluation_summary.json" \
  "$EVIDENCE/initial/initial_episode_results.jsonl" \
  "$EVIDENCE/initial/model_initial.pt" \
  "$EVIDENCE/initial/INITIAL_EVALUATION_FINAL_GUARD" \
  "$EVIDENCE/initial_child_rc.txt" \
  "$EVIDENCE/training/first_rollout_transitions.jsonl" \
  "$EVIDENCE/training/first_rollout_reward_support_summary.json" \
  "$EVIDENCE/training/FIRST_ROLLOUT_REWARD_SUPPORT_FINAL_GUARD" \
  "$EVIDENCE/training_child_rc.txt" \
  "$EVIDENCE/result/task_progress_canary_result.json" \
  "$EVIDENCE/result/V56_RESULT_FINAL_GUARD"; do
  [[ -f "$path" && ! -L "$path" ]] || fail "REQUIRED_FILE_MISSING_OR_UNSAFE=$path"
done
for path in "$RUN" "$EVIDENCE" "$EVIDENCE/contact_reset_probe" "$EVIDENCE/initial" "$EVIDENCE/training" \
  "$EVIDENCE/final" "$EVIDENCE/result" \
  "$RUN/source/runtime_project/source/isaac_go2_low_level_rl"; do
  [[ -d "$path" && ! -L "$path" ]] || fail "REQUIRED_DIRECTORY_MISSING_OR_UNSAFE=$path"
done

checkpoint_receipt() {
  local summary=$1 key=$2 checkpoint=$3 kind=$4 iteration=$5 optimizer=$6 label=$7 values
  values=$("$CHECK_PY" -B -I -S -c 'import json,pathlib,sys
value=json.loads(pathlib.Path(sys.argv[1]).read_text()); item=value.get(sys.argv[2])
if not isinstance(item,dict): raise SystemExit("checkpoint receipt missing")
sha=item.get("sha256"); state=item.get("policy_state_sha256")
if not all(isinstance(x,str) and len(x)==64 for x in (sha,state)): raise SystemExit("checkpoint receipt invalid")
print(sha,state)
' "$summary" "$key") || fail "${label}_RECEIPT_UNREADABLE"
  local checkpoint_sha policy_sha extra
  read -r checkpoint_sha policy_sha extra <<< "$values"
  [[ -n "$checkpoint_sha" && -n "$policy_sha" && -z "${extra:-}" ]] \
    || fail "${label}_RECEIPT_OUTPUT_INVALID"
  env -u PYTHONHOME -u PYTHONSTARTUP -u PYTHONPATH \
    "$CHECK_PY" -B -I "$CHECKPOINT_VERIFY" \
    --checkpoint "$checkpoint" --kind "$kind" --iteration "$iteration" \
    --sha256 "$checkpoint_sha" --policy-state-sha256 "$policy_sha" \
    --optimizer "$optimizer" || fail "${label}_PAYLOAD_INVALID"
}

checkpoint_receipt "$EVIDENCE/initial/initial_evaluation_summary.json" initial_checkpoint \
  "$EVIDENCE/initial/model_initial.pt" initial_before_any_act 0 absent INITIAL_CHECKPOINT
if [[ -e "$EVIDENCE/training/model_first_update.pt" || -L "$EVIDENCE/training/model_first_update.pt" ]]; then
  checkpoint_receipt "$EVIDENCE/training/first_update_summary.json" first_update_checkpoint \
    "$EVIDENCE/training/model_first_update.pt" first_update 1 present FIRST_UPDATE_CHECKPOINT
fi
if [[ -e "$EVIDENCE/training/model_final.pt" || -L "$EVIDENCE/training/model_final.pt" ]]; then
  checkpoint_receipt "$EVIDENCE/training/training_summary.json" final_checkpoint \
    "$EVIDENCE/training/model_final.pt" bounded_training_final 12 present FINAL_CHECKPOINT
fi

checker_args=("$RUN")
[[ "$IN_JOB" == 1 ]] && checker_args+=(--in-job)
"$CHECK_PY" -B -I -S "$VERIFY" "${checker_args[@]}"

job_id=$(basename -- "$RUN" | sed -n 's/^sparse_high_task_progress_canary_v1_\([0-9][0-9]*\)_.*/\1/p')
if [[ "$IN_JOB" != 1 && -n "$job_id" ]] && command -v sacct >/dev/null 2>&1; then
  accounting=$(sacct -X -n -P -j "$job_id" \
    --format=JobIDRaw,State,ExitCode,Elapsed,MaxRSS 2>/dev/null \
    | awk -F'|' -v id="$job_id" '$1==id {print; exit}')
  if [[ -n "$accounting" ]]; then
    IFS='|' read -r _ state exit_code elapsed max_rss <<< "$accounting"
    echo "SLURM_STATE=$state"
    echo "SLURM_EXIT_CODE=$exit_code"
    echo "SLURM_ELAPSED=$elapsed"
    echo "SLURM_MAX_RSS=$max_rss"
  else
    echo 'SLURM_ACCOUNTING=NOT_YET_AVAILABLE'
  fi
fi
echo "SPARSE_HIGH_TASK_PROGRESS_RUN=$RUN"

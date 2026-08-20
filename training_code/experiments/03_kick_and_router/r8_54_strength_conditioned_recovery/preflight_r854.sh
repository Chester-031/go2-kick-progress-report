#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT="/scratch/lj2634/Isaac-Go2-Low-Level-RL"; EXP="$PROJECT/experiments/r8_54_strength_conditioned_recovery"; PY="/scratch/lj2634/envs/isaaclab232_v2/bin/python"; FREEZE="/scratch/lj2634/runs/go2_soccer_kick_freeze/ballfree_r841_grace6_20260810_185244"
echo '===== R8.54 PREFLIGHT (NO GPU) ====='; bash -n "$EXP/run_r854_recovery.sbatch" "$EXP/check_r854_latest.sh"; "$PY" -m py_compile "$EXP/patch_r854_source.py"; echo SCRIPT_SYNTAX=OK
R853_RUN="$(tr -d '\r\n' < /scratch/lj2634/runs/go2_soccer_r8_53_phase_recovery64/latest_r853.txt)"; R851_RUN="$(tr -d '\r\n' < /scratch/lj2634/runs/go2_soccer_r8_51_kick64_true_contact/latest_r851.txt)"; R850_RUN="$(tr -d '\r\n' < "$PROJECT/experiments/r8_50_real_contact_truth/latest_r850.txt")"
SOURCE="$R853_RUN/train_r853_phase_recovery64.py"
for required in "$SOURCE" "$R853_RUN/output/recovery64_final.pt" "$R851_RUN/output/kick64_final.pt" "$R851_RUN/output/r851_summary.json" "$R850_RUN/body_names.json" "$FREEZE/artifacts/r821_lift_base.pt"; do [[ -s "$required" ]] || { echo "PREFLIGHT_FATAL missing: $required" >&2; exit 32; }; stat --printf='READY %n %s bytes\n' "$required"; done
TEMP_DIR="$(mktemp -d /scratch/lj2634/r854_preflight_XXXXXX)"; trap 'rm -rf -- "$TEMP_DIR"' EXIT; "$PY" "$EXP/patch_r854_source.py" --source "$SOURCE" --output "$TEMP_DIR/train_r854.py"; "$PY" -m py_compile "$TEMP_DIR/train_r854.py"
grep -q 'def _r854_condition' "$TEMP_DIR/train_r854.py"; grep -q 'R854_STRENGTH_RECOVERY_ONLY' "$TEMP_DIR/train_r854.py"; grep -q '^export R854_STRENGTH_RECOVERY_ONLY=1.*R851_BODY_JSON=' "$EXP/run_r854_recovery.sbatch"
echo "R853_RUN=$R853_RUN"; echo "SOURCE_SHA256=$(sha256sum "$SOURCE" | awk '{print $1}')"; echo PATCH_INJECTION=OK; echo BODY_JSON_ALIAS=OK; echo R854_PREFLIGHT_OK

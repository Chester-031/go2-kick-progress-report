#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT="/scratch/lj2634/Isaac-Go2-Low-Level-RL"
EXP="$PROJECT/experiments/r8_53_phase_recovery64"
PY="/scratch/lj2634/envs/isaaclab232_v2/bin/python"
FREEZE="/scratch/lj2634/runs/go2_soccer_kick_freeze/ballfree_r841_grace6_20260810_185244"

echo '===== R8.53 PREFLIGHT (NO GPU) ====='
bash -n "$EXP/run_r853_recovery64.sbatch"
bash -n "$EXP/check_r853_latest.sh"
"$PY" -m py_compile "$EXP/patch_r853_source.py"
echo 'SCRIPT_SYNTAX=OK'

R851_RUN="$(tr -d '\r\n' < /scratch/lj2634/runs/go2_soccer_r8_51_kick64_true_contact/latest_r851.txt)"
R850_RUN="$(tr -d '\r\n' < "$PROJECT/experiments/r8_50_real_contact_truth/latest_r850.txt")"
SOURCE="$R851_RUN/train_r851_kick64.py"

for required in \
  "$SOURCE" \
  "$R851_RUN/output/kick64_final.pt" \
  "$R851_RUN/output/r851_summary.json" \
  "$R850_RUN/body_names.json" \
  "$FREEZE/artifacts/r839_recovery_vector_u32.pt" \
  "$FREEZE/artifacts/r835_swing_stability_u32.pt" \
  "$FREEZE/artifacts/r821_lift_base.pt"; do
  if [[ ! -s "$required" ]]; then echo "PREFLIGHT_FATAL missing: $required" >&2; exit 32; fi
  stat --printf='READY %n %s bytes\n' "$required"
done

TEMP_DIR="$(mktemp -d /scratch/lj2634/r853_preflight_XXXXXX)"
trap 'rm -rf -- "$TEMP_DIR"' EXIT
"$PY" "$EXP/patch_r853_source.py" --source "$SOURCE" --output "$TEMP_DIR/train_r853_phase_recovery64.py"
"$PY" -m py_compile "$TEMP_DIR/train_r853_phase_recovery64.py"

grep -q 'class R853Recovery64ActorCritic' "$TEMP_DIR/train_r853_phase_recovery64.py"
grep -q 'R853_PHASE_RECOVERY64_ONLY' "$TEMP_DIR/train_r853_phase_recovery64.py"
grep -q 'R851_KICK64_ONLY.*R853_PHASE_RECOVERY64_ONLY' "$TEMP_DIR/train_r853_phase_recovery64.py"
grep -q '^export R851_BODY_JSON="\$BODY_JSON"$' "$EXP/run_r853_recovery64.sbatch"

echo "R851_RUN=$R851_RUN"
echo "R850_RUN=$R850_RUN"
echo "SOURCE_SHA256=$(sha256sum "$SOURCE" | awk '{print $1}')"
echo 'PATCH_INJECTION=OK'
echo 'BODY_JSON_ALIAS=OK'
echo 'R853_PREFLIGHT_OK'

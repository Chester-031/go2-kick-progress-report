#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT="/scratch/lj2634/Isaac-Go2-Low-Level-RL"
EXP="$PROJECT/experiments/r8_59_parent_freeze"
PYTHON_BIN="${ISAAC_PYTHON:-/scratch/lj2634/envs/isaaclab232_v2/bin/python}"

V4D="/scratch/lj2634/runs/go2_soccer_v4d_joint/v4d_staged_joint_20260803_223518/full"
BALLFREE="/scratch/lj2634/runs/go2_soccer_kick_freeze/ballfree_r841_grace6_20260810_185244"

echo '============================================================'
echo ' R8.59 PARENT PREFLIGHT (READ ONLY)'
echo '============================================================'
echo "TIME=$(date -Is)"
echo "PROJECT=$PROJECT"
echo "PYTHON_BIN=$PYTHON_BIN"

bash -n "$EXP/freeze_r859_parents.sh"
bash -n "$EXP/check_r859_freeze.sh"
"$PYTHON_BIN" - "$EXP/inspect_r859_contracts.py" <<'PY'
import pathlib,sys
p=pathlib.Path(sys.argv[1])
compile(p.read_text(encoding='utf-8'),str(p),'exec')
PY
echo 'SCRIPT_SYNTAX=OK'

for pointer in \
  /scratch/lj2634/runs/go2_soccer_r8_51_kick64_true_contact/latest_r851.txt \
  /scratch/lj2634/runs/go2_soccer_r8_55_touchdown_hold/latest_r855.txt \
  /scratch/lj2634/runs/go2_soccer_r8_58_targeted_hold/latest_r858.txt \
  "$PROJECT/experiments/r8_50_real_contact_truth/latest_r850.txt"
do
  [[ -s "$pointer" ]] || { echo "MISSING_POINTER=$pointer"; exit 30; }
  target="$(tr -d '\r\n' < "$pointer")"
  [[ -d "$target" ]] || { echo "BROKEN_POINTER=$pointer target=$target"; exit 31; }
  echo "POINTER_OK=$pointer -> $target"
done

echo '1a05964f6330b532fe944d24028c67b880664eee3218a984cf17db6bf8010397' "$V4D/logs/high_model_final.pt" | sha256sum -c -
echo '52acb55160f4d4db0ac5c3da41aadbc5383a90555c64ab41851ba7383989996a' "$V4D/logs/low_model_final.pt" | sha256sum -c -
(cd "$BALLFREE" && sha256sum -c SHA256SUMS)

echo 'MUTATION=NONE'
echo 'TRAINING=FALSE'
echo 'VERDICT=R859_PREFLIGHT_OK'

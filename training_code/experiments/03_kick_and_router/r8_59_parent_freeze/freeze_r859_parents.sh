#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT="/scratch/lj2634/Isaac-Go2-Low-Level-RL"
ROOT="/scratch/lj2634/runs/go2_soccer_r8_59_parent_freeze"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$ROOT/r859_parent_freeze_${STAMP}"
EXP="$PROJECT/experiments/r8_59_parent_freeze"
PYTHON_BIN="${ISAAC_PYTHON:-/scratch/lj2634/envs/isaaclab232_v2/bin/python}"

V4D="/scratch/lj2634/runs/go2_soccer_v4d_joint/v4d_staged_joint_20260803_223518/full"
V3="/scratch/lj2634/runs/go2_soccer_high_level/v3_zero_sum_from_v3a_20260730_233921/full/logs/model_final.pt"
R851="$(tr -d '\r\n' < /scratch/lj2634/runs/go2_soccer_r8_51_kick64_true_contact/latest_r851.txt)"
R855="$(tr -d '\r\n' < /scratch/lj2634/runs/go2_soccer_r8_55_touchdown_hold/latest_r855.txt)"
R858="$(tr -d '\r\n' < /scratch/lj2634/runs/go2_soccer_r8_58_targeted_hold/latest_r858.txt)"
R850="$(tr -d '\r\n' < "$PROJECT/experiments/r8_50_real_contact_truth/latest_r850.txt")"
BALLFREE="/scratch/lj2634/runs/go2_soccer_kick_freeze/ballfree_r841_grace6_20260810_185244"

declare -a REQUIRED=(
  "$V4D/logs/high_model_final.pt"
  "$V4D/logs/low_model_final.pt"
  "$V4D/logs/v4d_joint_contract.json"
  "$V4D/logs/v4d_joint_summary.json"
  "$V4D/train_go2_soccer_v4d_joint.snapshot.py"
  "$V3"
  "$R851/output/kick64_final.pt"
  "$R851/output/r851_summary.json"
  "$R851/contract.txt"
  "$R851/train_r851_kick64.py"
  "$R855/output/recovery64_final.pt"
  "$R855/output/r855_summary.json"
  "$R855/contract.txt"
  "$R855/train_r855_touchdown_hold.py"
  "$R858/output/r858_hold_winner.json"
  "$R858/output/r858_summary.json"
  "$R858/contract.txt"
  "$R858/train_r858_targeted.py"
  "$R850/body_names.json"
  "$R850/r850_real_contact_summary.json"
  "$BALLFREE/SHA256SUMS"
  "$BALLFREE/source_r841_grace6.py"
)
for f in "${REQUIRED[@]}"; do [[ -s "$f" ]] || { echo "R859_FATAL missing=$f" >&2; exit 20; }; done
[[ -x "$PYTHON_BIN" ]] || { echo "R859_FATAL python_not_executable=$PYTHON_BIN" >&2; exit 21; }

# Guard the historical V4D accepted outputs before copying.
echo '1a05964f6330b532fe944d24028c67b880664eee3218a984cf17db6bf8010397' "${V4D}/logs/high_model_final.pt" | sha256sum -c -
echo '52acb55160f4d4db0ac5c3da41aadbc5383a90555c64ab41851ba7383989996a' "${V4D}/logs/low_model_final.pt" | sha256sum -c -
(cd "$BALLFREE" && sha256sum -c SHA256SUMS)

mkdir -p "$OUT/v4d" "$OUT/kick/ballfree_artifacts" "$OUT/ancestry" "$OUT/source" "$OUT/contracts"
sha256sum "${REQUIRED[@]}" > "$OUT/PARENT_INPUT_SHA256SUMS"
cp -a "$V4D/logs/high_model_final.pt" "$OUT/v4d/"
cp -a "$V4D/logs/low_model_final.pt" "$OUT/v4d/"
cp -a "$V4D/logs/v4d_joint_contract.json" "$OUT/v4d/"
cp -a "$V4D/logs/v4d_joint_summary.json" "$OUT/v4d/"
cp -a "$V4D/train_go2_soccer_v4d_joint.snapshot.py" "$OUT/source/"
cp -a "$V3" "$OUT/ancestry/v3_model_final.pt"
cp -a "$R851/output/kick64_final.pt" "$OUT/kick/"
cp -a "$R851/output/r851_summary.json" "$OUT/kick/"
cp -a "$R851/contract.txt" "$OUT/contracts/r851_contract.txt"
cp -a "$R851/train_r851_kick64.py" "$OUT/source/"
cp -a "$R855/output/recovery64_final.pt" "$OUT/kick/"
cp -a "$R855/output/r855_summary.json" "$OUT/kick/"
cp -a "$R855/contract.txt" "$OUT/contracts/r855_contract.txt"
cp -a "$R855/train_r855_touchdown_hold.py" "$OUT/source/"
cp -a "$R858/output/r858_hold_winner.json" "$OUT/kick/"
cp -a "$R858/output/r858_summary.json" "$OUT/kick/"
cp -a "$R858/contract.txt" "$OUT/contracts/r858_contract.txt"
cp -a "$R858/train_r858_targeted.py" "$OUT/source/"
cp -a "$R850/body_names.json" "$OUT/kick/"
cp -a "$R850/r850_real_contact_summary.json" "$OUT/kick/"
cp -a "$BALLFREE"/artifacts/. "$OUT/kick/ballfree_artifacts/"
cp -a "$BALLFREE/source_r841_grace6.py" "$OUT/source/"
cp -a "$EXP/inspect_r859_contracts.py" "$OUT/source/"

{
  printf 'created=%s\nproject=%s\n' "$(date -Is)" "$PROJECT"
  printf 'v4d_parent=%s\nv3_ancestry=%s\n' "$V4D" "$V3"
  printf 'r851_parent=%s\nr855_parent=%s\nr858_parent=%s\nr850_contact_parent=%s\nballfree_parent=%s\n' "$R851" "$R855" "$R858" "$R850" "$BALLFREE"
  printf 'training=false\nparent_mutation=false\nfreeze_scope=v4d_high+v4d_low+r858_complete_kick_chain\n'
} > "$OUT/PARENT_PATHS.txt"

sha256sum -c "$OUT/PARENT_INPUT_SHA256SUMS"
"$PYTHON_BIN" "$OUT/source/inspect_r859_contracts.py" --root "$OUT"
{
  printf 'time=%s\n' "$(date -Is)"
  printf 'state=FROZEN\ntraining=false\nparent_mutation=false\n'
  printf 'v4d_hash_guard=passed\nballfree_hash_guard=passed\n'
  printf 'parent_recheck_after_copy=passed\ncontract_inspection=passed\n'
} > "$OUT/FREEZE_STATUS.txt"
(cd "$OUT" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
chmod -R a-w "$OUT"
mkdir -p "$ROOT"
printf '%s\n' "$OUT" > "$ROOT/latest_r859_freeze.txt"
printf '%s\n' "$OUT" > "$EXP/latest_r859_freeze.txt"
echo "R859_PARENT_FREEZE_OK"
echo "FREEZE=$OUT"

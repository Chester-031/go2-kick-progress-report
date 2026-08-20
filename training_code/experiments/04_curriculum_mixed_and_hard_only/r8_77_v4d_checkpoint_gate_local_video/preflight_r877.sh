#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT="/scratch/lj2634/Isaac-Go2-Low-Level-RL"
EXP="$PROJECT/experiments/r8_77_v4d_checkpoint_gate_local_video"
PY="${ISAAC_PYTHON:-/scratch/lj2634/envs/isaaclab232_v2/bin/python}"
for FILE in select_r877.py run_r877_checkpoint_gate.sbatch check_r877_latest.sh \
  patch_r876_eval_player.py build_r877_local_bundle.sh render_local.sh \
  check_local.sh EXPECTED_BEHAVIOR.txt; do
  [[ -s "$EXP/$FILE" ]] || { echo "R877_PREFLIGHT_FATAL=missing:$FILE"; exit 21; }
done
"$PY" -m py_compile "$EXP/select_r877.py" "$EXP/patch_r876_eval_player.py"
for FILE in run_r877_checkpoint_gate.sbatch check_r877_latest.sh \
  build_r877_local_bundle.sh render_local.sh check_local.sh; do
  bash -n "$EXP/$FILE"
done
grep -Fq 'training=false' "$EXP/run_r877_checkpoint_gate.sbatch"
grep -Fq 'one_selected_checkpoint_only' "$EXP/select_r877.py"
grep -Fq 'R876_EVAL_FULL_SUPPORT=1' "$EXP/render_local.sh"
echo "R877_PREFLIGHT_OK"

#!/usr/bin/env bash
#SBATCH --job-name=r6-envprobe
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/lj2634/runs/go2_soccer_r6_runtime_discovery/runtime_probe_20260806_113635/slurm-%j.out

set -uo pipefail

ROOT="/scratch/lj2634/runs/go2_soccer_r6_runtime_discovery/runtime_probe_20260806_113635"
REPORT="${ROOT}/compute_probe.txt"
exec > >(tee "${REPORT}") 2>&1

echo "========================================================================"
echo "R6 COMPUTE-NODE RUNTIME PROBE"
echo "========================================================================"
echo "HOST=$(hostname)"
echo "DATE=$(date -Is)"
echo "USER=${USER}"
echo "HOME=${HOME}"
echo "PATH=${PATH}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

echo
echo "===== A. Shell initialization candidates ====="
for file in \
  /etc/profile \
  /etc/bashrc \
  /etc/profile.d/modules.sh \
  /usr/share/Modules/init/bash \
  /share/apps/lmod/lmod/init/bash
do
  if [ -f "${file}" ]; then
    echo "FOUND ${file}"
  fi
done

# Initialize modules where available, without failing the probe.
if [ -f /etc/profile.d/modules.sh ]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
elif [ -f /usr/share/Modules/init/bash ]; then
  # shellcheck disable=SC1091
  source /usr/share/Modules/init/bash
elif [ -f /share/apps/lmod/lmod/init/bash ]; then
  # shellcheck disable=SC1091
  source /share/apps/lmod/lmod/init/bash
fi

echo
echo "===== B. Commands before module loading ====="
for cmd in python python3 conda mamba micromamba module apptainer singularity nvidia-smi; do
  printf '%-12s ' "${cmd}"
  command -v "${cmd}" 2>/dev/null || echo "NOT_FOUND"
done

nvidia-smi 2>/dev/null || true

echo
echo "===== C. Module inventory ====="
if type module >/dev/null 2>&1; then
  echo "--- module list ---"
  module list 2>&1 || true
  echo "--- relevant module avail/spider ---"
  timeout 30s bash -lc \
    'module -t avail 2>&1 | grep -Ei "conda|anaconda|miniconda|mamba|python|pytorch|torch|cuda|isaac|apptainer" | head -n 400' \
    || true
  timeout 30s bash -lc \
    'module spider 2>&1 | grep -Ei "conda|anaconda|miniconda|mamba|python|pytorch|torch|cuda|isaac|apptainer" | head -n 400' \
    || true
else
  echo "MODULE_COMMAND_NOT_AVAILABLE"
fi

echo
echo "===== D. Candidate runtime search ====="
CANDIDATE_FILE="${ROOT}/compute_python_candidates.txt"
: > "${CANDIDATE_FILE}"

add_candidate() {
  local p="${1:-}"
  [ -n "${p}" ] || return 0
  [ -x "${p}" ] || return 0
  grep -Fxq "${p}" "${CANDIDATE_FILE}" 2>/dev/null || echo "${p}" >> "${CANDIDATE_FILE}"
}

add_candidate "$(command -v python 2>/dev/null || true)"
add_candidate "$(command -v python3 2>/dev/null || true)"

for p in \
  "${HOME}/.conda/envs/isaaclab232/bin/python" \
  "${HOME}/miniconda3/envs/isaaclab232/bin/python" \
  "${HOME}/anaconda3/envs/isaaclab232/bin/python" \
  "/scratch/${USER}/.conda/envs/isaaclab232/bin/python" \
  "/scratch/${USER}/miniconda3/envs/isaaclab232/bin/python" \
  "/scratch/${USER}/anaconda3/envs/isaaclab232/bin/python" \
  "/scratch/${USER}/conda/envs/isaaclab232/bin/python" \
  "/scratch/${USER}/Isaac-Go2-Low-Level-RL/.venv/bin/python" \
  "/scratch/${USER}/Isaac-Go2-Low-Level-RL/venv/bin/python"
do
  add_candidate "${p}"
done

for base in "${HOME}" "/scratch/${USER}" /share/apps /opt; do
  [ -d "${base}" ] || continue
  timeout 25s find "${base}" \
    -maxdepth 9 \
    \( -type f -o -type l \) \
    \( \
      -path '*/envs/*/bin/python' \
      -o -path '*/envs/*/bin/python3' \
      -o -path '*/.venv/bin/python' \
      -o -path '*/venv/bin/python' \
    \) \
    -perm -u+x \
    2>/dev/null |
  head -n 500 |
  while IFS= read -r p; do
    add_candidate "${p}"
  done
done

sort -u "${CANDIDATE_FILE}" -o "${CANDIDATE_FILE}"
cat "${CANDIDATE_FILE}"

echo
echo "===== E. Candidate import tests ====="
WORKING_FILE="${ROOT}/working_python.txt"
: > "${WORKING_FILE}"

while IFS= read -r p; do
  [ -n "${p}" ] || continue
  echo
  echo "--- testing ${p} ---"
  if timeout 30s "${p}" - <<'PY'
import importlib.util
import os
import sys

print("executable=", sys.executable)
print("version=", sys.version.replace("\n", " "))
for name in ("torch", "numpy", "isaaclab", "omni", "rsl_rl"):
    print(f"{name}_spec=", importlib.util.find_spec(name))
try:
    import torch
    print("torch_version=", torch.__version__)
    print("cuda_available=", torch.cuda.is_available())
    print("cuda_device_count=", torch.cuda.device_count())
except Exception as exc:
    print("torch_import_error=", type(exc).__name__, str(exc))
    raise
PY
  then
    echo "${p}" > "${WORKING_FILE}"
    echo "R6_WORKING_TORCH_PYTHON=${p}"
    break
  fi
done < "${CANDIDATE_FILE}"

echo
echo "===== F. Conda initialization and container search ====="
for base in "${HOME}" "/scratch/${USER}" /share/apps /opt; do
  [ -d "${base}" ] || continue
  timeout 20s find "${base}" \
    -maxdepth 9 \
    \( \
      -path '*/etc/profile.d/conda.sh' \
      -o -name 'conda' \
      -o -name 'mamba' \
      -o -name 'micromamba' \
      -o -name 'isaaclab.sh' \
      -o -name '*.sif' \
    \) \
    2>/dev/null |
  head -n 500
done

echo
if [ -s "${WORKING_FILE}" ]; then
  echo "R6_RUNTIME_PROBE_VERDICT=WORKING_TORCH_PYTHON_FOUND"
  echo "R6_RUNTIME_PYTHON=$(cat "${WORKING_FILE}")"
else
  echo "R6_RUNTIME_PROBE_VERDICT=NO_WORKING_TORCH_PYTHON_FOUND"
fi
echo "R6_RUNTIME_PROBE_COMPLETE"

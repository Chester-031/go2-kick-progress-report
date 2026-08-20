#!/usr/bin/env bash
set -uo pipefail

RUN_ROOT="/scratch/lj2634/runs/go2_soccer_r8_2_contact_load_probe"
ROOT="$(tr -d '\r\n' < "${RUN_ROOT}/latest_r8_2_contact_load_probe.txt")"
JOB="$(tr -d '\r\n' < "${ROOT}/probe/job_id.txt")"
LOG="${ROOT}/probe/worker.log"
JSON="${ROOT}/probe/r8_2_contact_load_probe.json"

echo "ROOT=${ROOT}"
echo "PROBE_JOB=${JOB}"

echo
echo "===== JOB ====="
sacct \
  -j "${JOB}" \
  --format=JobID,JobName%20,State,ExitCode,Elapsed \
  2>/dev/null || true

echo
echo "===== SIGNAL ====="
grep -E \
'R8_2_CONTACT_SOURCE|R8_2_CONTACT_MAPPING|R8_2_CONTACT_PHASE|R8_2_CONTACT_SIGNAL_READY|R8_2_CONTACT_PROBE_COMPLETE|R8_2_FATAL|RUN_RC=' \
  "${LOG}" \
  2>/dev/null |
tail -n 30 || true

echo
echo "===== SHORT JSON ====="
/scratch/lj2634/envs/isaaclab232_v2/bin/python - \
"${JSON}" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("contact_signal_available=", data.get("contact_signal_available"))
print("contact_signal_plausible=", data.get("contact_signal_plausible"))
print("decision=", data.get("decision"))

for name in ("stance", "lift_onset", "lift_hold", "recovery", "final"):
    row = data.get("phase_stats", {}).get(name)
    if not row or not row.get("samples"):
        continue
    print(
        f"{name:10s} "
        f"FL={row['fl_abs_fz_mean_N']:.3f}N "
        f"SUP3={row['support_three_abs_fz_mean_N']:.3f}N "
        f"ratio={row['fl_vertical_load_ratio']:.4f} "
        f"contact2N={row['fl_contact_fraction_ge_2N']:.4f} "
        f"lift_mean={row['selected_foot_lift_mean_m']:.5f}m "
        f"lift_max={row['selected_foot_lift_max_m']:.5f}m"
    )
PY

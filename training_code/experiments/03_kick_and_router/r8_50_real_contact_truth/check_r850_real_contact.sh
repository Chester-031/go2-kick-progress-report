#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 JOB_ID" >&2
  exit 2
fi

JOB_ID="$1"
PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
EXP_DIR="$PROJECT_ROOT/experiments/r8_50_real_contact_truth"
LOG="$PROJECT_ROOT/slurm-r850-${JOB_ID}.out"

sacct -j "$JOB_ID" --format=JobID,State,ExitCode,Elapsed,MaxRSS -P

if [[ ! -f "$LOG" ]]; then
  echo "missing log: $LOG" >&2
  exit 3
fi

if rg -n -i 'traceback|runtimeerror|cuda.*error|physx.*error|segmentation|killed|out of memory|(^|[^[:alpha:]])(nan|inf)([^[:alpha:]]|$)' "$LOG"; then
  echo "fatal/error signature found in $LOG" >&2
  exit 4
fi

rg -n 'R850_BODY_DISCOVERY_OK|R850_PATCH_OK|R850_RESULT|R850_REAL_CONTACT_OK' "$LOG"

RUN_DIR="$PROJECT_ROOT/runs/r850_real_contact_${JOB_ID}"
SUMMARY="$RUN_DIR/r850_real_contact_summary.json"
test -s "$SUMMARY"

python - "$SUMMARY" <<'PY'
import json, sys
p = sys.argv[1]
d = json.load(open(p, encoding="utf-8"))
assert d["schema"] == "r850-real-contact-v1"
assert d["training"] is False
assert len(d["cases"]) == 25
assert d["ball_center_height_m"] >= d["ball_radius_m"]
assert d["controls"]["far_ball_pass"]
assert d["controls"]["overlap_pass"]
w = d["winner"]
print("winner", w["case_id"], "forward", w["forward_m"], "lateral", w["lateral_m"])
print("real_strike_rate", w["real_strike_rate"], "clean_rate", w["clean_rate"])
print("precontact", w["precontact"], "wrong_first", w["wrong_first"], "body_first", w["body_first"])
PY

cd "$RUN_DIR"
sha256sum -c SHA256SUMS
echo R850_CHECK_OK

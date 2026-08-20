# R8.58 targeted touchdown hold

This package keeps the validated true-contact, kick-strength, and high-strength hold chain frozen. It searches only the missing low/mid touchdown-retention controller using an expanded static target plus age ramp, promotes eight candidates to a larger semifinal, then requires both exact-geometry and 4 mm jitter confirmation.

Install from project root, run the preflight, submit, and use the checker without a job ID:

```bash
cp -a /path/to/r8_58_targeted_hold experiments/
bash experiments/r8_58_targeted_hold/preflight_r858.sh
sbatch experiments/r8_58_targeted_hold/run_r858_targeted.sbatch
bash experiments/r8_58_targeted_hold/check_r858_latest.sh
```

Do not build router integration unless the checker returns `COMPLETED_READY_TO_BUILD_VIDEO`. The accepted winner is then consumed by the proven R8.56 local RGB-array video route.

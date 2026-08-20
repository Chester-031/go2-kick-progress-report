# R8.50 — real-contact truth gate

This package is the next executable step after the accepted R8.42 ball-free
controller.  It does **not** train a new policy.  It establishes the physical
contact label that R8.43--R8.49 did not have, then produces a ranked geometry
set for R8.51.

## Why this gate is mandatory

The soccer ball radius is `0.11 m`, while the old diagnostic placement used a
ball-center height of approximately `ground + 0.055 m`.  That starts the ball
inside the floor.  Old `geom/causal/impact` metrics also inferred contact from
distance and ball motion.  They are useful hypotheses, not contact truth.

R8.50 fixes the center height to `ground + 0.115 m`, requires the ball to
settle, enables contact reporting on the ball, and reads a one-body ball sensor
filtered against every exact robot rigid-body name discovered at runtime.

## Execution

Run from the project root on the Isaac/Slurm cluster:

```bash
sbatch experiments/r8_50_real_contact_truth/run_r850_real_contact.sbatch
```

The worker:

1. verifies the frozen R8.42 hashes;
2. discovers the runtime robot rigid-body names;
3. patches the exact frozen `source_r841_grace6.py` into an R8.50 probe;
4. runs a 25-cell geometry grid in parallel across the vector environments;
5. writes `r850_real_contact_summary.json` and raw JSONL episode rows.

Check a job without relying on the worker log alone:

```bash
bash experiments/r8_50_real_contact_truth/check_r850_real_contact.sh JOB_ID
```

## Acceptance and next step

R8.50 may nominate a geometry only when:

- the selected FL-foot pair has a filtered force event;
- no robot/ball contact occurred before activation;
- no wrong foot or non-foot body contacted first;
- the selected foot was moving toward the ball;
- the ball gained target-direction velocity in the contact step/window.

Threshold sensitivity is reported at `0.1`, `1.0`, and `5.0 N`; the operational
threshold defaults to `1.0 N`.  A far-ball negative control and a forced-overlap
positive control must both pass before the geometry ranking is trusted.

Only after this report passes should R8.51 train a small strike residual.  Do
not use the former 70-D isolated Kick observation as the final game interface:
the accepted Delta-Pose V4D low-level observation is 62-D.  Final integration
should deliberately expand it (recommended: 4-skill one-hot plus explicit
strength) and keep the high-level reward exactly `+1 / -1 / 0`.


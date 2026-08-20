# Go2 Soccer training-code workspace

This directory is the source snapshot through V1.8, prepared for continued Codex work.

## Read this first

- `current_v1_8_forensic/train_r898.py` is preserved for diagnosis and history.
- V1.8 route-first/hold behavior was diagnosed as a deadlock. Do not continue training it unchanged.
- The next clean branch should train dynamic behind-ball staging locomotion before reconnecting push/kick.
- V1.8 SHA256: `bfc4295212d216a8ab80351170d32c859168528ed95b9696886ec94274ece63d`

## Directory map

- `src/`: project package and soccer environment code used at runtime.
- `experiments/01_initial_v4d/`: original V4D baseline.
- `experiments/02_delta_pose_high_low/`: Delta-Pose and high/low lineage.
- `experiments/03_kick_and_router/`: Kick, contact, recovery and router lineage.
- `experiments/04_curriculum_mixed_and_hard_only/`: curriculum, mixed continuation and R8.98 repair code.
- `current_v1_8_forensic/`: current local V1.8 plus local patch backups.
- `submitted_jobs/`: exact submitted cluster scripts that lived outside the checkout.
- `provenance/`: source archive manifest and selected run-side snapshots.
- `vendor_refs/`: small Isaac Lab API references; not a vendored full installation.

Model checkpoints, videos and giant logs are intentionally excluded from Git.

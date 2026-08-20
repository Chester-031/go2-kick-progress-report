# R8.51 Kick64 true-contact training

R8.51 continues from the accepted R8.42 controller, the R8.48 strike head,
the R8.49 recovery vector, and the successful R8.50 true-contact geometry.

It trains only a bounded 64-D Kick residual adapter.  The frozen controller
still owns Load/Lift/Swing/Recovery.  The adapter input is built from the
historical 70-D isolated observation as:

```text
64D = old[0:63] + old[strength=69]
```

This is the integration-ready context: native 48-D proprioception, Delta-Pose
context, remaining phase, four-skill one-hot, ball XY, teacher flag zero, and
Kick strength.  The obsolete phase/leg/direction extras are removed; phase is
represented by remaining time, the leg is fixed FL for this first skill, and
direction is represented by ball XY.

The R8.48 first-layer columns are migrated into the 64-D head.  Shared layers
and outputs are copied exactly.  R8.49 Recovery remains frozen.  PPO rewards
use the R8.50 filtered SoccerBall-to-body contact matrix, not proximity.

Strength curriculum:

- updates 1--4: strength 1.0;
- updates 5--8: strengths 0.75, 1.0, 1.25;
- updates 9+: continuous [0.70, 1.30].

Run from the project root:

```bash
sbatch experiments/r8_51_kick64_true_contact/run_r851_kick64.sbatch
```

The worker automatically resolves the latest completed R8.50 run and the
existing R8.48/R8.49 checkpoints.  Every run writes `latest_r851.txt` as soon
as it starts, so the checker never needs a Job ID:

```bash
bash experiments/r8_51_kick64_true_contact/check_r851_latest.sh
```


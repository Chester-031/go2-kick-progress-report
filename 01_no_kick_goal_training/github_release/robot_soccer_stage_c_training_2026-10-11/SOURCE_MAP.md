# Source map

This export is a manually curated explanatory projection. It is not a
byte-for-byte copy of the frozen trainer and must not be substituted for it.

| Projection | Frozen source concepts |
| --- | --- |
| `code/parameters.py` | continuation contract constants, trainer constants, and the declared optimizer-screen matrix |
| `code/reset_strategy.py` | `phased_reset_schedule.py`, restricted to the Stage C long-distance-weighted phase |
| `code/policy_architecture.py` | recurrent high-policy configuration, 70-D observation construction, and qualified command mapping |
| `code/task_logic.py` | task-progress reward geometry and the high-level environment's same-tick terminal logic |
| `code/training_outline.py` | `checkpoint220_resume.py` and the scientific core of `train_stage_a.py` |
| `code/analysis_axes.py` | completed-episode stratification, with orientation, robot–ball distance, and ball–goal distance treated as peer axes |

The Stage B reader-facing package was used only as a presentation reference:
short modules, grouped constants, explicit data structures, and minimal
comments around non-obvious invariants. No Stage B file is imported or
modified by this export.

Production-only source concerned with delivery, supervision, provenance,
telemetry persistence, terminal checking, and result publication is outside
the projection.

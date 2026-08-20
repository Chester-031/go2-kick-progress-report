# R8.63 full-support V4D candidate

This patch removes the progress-dependent difficulty reset and samples robot
and ball independently across the playable interior. It preserves 17D high,
62D Delta Pose low, 12D low action, the single low_env.step site, and the
sparse zero-sum reward contract. It does not train.

The next step is a frozen-checkpoint coverage scan. Only if the coverage scan
shows holes should this candidate be used for direct full-distribution PPO
fine-tuning.

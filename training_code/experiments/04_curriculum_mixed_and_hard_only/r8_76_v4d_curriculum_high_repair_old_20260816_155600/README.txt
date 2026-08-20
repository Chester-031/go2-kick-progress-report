R8.76 V4D curriculum-assisted high-level repair

Decision
  R8.74b high is rejected.  R8.76 warm-starts the accepted V4D high and loads
  the accepted V4D low exactly.  Only high is trainable; low remains frozen.

Reward
  High reward remains strictly sparse and zero-sum:
    goal +1
    own goal -1
    all other transitions 0
  No dense approach, wall, alignment or locomotion reward is added.

Curriculum (600 iterations)
  0-74    easy: ball nearby, robot mostly behind ball, small heading error
  75-199  intermediate: larger range, side offsets, wider heading error,
            with 25 percent easy retention
  200-449 hard mix: 10 percent easy, 30 percent intermediate,
            35 percent full support, 15 percent back-facing,
            10 percent symmetric left/right wall cases
  450-599 full support: robot and ball independently uniform over the playable
            interior and yaw uniform over [-pi, pi]

The training history must be reported as curriculum-assisted.  The final model
is intended for evaluation on stationary full-support resets without curriculum.

No video is produced during training.  Important checkpoint candidates are:
  high_model_199.pt  end of intermediate learning
  high_model_449_joint_finetune.pt  end of hard mix
  high_model_final.pt  end of full-support stabilization

Cluster usage
  cd /scratch/lj2634/Isaac-Go2-Low-Level-RL
  bash experiments/r8_76_v4d_curriculum_high_repair/preflight_r876.sh
  sbatch experiments/r8_76_v4d_curriculum_high_repair/run_r876_curriculum_high_repair.sbatch

Monitor
  bash experiments/r8_76_v4d_curriculum_high_repair/check_r876_latest.sh

Completion only means checkpoints are ready for metric evaluation.  Accept or
reject using the R8.75 full-support owner-attribution cases before rendering any
important checkpoint video through the already verified local-render chain.

The READY marker additionally requires every curriculum phase, including the
last stationary full-support phase, to have observed a positive goal event as
well as nonzero sparse reward.  A zero-goal phase is treated as failed learning,
not a valid run; own goals alone cannot satisfy this gate.

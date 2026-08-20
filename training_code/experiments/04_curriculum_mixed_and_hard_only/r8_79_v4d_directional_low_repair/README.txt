R8.79 directional low repair
============================

R8.78 was correctly rejected at 4/8 sustained primitives. It improved average
speed and reduced stall, but automatic soccer push classification converted
many lateral and turn training requests into forward push samples. The result
was a forward-biased low policy.

R8.79 continues from the completed R8.78 low checkpoint for 200 iterations. It
freezes the accepted high policy, disables automatic push only during primitive
repair, strengthens backward input, and adds signed per-axis reference-velocity
tracking plus wrong-direction penalties. It still produces no video.

Install and run:

  cd /scratch/lj2634/Isaac-Go2-Low-Level-RL
  if [[ -e experiments/r8_79_v4d_directional_low_repair ]]; then
    mv experiments/r8_79_v4d_directional_low_repair \
       experiments/r8_79_v4d_directional_low_repair_old_$(date +%Y%m%d_%H%M%S)
  fi
  tar --no-same-owner -xzf /scratch/lj2634/r8_79_v4d_directional_low_repair_20260816.tgz -C experiments
  bash experiments/r8_79_v4d_directional_low_repair/preflight_r879.sh
  sbatch experiments/r8_79_v4d_directional_low_repair/run_r879_directional_low_repair.sbatch

Monitor:

  bash experiments/r8_79_v4d_directional_low_repair/check_r879_latest.sh

Only VERDICT=R879_DIRECTIONAL_LOW_READY_FOR_PAIRED_GATE permits the next paired
full-support metric test. Any other result rejects the checkpoint.

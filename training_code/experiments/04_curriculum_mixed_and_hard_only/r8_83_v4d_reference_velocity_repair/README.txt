R8.81 continues the rejected 5/8 R8.79 low checkpoint for 250 iterations.
It fixes two objective loopholes proven by R8.80: yaw stalls were absent from
the stall penalty, and total speed allowed backward failure to hide in lateral
drift. It applies signed per-axis minimum-response penalties, yaw-aware stall,
stronger turn targets, and refreshes the low observation immediately after each
new Delta Pose target. High remains frozen and no video is produced.

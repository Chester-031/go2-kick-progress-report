# R8.62 V4D audit

This is the first step toward full-support, no-progressive-curriculum V4D.
It copies the R8.59 frozen V4D source and reports the exact reset, reward,
observation, row-storage, curriculum, and low-level step sites. It does not
train or patch anything.

The next run is a frozen-checkpoint coverage scan over robot pose, ball pose,
relative distance, and yaw. Only after that scan can we decide whether a
full-support direct fine-tune is necessary.

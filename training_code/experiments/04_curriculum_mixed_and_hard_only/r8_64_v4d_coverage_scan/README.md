# R8.64 V4D full-support coverage gate

This is a no-training reset-distribution gate for the R8.63 source candidate.
It verifies independent robot/ball support, full yaw support, all four relative
position quadrants, minimum initial separation, the fixed curriculum slot, and
the single low-level step boundary. It does not claim that the frozen policy
already scores from every pose; that is the next Isaac behavior scan.

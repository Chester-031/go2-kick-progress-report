R8.80 is a short, non-training attribution run. It evaluates the exact R8.79
low checkpoint on the same isolated primitives twice: deterministic mean action
and stochastic sampled action. It also records action standard deviation and
verifies zero parameter change. The result chooses between low-noise mean-policy
fine-tuning and a mechanical/objective persistence repair. No video is made.

# R8.52 gain × recovery-delay causal scan

R8.51 established clean true contact but showed inverse strength response and
zero Recovery.  R8.52 is a short, non-training scan that holds the accepted
geometry and 64-D policy fixed while varying only:

- Kick residual gain: `0.0, 0.5, 1.0, 1.5, 2.0`;
- Recovery handoff delay after real strike: `0, 2, 4, 6` control steps.

All 20 cases run concurrently across 128 environments.  It records filtered
contact, target-direction speed/displacement, and every R8.29 Recovery subgate.
This identifies whether strength can be controlled by action gain and which
Recovery gate actually blocks before another PPO run is launched.


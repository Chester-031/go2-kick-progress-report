# R8.4 load-preserving progressive lift PPO

R8.3 passed all three real-load gates:
- unload_24pct: 53.23% recent episode success
- unload_16pct: 89.18%
- unload_10pct: 52.29%

R8.4 continues from the latest R8.3 final checkpoint.

Curriculum:
`unload_10pct -> lift_1cm -> lift_2cm -> lift_3cm -> lift_4cm`

For every lift stage:
1. the selected front foot must first unload in the early pre-lift window;
2. the unload must be sustained for multiple consecutive steps;
3. lift reward is gated by the real contact-load state;
4. stage success requires both preserved unload and geometric lift;
5. forward/lateral foot drift is discouraged because swing is not trained yet.

The R8.3 video failure was a TensorDict inference API bug. R8.4 fixes the
evaluator by passing the full TensorDict into `act_inference`.

Unchanged interfaces:
- Actor observation: 70D, teacher-free
- low action: 12D joint-position residual
- high-level Router: unchanged
- high-level sparse soccer reward: unchanged and unused here
- ball-contact training: still not started

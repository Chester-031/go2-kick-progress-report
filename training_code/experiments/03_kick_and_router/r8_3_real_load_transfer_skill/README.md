# R8.3 real-load transfer PPO pilot

R8.2 validated the real contact-force source:
`scene["contact_forces"].data.net_forces_w`.

Measured R8.1 behavior:
- stance FL load ratio: ~0.2936
- intended lift onset: ~0.4114
- lift hold: ~0.4419
- final: ~0.4553

So R8.1 transferred weight onto FL instead of unloading it.

R8.3 isolates the missing prerequisite and trains only:
`unload_24pct -> unload_16pct -> unload_10pct`.

Each stage requires the FL load ratio to remain below the threshold for
six consecutive steps, while the other three feet carry at least 80 N total
and the body remains stable.

No lift or swing target is trained in R8.3. If all three load gates pass,
the next experiment should resume from this checkpoint and add 1 cm lift.

Interfaces remain unchanged:
- Actor observation: 70D, teacher-free
- low action: 12D joint-position residuals
- high-level Router: unchanged
- high-level sparse soccer reward: unchanged and unused here
- ball-contact training: not started

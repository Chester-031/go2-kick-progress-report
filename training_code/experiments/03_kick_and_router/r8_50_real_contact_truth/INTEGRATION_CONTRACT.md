# Kick → Delta-Pose/PPO integration contract

## Frozen facts

| Layer | Current accepted contract | Do not change during R8.50/R8.51 |
|---|---|---|
| High observation | 17-D | Yes |
| High action | 3-D continuous Delta-Pose intent | Yes, until Kick is physically accepted |
| High reward | opponent goal `+1`, own goal `-1`, otherwise `0` | Always |
| Low observation | 62-D teacher-free (`48 + 14`) | Yes, during isolated Kick proof |
| Low action | 12-D joint-position residual | Yes |
| Existing skills | stand / locomotion / push | Yes |

The isolated R8 controller uses a historical 70-D input.  Its weights are not
shape-compatible with the V4D game policy and must not be copied into the 62-D
deployment path.

## Integration after physical Kick acceptance

Recommended high action proposal: five continuous values:

1. Delta x
2. Delta y
3. Delta yaw
4. Kick gate / skill intent
5. Kick strength

Recommended low context change: expand the skill one-hot from 3 to 4 and add
one normalized strength scalar.  This changes the current 62-D input to 64-D.
Initialize the new network by copying all shared input columns and zero-init the
two new columns, so stand/locomotion/push behaviour is preserved at step zero.

Kick eligibility should be a deterministic safety mask based on ball reach,
alignment, support state, and cooldown.  PPO chooses intent and strength; the
mask prevents physically invalid execution.  The mask is not a dense reward.

## Required sequence

1. R8.50: actual-contact truth and geometry ranking (`training=false`).
2. R8.51: train only a bounded strike/strength residual on the top geometries.
3. R8.52: validate clean contact, target-direction ball velocity, recovery, and
   at least three separable strength bands.
4. V4K integration: add Kick/strength interface and distill/consolidate into the
   teacher-free low policy.
5. Joint high/low PPO: retain separated optimizers/rewards and the strict sparse
   high-level goal result.

Never repeat distance-only “contact” validation or regard ball motion from an
invalid initial pose as proof of a kick.

# R8.59 pre-integration source audit

## Selected no-Kick execution parent

Use the accepted V4D full-run pair together:

- High: `v4d_staged_joint_20260803_223518/full/logs/high_model_final.pt`
- Low: `v4d_staged_joint_20260803_223518/full/logs/low_model_final.pt`
- Exact source: `full/train_go2_soccer_v4d_joint.snapshot.py`

The older V3 `model_final.pt` is ancestry evidence for strict zero-sum behavior, not the execution high policy for integration. V4D's high policy was warm-started from V3 and then jointly adapted with the V4D low policy.

Known accepted V4D hashes:

- High: `1a05964f6330b532fe944d24028c67b880664eee3218a984cf17db6bf8010397`
- Low: `52acb55160f4d4db0ac5c3da41aadbc5383a90555c64ab41851ba7383989996a`

## V4D exact interfaces

- High observation: 17D soccer state.
- High action: 3D continuous intent, clipped to `[-1, 1]`.
- High cadence: one action is held for 5 low-level control steps.
- Router skills: stand=0, locomotion=1, push=2.
- Low observation: 62D = native 48D + direct-pose/context 14D.
- Native command slots `[9:12]` are explicitly zeroed before actor inference.
- Context `[48:62]`: normalized body-frame pose error x/y, sin/cos yaw error, terminal vx/vy/wz, remaining fraction, three-skill one-hot, ball x/y, zero teacher flag.
- Low action: 12D clipped joint-position residual passed directly to `low_env.step()`.
- High reward support: goal `+1`, own goal `-1`, all other `0`; it is not mixed with low execution reward.
- Actor/critic observation normalization: disabled for the high policy; the V4D low path retains its checkpoint/runtime policy behavior and direct 62D contract.

High action to Delta-Pose mapping outside push mode:

- `dx = 0.32*a0` for forward, `0.16*a0` for reverse.
- `dy = 0.18*a1`.
- `dyaw = 0.30*a2`.
- terminal velocity = `[0.12*a0, 0.08*a1, 0.15*a2]`.

Push is still selected by the deterministic V4D router when the robot is close to and behind the ball and aligned with goal. Stand is selected when `max(abs(action)) < 0.05`.

## Selected Kick execution parent

The R8.58 controller is not one standalone checkpoint. Its frozen execution chain is:

1. Accepted ball-free components: R821 lift policy, R823 lift residual, R815 swing policy, R835 swing stability, R827 recovery head, R839 recovery vector.
2. R8.51 `kick64_final.pt`: 64D to 12D true-contact strike residual.
3. R8.55 `recovery64_final.pt`: 64D to 12D post-contact recovery.
4. R8.58 `r858_hold_winner.json`: three strength-specific touchdown-hold controllers.
5. R8.50 body/contact map and R8.51 geometry contract.

The freeze also retains the exact generated R8.51, R8.55, and R8.58 source files, their contract files, the R8.50 contact summary, and the accepted ball-free source. This is provenance evidence; integration still starts only from copies in a new directory.

R8.58 strength anchors:

- 0.75: thigh `+0.32`, calf `+0.12`, ramp `0.35`.
- 1.00: thigh `+0.32`, calf `+0.16`, ramp `0.70`.
- 1.25: thigh `-0.04`, calf `+0.04`, ramp `0.00`.

## Critical integration boundary

V4D low and R8.58 both ultimately own a full 12D action tensor, but their input contracts differ:

- V4D low actor: 62D.
- Isolated kick wrapper state: 70D.
- R8.51 strike and R8.55 recovery actors: selected 64D from the 70D kick state.

Therefore integration must use exclusive action ownership immediately before the single underlying environment step:

- locomotion state: execute the V4D low 12D action;
- kick/recovery state: execute the complete R8.58 12D action;
- never add, blend, or average the two policy outputs;
- never load a 64D kick checkpoint into the 62D V4D actor.

The combined implementation must be derived from copies of the exact source snapshots inside the new integration directory. Parent runs and current project sources remain untouched.

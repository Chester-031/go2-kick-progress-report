# V56 sparse-high task-progress canary

V56 is one bounded, single-seed causal test of one change only: the fresh
10 Hz high policy is trained with same-transition task-path progress plus the
unchanged strict terminal reward.  The frozen 50 Hz low policy, observations,
3-D command interface, declared V14 XY/yaw reset distribution, terminal
semantics, PPO budget and fixed evaluation remain unchanged.  R2 restores the
ball's canonical reset height instead of inheriting terminal Z.  R3 changes
only the pre-training probe that tests that repair, and R4 corrects only that
probe's live body-index join; the production environment, contract, reward,
action, PPO and evaluator bytes remain unchanged.  This reward
is intentionally not claimed
to be policy-invariant PBRS, and its dense return is not task success evidence.

The release is pinned to the exact V55 OWN run
`sparse_high_offline_bc_warm_start_v1_17938567_20260918T045821635312091Z`.
Submission fails before requesting a GPU if the installed V56 source, exact
V55 evidence, live task package, qualified low checkpoint, Isaac roots, or
RSL-RL 3.1.2 recurrent lifecycle differs.

## One-job protocol

The formal submitter runs the complete exact-environment Python suite and the
non-skipping RSL recurrent smoke once on the login node before it requests a
GPU.  The Slurm job revalidates the bound source manifest, syntax, parent,
task-package and frozen-low identities, but does not repeat that complete
suite.  This keeps the queued transaction evidence-bound without paying for
the same full suite a second time.

The Slurm transaction performs, in order:

1. Exact source, parent, task-package, frozen-low and RSL identity checks.
2. A bounded actual-PhysX reset probe that uses an already-qualified calf
   contact fixture, exercises a selected-slot reset, and proves cached
   ball/robot sensor reset, canonical full ball state and an exactly-one-tick
   first-post-physics refresh before any evaluation, policy or optimizer is
   constructed.
3. A read-only 288-case initial fixed evaluation (36 named hard cases).
4. One 200-step × 128-env raw rollout, durably recording all 25,600 physical
   pre/reset-free-post transitions before returns or optimizer updates.
5. Independent reward/support reconstruction.  A scientific support miss is a
   successful RC=0 stop with zero optimizer steps and no learned checkpoint.
6. Only after support passes: 12 bounded PPO iterations (20 Adam steps each,
   240 total), then the same read-only fixed evaluation.
7. Hash-bound aggregation, independent stdlib evidence checking, checkpoint
   payload checking and durable inner/outer completion guards.

The checker recomputes terminal facts, reset continuity, float32 goal/staging
geometry, path progress, sparse and total rewards, both-side support, the
command hull/rate invariants, summary metrics, result decision and legal branch
layout.  Torch is used only by the separate checkpoint-payload validator.

## R1 engineering repair

The first Cluster transaction, job `17939578` / run
`sparse_high_task_progress_canary_v1_17939578_20260918T071005373934975Z`,
passed reward support and durably completed the first PPO update. At the first
configured 180-second periodic faulthandler dump it then terminated with native
SIGSEGV (child RC 139), with no Python exception and no terminal result. This
is engineering failure evidence, not a behavioral result, and its partial
checkpoint is not a continuation parent.

R1 removes only `faulthandler.dump_traceback_later(..., repeat=True)` from the
two Isaac child CLIs. Synchronous faulthandler support and all shell child
timeouts remain. The contract, reward, reset/terminal semantics, PPO budget and
checker are byte-unchanged; fixed cases and evaluation semantics are unchanged,
and the evaluator CLI's sole byte delta is the same timer removal. A fresh
transaction must avoid the exact 180-second/RC-139 signature and close through
an original V56 terminal branch; otherwise this hypothesis is rejected before
another repair.

## R2 atomic reset repair

R1 job `17940941` / run
`sparse_high_task_progress_canary_v1_17940941_20260918T111911340609851Z`
avoided the periodic-dump crash and durably completed 8 of 12 PPO iterations
and 160 of 240 optimizer steps.  During the ninth rollout it stopped in
`step -> _reset_idx` with `ContactSensor history did not reset`.  This is a
second engineering failure, not a task result, and none of its partial
checkpoints may be continued.

The sensor had in fact cleared its cached history.  IsaacLab reset also marks
the sensor outdated; the immediate check then read the lazy public `data`
property before another physics step, which refreshed from the previous
episode's still-current PhysX contact view and wrote that terminal contact back
into the cleared history.  The same reset path also inherited the old live
ball z instead of installing canonical reset height.  R2 treats both as one
atomic reset-boundary repair.  It changes no reward, observation, command,
terminal rule, reset x/y/yaw sample, PPO setting, budget, fixed case or
scientific decision.

Before initial evaluation, the same Slurm transaction now runs a short fresh
engineering child that first establishes real robot-ball contact and then
exercises the repaired selected-slot reset.  It must prove non-lazy cached
ball/base sensor clearing, unchanged non-selected slots, canonical ball z and
a clean first refresh after normal physics.  Any failure stops before fresh
high-policy or PPO optimizer construction.  Passing only authorizes the
unchanged fresh V56 transaction to begin; it is not behavioral evidence.

## R3 probe-only repair

R2 job `17943801` / run
`sparse_high_task_progress_canary_v1_17943801_20260918T121821021175688Z`
passed its source-manifest gate but stopped inside the contact-reset probe.  Its
base-center fixture produced only `0.006549851968884468 N` in the ball-side
filtered matrix and `0.005206768400967121 N` at the robot base body, so
`_establish_actual_contact` raised.  This happened before sentinel injection
and before `_reset_idx`; PPO optimizer steps remained zero.  The run therefore
rejects only that contact fixture.  It did not exercise, pass or falsify the R2
reset repair.

R3 changes only the probe, its evidence/checker and its pre-training transaction
gate.  It reuses the already-qualified calf-center fixture from job `16315309`:
on the unchanged exact 19-body soccer contact interface, all four calf bodies
previously produced `148--160 N` in their expected filter slots.  A selected
slot now counts as physically contacted only when the current
`robot_ball_contact.force_matrix_w` entry and the current
`contact_forces.net_forces_w` entry are both at least the unchanged `1 N`
threshold on the same tick and same calf body.  History maxima, different
ticks or bodies, ball displacement and millinewton noise cannot be combined
into a witness.  Ball placement and contact acquisition use only state writes
and raw physics; they do not call the learned low actor, `low_env.step` or the
action manager.  A collision sweep is not an alternate acceptance route.

The physical-contact lane resets the selected live slots immediately after the
same-frame witness, without replacing their physical force caches with
sentinels.  Before any public sensor-data read, it requires selected private
caches and lifecycle metadata to be reset, unselected slots to be bit-exact,
and the complete ball root state to equal its canonical default plus the new
sampled XY.  It then snapshots the private caches, queries the raw PhysX contact
views, and proves a second private snapshot is bit-exact.  The raw force values
are attribution only: PhysX may still expose the previous frame before another
physics step, so neither zero nor nonzero raw values decide PASS.

The first-refresh lane advances exactly one raw `0.005 s` physics tick before
the first public sensor read.  Both sensors must then be fresh and finite, and
every history entry after index zero must remain exactly zero.  This cannot be
replaced by the four-tick low-environment step: the three-frame robot history
and four-frame ball history can roll an illicit pre-physics sample completely
out by that boundary.  A normal frozen-low step is exercised separately as a
production-integration check, not as first-refresh evidence.  A separate
sentinel lane covers every cached field plus vectorized partial and full reset;
it cannot substitute for the real-contact lane.

The production `current_task_progress_env.py`, `CONTRACT.json`, task, reward,
observation, action, terminal, recurrent PPO, optimizer-budget and fixed
evaluator bytes remain unchanged.  Any R3 probe failure closes the same job
before evaluation or policy/optimizer construction with
`PPO_OPTIMIZER_STEPS=0`.  Only a durable independently checked probe PASS may
continue in that same job into a completely fresh full V56 transaction.  No
partial checkpoint from jobs `17939578`, `17940941` or any other failed V56 run
may be loaded, continued or evaluated as a candidate.

## R4 name-based contact-index repair

R3 job `17944428` stopped after about 41 seconds in the pre-training probe's
contact-contract receipt, before any calf trial, `_reset_idx`, high-policy or
optimizer construction.  Its live receipt showed that `accepted_body_names`
and `live_body_names` were the exact qualified ordered 19-name array.  The
robot-side `base_contact_body_names` contained those same 19 unique names but
in a different deterministic grouped order: base, then each front leg, both
head bodies, and then each rear leg.  The probe rejected this valid enumeration
because it incorrectly required all three interfaces to share one positional
index.  PPO optimizer steps remained zero, so this is engineering evidence and
does not exercise or falsify the R2 reset repair or any task behavior.

Job `16315309` qualified the live articulation-name to exact ball-filter
mapping.  It did not qualify the independently enumerated robot-side
`ContactSensor.body_names` order.  The 75 static tests used synthetic arrays
copied from the accepted contract and therefore could verify the declared
positional contract only; they had no live Isaac/PhysX enumeration from which
to discover the mismatch.

R4 makes the smallest probe-only correction: join the three interfaces by the
calf name and record three separate indices.  `robot_body_index` is used only
to obtain the live calf pose, `ball_filter_index` only to read the ball-side
filtered pair force, and `base_sensor_body_index` only to read the robot-side
net force.  The accepted and live arrays must still equal the exact qualified
19-name order.  The base-sensor array must be a duplicate-free permutation of
that same exact name set, and every recorded index must map back to the same
calf name.  The same-tick, same-calf, both-sensors `>=1 N` gate and every reset,
raw-view, one-tick, normal-low and sentinel requirement remain unchanged.

Only the probe, its evidence/checker and closure tests may change.  Production
environment/reset, contract, reward, observation, action, terminal, PPO,
budget and fixed-evaluation bytes remain unchanged.  Any mapping or probe
failure stops before evaluation or policy/optimizer construction with zero PPO
steps; only a durable full probe PASS may continue into the unchanged fresh
one-job transaction.  No failed V56 partial checkpoint is reusable.  R4
delivery identities are published outside this self-hashed source tree in the
workspace README, FILE_INDEX and external source-manifest receipt.

## Cluster use

After installing the delivered archive at
`/scratch/lj2634/Isaac-Go2-Low-Level-RL/experiments/56_sparse_high_task_progress_canary_v1`, run:

```bash
cd /scratch/lj2634/Isaac-Go2-Low-Level-RL/experiments/56_sparse_high_task_progress_canary_v1
bash submit_and_check_sparse_high_task_progress_canary.sh
```

That one command performs the full login-node gate once, waits for the single
Slurm transaction and invokes the external checker.  Do not run the evaluator,
probe or trainer directly: the submitter and job wrapper carry the exact
source/parent bindings and evidence closure.

`V56_PREFLIGHT_ONLY=1` remains an optional standalone login-node audit when no
formal submission will immediately follow:

```bash
V56_PREFLIGHT_ONLY=1 bash submit_and_check_sparse_high_task_progress_canary.sh
```

It is not a prerequisite.  Do not run this optional command and then run the
formal command in the same sequence, because the formal command deliberately
repeats the complete login-node gate against the bytes it actually submits.

## Decision boundary

- `V56_FIRST_ROLLOUT_REWARD_SUPPORT_REJECTED_STOP_BEFORE_PPO_UPDATE` means the
  reward signal failed its declared support gate; no learning claim follows.
- Other completed decisions require the full bounded training and final fixed
  evaluation.  Update integrity alone never establishes that a direction or
  the task is solved.
- Every branch prints `LONG_TRAINING_AUTHORIZED=NO`.  A later multi-seed or long
  run needs separate evidence and authorization.

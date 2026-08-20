# R8.59 parent freeze before integration

This is a non-training, non-mutating freeze of the exact accepted parents. It copies V4D final high/low, V3 ancestry, the R8.58 complete kick chain, accepted contracts, and source snapshots into a new read-only directory and creates one SHA256 manifest.

From the project root:

```bash
cp -a /path/to/r8_59_parent_freeze experiments/
bash experiments/r8_59_parent_freeze/preflight_r859_parents.sh
bash experiments/r8_59_parent_freeze/freeze_r859_parents.sh
bash experiments/r8_59_parent_freeze/check_r859_freeze.sh
```

Only continue after preflight prints `VERDICT=R859_PREFLIGHT_OK`. No source patch and no training should start unless the checker ends with `R859_FROZEN_PARENTS_READY_FOR_ISOLATED_ROUTER_SOURCE`.

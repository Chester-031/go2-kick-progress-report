# R8.60 isolated action-owner smoke

This package is deliberately independent of Isaac Sim and PyTorch. It verifies
the integration boundary before any parent source is copied or patched:

- V4D owns normal locomotion/push/stand actions.
- R8.58 owns the complete kick strike option.
- R8.55 owns post-contact recovery.
- The kick option cannot be interrupted by a new high-level request.
- Exactly one 12D action reaches the underlying low-level step per tick.
- The terminal zero-sum reward contract remains `+1 / -1 / 0`.

This is a mechanics test, not a training result. Only after it passes should the
same router be inserted into a copy of the R8.59 frozen V4D source for Isaac
deterministic smoke and close-up slow-motion video.

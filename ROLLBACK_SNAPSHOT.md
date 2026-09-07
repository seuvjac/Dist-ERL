# FedEvoSAC rollback snapshot

This worktree starts from commit `8891211`, the implementation used by the
July 2026 repeated experiment that produced the selected Walker2d and Hopper
figures. The historical source data are stored in the main worktree under:

`logs/experiments/fedevosac_20x2_converged_20260714`

Only environment registration and per-environment launch presets for
`Ant-v5` and `HalfCheetah-v5` are added here. The FedEvoSAC optimization,
archive, local SAC, and aggregation implementation remains at the rollback
revision.

The three-seed multi-environment run uses the fixed seeds `0 1 2`. Existing
Walker2d, Hopper, and Swimmer runs are reused; only Ant and HalfCheetah are
newly trained. This is an exploratory comparison and must not replace the
complete, pre-registered formal aggregate.

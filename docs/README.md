# Dist-ERL Documentation

## Project Direction

Dist-ERL is now the paper method. New experiments should use:

```bash
./run_dist_erl.sh --mode dist_erl
```

The old distributed Re2 direction is abandoned. `erl_re2` remains available only as a single-worker baseline for comparison with ERL-Re2.

## Main Components

| Component | File | Role |
|-----------|------|------|
| EA Manager | `src/manager.py` | Maintains the EA population, elites, crossover, mutation, diversity boosts |
| RL Learner | `src/learner.py` | Runs DDPG/TD3/PPO policy learning and replay-buffer updates |
| Rollout Worker | `src/worker.py` | Evaluates individuals in parallel with Ray |
| Training Loop | `src/main.py` | Coordinates EA evaluation/evolution, RL updates, logging, and plotting |

## Recommended Commands

```bash
cd ~/code/Dist-ERL

# Quick smoke
./run_dist_erl.sh --env Hopper-v2 --mode dist_erl --max-generations 10

# Paper-scale multi-seed comparison
./run_seeds.sh

# Worker scaling and bandwidth
./run_scaling.sh
python3 scripts/plot_scaling_bandwidth.py --log-dir logs

# Publication plots from real logs
python3 generate_plots.py --log-dir logs --require-real
```

## Baselines

Default paper comparison modes:

```text
pure_rl pure_ea standard_erl erl_re2 dist_erl
```

`dist_erl` is the final method. `erl_re2` is retained as a baseline, not as the project thesis.

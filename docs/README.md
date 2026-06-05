# FedEvoRL Documentation

## Project Direction

FedEvoRL is now the paper method. New experiments should use:

```bash
./run_fed_evo_rl.sh --mode fed_evo_rl
```

The previous Dist-ERL direction is retained as a baseline (`dist_erl`). The old distributed Re2 direction is abandoned. `erl_re2` remains available only as a single-worker baseline.

## Main Components

| Component | File | Role |
|-----------|------|------|
| Federated Client | `src/federated.py` | Owns private environment, local replay buffer, local RL update, model upload |
| EA Server | `src/manager.py` | Maintains policy population, selection, crossover, mutation, diversity |
| Training Loop | `src/main.py` | Coordinates cross-client evaluation, federated aggregation, EA injection, logging |
| Baseline Learner | `src/learner.py` | Supports pure RL / ERL / ERL-Re2 baselines |
| Baseline Worker | `src/worker.py` | Supports distributed ERL baseline evaluation |

## Recommended Commands

```bash
cd ~/code/Dist-ERL

# Quick smoke
./run_fed_evo_rl.sh --env Pendulum-v1 --population-size 6 --num-clients 2 --max-generations 3

# Paper-scale comparison
./run_seeds.sh

# Client scaling and bandwidth
./run_scaling.sh
python3 scripts/plot_scaling_bandwidth.py --log-dir logs

# Publication plots from real logs
python3 generate_plots.py --log-dir logs --require-real
```

## Baselines

Default paper comparison modes:

```text
pure_rl pure_ea standard_erl erl_re2 dist_erl fed_evo_rl
```

`fed_evo_rl` is the final method. `dist_erl` and `erl_re2` are retained as baselines, not as the project thesis.

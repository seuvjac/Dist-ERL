#!/usr/bin/env bash
# Stable-Baselines3 baselines for the FedRL heterogeneous environment suite.
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

ENVS=${ENVS:-"CartPole-v1 MountainCar-v0 Acrobot-v1 LunarLander-v3"}
SEEDS=${SEEDS:-"0 1 2"}
SB3_ALGOS=${SB3_ALGOS:-"PPO"}
SB3_LOG_DIR=${SB3_LOG_DIR:-"logs/logs_sb3"}

for ENV_NAME in $ENVS; do
  read _POP _CLIENTS GENS STEPS <<<"$(python3 - <<PY
from src.config import env_run_preset
p = env_run_preset('$ENV_NAME')
print(p['population_size'], p['num_workers'], p['max_generations'], p['max_episode_steps'])
PY
)"
  for SEED in $SEEDS; do
    for ALGO in $SB3_ALGOS; do
      EXP="sb3_${ALGO,,}_${ENV_NAME}_s${SEED}"
      python3 scripts/train_sb3_baseline.py \
        --env "$ENV_NAME" \
        --algo "$ALGO" \
        --seed "$SEED" \
        --total-timesteps $(( GENS * STEPS )) \
        --eval-interval "$STEPS" \
        --eval-episodes 4 \
        --max-episode-steps "$STEPS" \
        --log-dir "$SB3_LOG_DIR" \
        --exp-name "$EXP"
    done
  done
done

python3 scripts/plot_fedrl_heterogeneous.py \
  --fed-log-dir logs/logs_fedrl_smoke \
  --sb3-log-dir "$SB3_LOG_DIR" \
  --out-dir plots/sb3_heterogeneous

#!/usr/bin/env bash
# FedEvoRL runs for every environment used by the SB3 comparison suite.
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

ENVS=${ENVS:-"CartPole-v1 Acrobot-v1 LunarLander-v3 LunarLanderContinuous-v3 BipedalWalkerHardcore-v3"}
SEEDS=${SEEDS:-"0"}
FED_VARIANTS=${FED_VARIANTS:-"full uniform_aggregation no_local_rl no_ea_injection no_heterogeneity"}
LOG_DIR=${LOG_DIR:-"logs_fedrl_hetero"}
SB3_LOG_DIR=${SB3_LOG_DIR:-"logs_sb3"}
OUT_DIR=${OUT_DIR:-"plots/fedrl_vs_sb3"}

for ENV_NAME in $ENVS; do
  read POP CLIENTS GENS STEPS <<<"$(python3 - <<PY
from src.config import env_run_preset
p = env_run_preset('$ENV_NAME')
print(p['population_size'], p['num_workers'], p['max_generations'], p['max_episode_steps'])
PY
)"

  # Keep the first all-environment FedEvoRL pass tractable; raise overrides for final paper runs.
  POP=${FED_POPULATION_SIZE:-$(( POP < 12 ? POP : 12 ))}
  CLIENTS=${FED_NUM_CLIENTS:-$(( CLIENTS < 2 ? CLIENTS : 2 ))}
  GENS=${FED_MAX_GENERATIONS:-$(( GENS < 20 ? GENS : 20 ))}
  EVAL_EPISODES=${FED_EVAL_EPISODES:-2}
  CLIENT_ROLLOUTS=${FED_CLIENT_ROLLOUTS:-1}
  CLIENT_UPDATES=${FED_CLIENT_UPDATES:-2}

  for SEED in $SEEDS; do
    for VARIANT in $FED_VARIANTS; do
      EXP="fedrlhet_${ENV_NAME}_${VARIANT}_s${SEED}"
      python3 -m src.main \
        --env "$ENV_NAME" \
        --mode fed_evo_rl \
        --fed-ablation "$VARIANT" \
        --algorithm DDPG \
        --population-size "$POP" \
        --num-clients "$CLIENTS" \
        --max-generations "$GENS" \
        --max-episode-steps "$STEPS" \
        --client-heterogeneity 0.35 \
        --client-heterogeneity-mode env_params \
        --fed-aggregation softmax \
        --fed-aggregation-interval 5 \
        --fed-aggregation-temperature 75 \
        --elite-archive-size 5 \
        --elite-archive-restore-copies 1 \
        --client-rollouts "$CLIENT_ROLLOUTS" \
        --client-updates "$CLIENT_UPDATES" \
        --batch-size 32 \
        --eval-episodes "$EVAL_EPISODES" \
        --seed "$SEED" \
        --log-dir "$LOG_DIR" \
        --exp-name "$EXP"
    done
  done
done

python3 scripts/plot_fedrl_heterogeneous.py \
  --fed-log-dir "$LOG_DIR" \
  --sb3-log-dir "$SB3_LOG_DIR" \
  --out-dir "$OUT_DIR"

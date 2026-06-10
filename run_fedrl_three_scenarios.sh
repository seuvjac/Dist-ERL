#!/usr/bin/env bash
# FedEvoFSAC on three heterogeneous federated scenarios for the original discrete envs.
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

ENVS=${ENVS:-"CartPole-v1 MountainCar-v0 Acrobot-v1 LunarLander-v3"}
SCENARIOS=${SCENARIOS:-"dynamics_mild sensor_reward mixed_hard"}
SEEDS=${SEEDS:-"0 1 2"}
FED_VARIANTS=${FED_VARIANTS:-"full"}
LOG_ROOT=${LOG_ROOT:-"logs_fedrl_scenarios"}
OUT_DIR=${OUT_DIR:-"plots/fedrl_scenarios"}
ALG=${FED_ALGORITHM:-"FSAC"}

for SCENARIO in $SCENARIOS; do
  read HET_MODE HET_STRENGTH <<<"$(python3 - <<PY
from src.config import FEDRL_HETEROGENEITY_SCENARIOS
s = FEDRL_HETEROGENEITY_SCENARIOS['$SCENARIO']
print(s['mode'], s['strength'])
PY
)"
  LOG_DIR="$LOG_ROOT/$SCENARIO"
  for ENV_NAME in $ENVS; do
    read POP CLIENTS GENS STEPS <<<"$(python3 - <<PY
from src.config import env_run_preset
p = env_run_preset('$ENV_NAME')
print(p['population_size'], p['num_workers'], p['max_generations'], p['max_episode_steps'])
PY
)"
    POP=${FED_POPULATION_SIZE:-$POP}
    CLIENTS=${FED_NUM_CLIENTS:-$CLIENTS}
    GENS=${FED_MAX_GENERATIONS:-$GENS}
    EVAL_EPISODES=${FED_EVAL_EPISODES:-4}
    CLIENT_ROLLOUTS=${FED_CLIENT_ROLLOUTS:-2}
    CLIENT_UPDATES=${FED_CLIENT_UPDATES:-8}

    for SEED in $SEEDS; do
      for VARIANT in $FED_VARIANTS; do
        EXP="scenario_${SCENARIO}_${ENV_NAME}_${VARIANT}_s${SEED}"
        python3 -m src.main \
          --env "$ENV_NAME" \
          --mode fed_evo_rl \
          --fed-ablation "$VARIANT" \
          --algorithm "$ALG" \
          --population-size "$POP" \
          --num-clients "$CLIENTS" \
          --max-generations "$GENS" \
          --max-episode-steps "$STEPS" \
          --client-heterogeneity "$HET_STRENGTH" \
          --client-heterogeneity-mode "$HET_MODE" \
          --fed-aggregation softmax \
          --fed-aggregation-interval 5 \
          --fed-aggregation-temperature 75 \
          --fed-delta-clip-norm 5 \
          --ea-weight-clip 5 \
          --elite-archive-size 5 \
          --elite-archive-restore-copies 1 \
          --client-rollouts "$CLIENT_ROLLOUTS" \
          --client-updates "$CLIENT_UPDATES" \
          --eval-episodes "$EVAL_EPISODES" \
          --seed "$SEED" \
          --log-dir "$LOG_DIR" \
          --exp-name "$EXP"
      done
    done
  done
done

python3 scripts/plot_fedrl_scenarios.py \
  --log-root "$LOG_ROOT" \
  --out-dir "$OUT_DIR" \
  --envs $ENVS

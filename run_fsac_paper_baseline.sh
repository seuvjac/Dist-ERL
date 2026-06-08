#!/usr/bin/env bash
set -euo pipefail

# Paper-style FSAC reproduction for CartPole, Acrobot and LunarLander.

ENVS=${ENVS:-"CartPole-v1 Acrobot-v1 LunarLander-v3"}
SEEDS=${SEEDS:-"0 1 2"}
MODES=${MODES:-"paper_fsac paper_sac"}
ROUNDS=${ROUNDS:-120}
NUM_WORKERS=${NUM_WORKERS:-5}
LOG_DIR=${LOG_DIR:-"logs_fsac_paper"}

for ENV_NAME in $ENVS; do
  for MODE in $MODES; do
    for SEED in $SEEDS; do
      FLAG="--federated"
      if [[ "$MODE" == "paper_sac" ]]; then
        FLAG="--no-federation"
      fi
      EXP="${MODE}_${ENV_NAME}_s${SEED}"
      echo ">>> ${EXP}"
      python -u scripts/train_fsac_paper_baseline.py \
        --env "$ENV_NAME" \
        --seed "$SEED" \
        --rounds "$ROUNDS" \
        --num-workers "$NUM_WORKERS" \
        --log-dir "$LOG_DIR" \
        --exp-name "$EXP" \
        $FLAG
    done
  done
done

python scripts/plot_fedrl_heterogeneous.py \
  --fed-log-dir logs_fedrl_hetero \
  --paper-log-dir "$LOG_DIR" \
  --out-dir plots/fedrl_heterogeneous

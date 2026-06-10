#!/usr/bin/env bash
set -euo pipefail

# Paper-style FSAC reproduction for CartPole, Acrobot and LunarLander.

ENVS=${ENVS:-"CartPole-v1 MountainCar-v0 Acrobot-v1 LunarLander-v3"}
SEEDS=${SEEDS:-"0 1 2"}
MODES=${MODES:-"paper_sac paper_fsac fedavg_sac fedsoftmax_sac_noea fedbest_sac fedmedian_sac fedtrimmedmean_sac attention_sac_lite"}
ROUNDS=${ROUNDS:-120}
NUM_WORKERS=${NUM_WORKERS:-5}
LOG_DIR=${LOG_DIR:-"logs_fsac_paper_mixed"}
CLIENT_HETEROGENEITY=${CLIENT_HETEROGENEITY:-"0.60"}
CLIENT_HETEROGENEITY_MODE=${CLIENT_HETEROGENEITY_MODE:-"mixed"}

for ENV_NAME in $ENVS; do
  for MODE in $MODES; do
    for SEED in $SEEDS; do
      EXP="${MODE}_${ENV_NAME}_s${SEED}"
      echo ">>> ${EXP}"
      python -u scripts/train_fsac_paper_baseline.py \
        --env "$ENV_NAME" \
        --seed "$SEED" \
        --rounds "$ROUNDS" \
        --num-workers "$NUM_WORKERS" \
        --client-heterogeneity "$CLIENT_HETEROGENEITY" \
        --client-heterogeneity-mode "$CLIENT_HETEROGENEITY_MODE" \
        --log-dir "$LOG_DIR" \
        --exp-name "$EXP" \
        --baseline-mode "$MODE"
    done
  done
done

python scripts/plot_fedrl_heterogeneous.py \
  --fed-log-dir logs_fedrl_hetero_mixed \
  --paper-log-dir "$LOG_DIR" \
  --out-dir plots/fedrl_comparison_mixed \
  --plot-kind comparison

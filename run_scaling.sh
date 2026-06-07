#!/bin/bash
# Client scalability + bandwidth logging for fed_evo_rl
set -e

ENV_NAME=${ENV_NAME:-"CartPole-v1"}
SEED=${SEED:-42}
GENS=${GENS:-80}

for NW in 1 2 4 8; do
  EXP="scaling_${ENV_NAME}_w${NW}_s${SEED}"
  echo ">>> workers=$NW"
  ./run_fed_evo_rl.sh \
    --env "$ENV_NAME" \
    --mode fed_evo_rl \
    --exp-name "$EXP" \
    --num-clients "$NW" \
    --population-size 40 \
    --max-generations "$GENS" \
    --seed "$SEED"
done

echo "Plot scaling: python3 scripts/plot_scaling_bandwidth.py --log-dir logs"

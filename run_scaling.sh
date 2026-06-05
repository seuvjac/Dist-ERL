#!/bin/bash
# Worker scalability + bandwidth logging for dist_erl
set -e

ENV_NAME=${ENV_NAME:-"Ant-v2"}
SEED=${SEED:-42}
GENS=${GENS:-80}

for NW in 1 2 4 8; do
  EXP="scaling_${ENV_NAME}_w${NW}_s${SEED}"
  echo ">>> workers=$NW"
  ./run_dist_erl.sh \
    --env "$ENV_NAME" \
    --mode dist_erl \
    --exp-name "$EXP" \
    --num-workers "$NW" \
    --population-size 40 \
    --max-generations "$GENS" \
    --seed "$SEED"
done

echo "Plot scaling: python3 scripts/plot_scaling_bandwidth.py --log-dir logs"

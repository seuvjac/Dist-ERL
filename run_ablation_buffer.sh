#!/bin/bash
# EA batch ratio ablation for the ERL-Re2 baseline
set -e

ENV_NAME=${ENV_NAME:-"HalfCheetah-v2"}
SEED=${SEED:-42}

for RATIO in 0.0 0.25 0.5 0.75; do
  ./run_dist_erl.sh \
    --env "$ENV_NAME" \
    --mode erl_re2 \
    --exp-name "abl_erl_re2_ea_ratio_${RATIO}_s${SEED}" \
    --ea-batch-ratio "$RATIO" \
    --num-workers 1 \
    --population-size 40 \
    --max-generations 80 \
    --seed "$SEED"
done

echo "Compare eval curves filtered by exp-name abl_ea_ratio_*"

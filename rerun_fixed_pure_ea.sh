#!/usr/bin/env bash
set -euo pipefail

source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate dist-erl-re2
cd "$HOME/code/Dist-ERL"

rm -rf \
  logs/logs_compare/codex_cmp_hopper_pure_ea_s0 \
  logs/logs_compare/codex_cmp_hopper_pure_ea_s1 \
  logs/logs_compare/codex_cmp_hopper_pure_ea_s2

for seed in 0 1 2; do
  echo "RUN fixed pure_ea seed=${seed}"
  python -m src.main \
    --mode pure_ea \
    --env Hopper-v2 \
    --seed "$seed" \
    --max-generations 10 \
    --population-size 10 \
    --num-workers 2 \
    --max-episode-steps 200 \
    --eval-episodes 3 \
    --sync-interval 2 \
    --rl-updates 3 \
    --elite-seeds 2 \
    --batch-size 32 \
    --rl-rollouts 1 \
    --rl-rollouts-between-sync 1 \
    --log-dir ./logs/logs_compare \
    --exp-name "codex_cmp_hopper_pure_ea_s${seed}"
done

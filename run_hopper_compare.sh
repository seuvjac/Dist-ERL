#!/usr/bin/env bash
set -euo pipefail

source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate dist-erl-re2
cd "$HOME/code/Dist-ERL"

mkdir -p logs_compare

for mode in pure_rl pure_ea dist_erl erl_re2 fed_evo_rl; do
  for seed in 0 1 2; do
    echo "RUN mode=${mode} seed=${seed}"
    python -m src.main \
      --mode "$mode" \
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
      --log-dir ./logs_compare \
      --exp-name "codex_cmp_hopper_${mode}_s${seed}"
  done
done

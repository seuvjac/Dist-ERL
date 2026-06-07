#!/bin/bash
# FedEvoRL benchmark: baselines + ERL-Re2 + Dist-ERL + FedEvoRL main method
set -e

ENV_NAME=${ENV_NAME:-"CartPole-v1"}
SEED=${SEED:-42}
ENV_SLUG=$(echo "$ENV_NAME" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/_/g')

echo "Benchmark suite: $ENV_NAME, seed=$SEED"

run_experiment() {
    local mode=$1
    local exp_name=$2
    shift 2
    echo "Running $exp_name ($mode)"
    ./run_dist_erl.sh \
        --mode "$mode" \
        --exp-name "$exp_name" \
        --env "$ENV_NAME" \
        --max-generations 100 \
        --population-size 30 \
        --seed "$SEED" \
        "$@"
    echo "Done $exp_name"
    echo ""
}

# Baselines
run_experiment "pure_rl" "bench_${ENV_SLUG}_pure_rl" --population-size 10 --num-workers 1
run_experiment "pure_ea" "bench_${ENV_SLUG}_pure_ea" --num-workers 2
run_experiment "standard_erl" "bench_${ENV_SLUG}_standard_erl" --num-workers 1
run_experiment "erl_re2" "bench_${ENV_SLUG}_erl_re2" --num-workers 1
run_experiment "dist_erl" "bench_${ENV_SLUG}_dist_erl" --num-workers 4
run_experiment "fed_evo_rl" "bench_${ENV_SLUG}_fed_evo_rl" --num-clients 4 --client-fraction 1.0

echo "Benchmarks completed."
echo "  python3 generate_plots.py --log-dir logs --env $ENV_NAME"

#!/bin/bash
# Dist-ERL ablation suite: baselines, Dist-ERL scalability, and ERL-Re2 baseline diagnostics
set -e

ENV_NAME=${ENV_NAME:-"LunarLanderContinuous-v3"}
POPULATION_SIZE=${POPULATION_SIZE:-30}
MAX_GENERATIONS=${MAX_GENERATIONS:-100}
SEEDS=${SEEDS:-"42"}

echo "Ablation suite: env=$ENV_NAME, generations=$MAX_GENERATIONS, seeds=$SEEDS"

run_one() {
    local extra_args=("$@")
    echo ">>> ${extra_args[*]}"
    ./run_dist_erl.sh "${extra_args[@]}"
    echo ""
}

for SEED in $SEEDS; do
    echo "========== Seed $SEED =========="

    # --- Main baselines (related experiments) ---
    run_one --mode pure_rl --exp-name "abl_${ENV_NAME}_pure_rl_s${SEED}" --seed "$SEED" \
        --population-size 10 --num-workers 1
    run_one --mode pure_ea --exp-name "abl_${ENV_NAME}_pure_ea_s${SEED}" --seed "$SEED" \
        --num-workers 2
    run_one --mode standard_erl --exp-name "abl_${ENV_NAME}_standard_erl_s${SEED}" --seed "$SEED" \
        --num-workers 1
    run_one --mode erl_re2 --exp-name "abl_${ENV_NAME}_erl_re2_s${SEED}" --seed "$SEED" \
        --num-workers 1
    run_one --mode dist_erl --exp-name "abl_${ENV_NAME}_dist_erl_s${SEED}" --seed "$SEED" \
        --num-workers 4

    # --- ERL-Re2 component ablations (baseline diagnostics only) ---
    run_one --mode erl_re2 --ablation no_re2 \
        --exp-name "abl_${ENV_NAME}_erl_re2_off_s${SEED}" --seed "$SEED" --num-workers 1
    run_one --mode erl_re2 --ablation no_reproduction \
        --exp-name "abl_${ENV_NAME}_erl_re2_no_repro_s${SEED}" --seed "$SEED" --num-workers 1
    run_one --mode erl_re2 --ablation no_migration \
        --exp-name "abl_${ENV_NAME}_erl_re2_no_migrate_s${SEED}" --seed "$SEED" --num-workers 1

    # --- Dist-ERL scalability: worker count ---
    for NW in 1 2 4; do
        run_one --mode dist_erl \
            --exp-name "abl_${ENV_NAME}_dist_erl_w${NW}_s${SEED}" --seed "$SEED" \
            --num-workers "$NW"
    done

    # --- RL update budget ablation for Dist-ERL ---
    run_one --mode dist_erl --rl-updates 5 \
        --exp-name "abl_${ENV_NAME}_dist_erl_rlupd5_s${SEED}" --seed "$SEED" --num-workers 4
    run_one --mode dist_erl --rl-updates 20 \
        --exp-name "abl_${ENV_NAME}_dist_erl_rlupd20_s${SEED}" --seed "$SEED" --num-workers 4
done

echo "Ablation suite done."
echo "  python3 generate_plots.py --log-dir logs --env $ENV_NAME"

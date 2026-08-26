#!/usr/bin/env bash
# Three-seed Hopper-Locomotion component ablation pilot.
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export MPLBACKEND=${MPLBACKEND:-Agg}

EXPERIMENT_ID=${EXPERIMENT_ID:-"fedevosac_hopper_locomotion_ablation_pilot_$(date +%Y%m%d)"}
SEEDS=${SEEDS:-"0 1 2"}
TARGET_ENV_STEPS=${TARGET_ENV_STEPS:-300000}
FED_VARIANTS=${FED_VARIANTS:-"full no_local_rl no_ea_injection no_heterogeneity"}
RUN_ROOT=${RUN_ROOT:-"logs/experiments/$EXPERIMENT_ID"}
PLOT_ROOT=${PLOT_ROOT:-"plots/experiments/$EXPERIMENT_ID"}

mkdir -p "$RUN_ROOT/fedevosac" "$PLOT_ROOT"
cat > "$RUN_ROOT/RUN_CONFIG.env" <<EOF
experiment_id=$EXPERIMENT_ID
git_commit=$(git rev-parse HEAD)
environment=Hopper-v5
display_task=Hopper-Locomotion
healthy_reward=0.05
forward_reward_weight=1.0
heterogeneity=0.25/env_params_only
client_updates=96
critic_warmup_updates=88
actor_lr=0.00003
client_validation_episodes=2
inject_margin=0.01
migration_copies=1
migration_blend=0.25
inject_noise=0.002
seeds=$SEEDS
fed_variants=$FED_VARIANTS
target_interactions_per_run=$TARGET_ENV_STEPS
EOF

pids=()
labels=()
for seed in $SEEDS; do
  seed_log_dir="$RUN_ROOT/fedevosac/seed_$seed"
  mkdir -p "$seed_log_dir"
  echo "Launch Hopper-Locomotion seed $seed"
  ENVS="Hopper-v5" \
  SEEDS="$seed" \
  FED_VARIANTS="$FED_VARIANTS" \
  SAC_BASELINES="" \
  BUDGET_PRESET=converged \
  TARGET_ENV_STEPS="$TARGET_ENV_STEPS" \
  LOG_DIR="$seed_log_dir" \
  SAC_LOG_DIR="$RUN_ROOT/baselines/seed_$seed" \
  SKIP_EXISTING=1 \
  SKIP_PLOTS=1 \
  bash run_continuous_fedevosac_suite.sh > "$RUN_ROOT/seed_${seed}.log" 2>&1 &
  pids+=("$!")
  labels+=("$seed")
done

failed=0
for idx in "${!pids[@]}"; do
  if wait "${pids[$idx]}"; then
    echo "Seed ${labels[$idx]} completed"
  else
    echo "Seed ${labels[$idx]} failed; inspect $RUN_ROOT/seed_${labels[$idx]}.log" >&2
    failed=1
  fi
done
(( failed == 0 )) || exit 1

FED_LOG_DIR="$RUN_ROOT/fedevosac" \
SAC_LOG_DIR="$RUN_ROOT/baselines" \
OUT_ROOT="$PLOT_ROOT/aggregate" \
ENVS="Hopper-v5" \
PLOT_VARIANCE=ci90 \
bash scripts/render_fedrl_paper_bundle.sh

echo "Completed Hopper-Locomotion pilot: $PLOT_ROOT/aggregate/paper_figures"

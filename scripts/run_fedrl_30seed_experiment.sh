#!/usr/bin/env bash
# Thirty pre-registered independent seeds for paper-level FedEvoSAC evaluation.
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export MPLBACKEND=${MPLBACKEND:-Agg}
export RAY_local_fs_capacity_threshold=${RAY_local_fs_capacity_threshold:-0.99}

EXPERIMENT_ID=${EXPERIMENT_ID:-"fedevosac_formal_30seed_95ci_$(date +%Y%m%d)"}
SEED_BASE=${SEED_BASE:-100}
NUM_SEEDS=${NUM_SEEDS:-30}
PARALLEL_SEEDS=${PARALLEL_SEEDS:-3}
START_INDEX=${START_INDEX:-0}
END_INDEX=${END_INDEX:-$((NUM_SEEDS - 1))}
ENVS=${ENVS:-"Walker2d-v5 Hopper-v5 Ant-v5 HalfCheetah-v5 Swimmer-v5"}
FED_VARIANTS=${FED_VARIANTS:-"full no_local_rl no_ea_injection no_heterogeneity"}
SAC_BASELINES=${SAC_BASELINES-"fedavg_sac fedbest_sac fedsoftmax_sac_noea fedmedian_sac"}
TARGET_ENV_STEPS=${TARGET_ENV_STEPS:-1200000}
RUN_ROOT=${RUN_ROOT:-"logs/experiments/$EXPERIMENT_ID"}
PLOT_ROOT=${PLOT_ROOT:-"plots_2/$EXPERIMENT_ID"}

if (( NUM_SEEDS < 1 || PARALLEL_SEEDS < 1 )); then
  echo "NUM_SEEDS and PARALLEL_SEEDS must be positive" >&2
  exit 2
fi
if (( END_INDEX >= NUM_SEEDS )); then
  END_INDEX=$((NUM_SEEDS - 1))
fi

mkdir -p "$RUN_ROOT/fedevosac" "$RUN_ROOT/baselines" "$PLOT_ROOT"

MANIFEST="$PLOT_ROOT/seed_manifest.csv"
printf 'seed_index,seed,fed_log_dir,baseline_log_dir\n' > "$MANIFEST"
for ((index = 0; index < NUM_SEEDS; index++)); do
  seed=$((SEED_BASE + index))
  printf '%s,%s,%s,%s\n' \
    "$index" "$seed" \
    "$RUN_ROOT/fedevosac/seed_$seed" \
    "$RUN_ROOT/baselines/seed_$seed" >> "$MANIFEST"
done

# These five are fixed before outcomes exist and are only eligible for optional
# individual-run illustrations. All aggregate curves, CIs, and tests use all 30.
DISPLAY_SEEDS=(
  "$SEED_BASE"
  "$((SEED_BASE + 7))"
  "$((SEED_BASE + 14))"
  "$((SEED_BASE + 21))"
  "$((SEED_BASE + 28))"
)

cat > "$PLOT_ROOT/PROTOCOL.txt" <<EOF
Environments: $ENVS
Methods: FedEvoSAC-full, four federated SAC baselines, and three separate ablations.
Seeds: $SEED_BASE through $((SEED_BASE + NUM_SEEDS - 1)); all $NUM_SEEDS are included in inference.
Optional pre-registered display seeds: ${DISPLAY_SEEDS[*]}.
Post-hoc seed selection based on scores is prohibited.
Target counted interactions per method/environment/seed: $TARGET_ENV_STEPS.
Curves: current deployable return mean with two-sided 95% Student-t CI.
Tables: mean +/- two-sided 95% Student-t CI; sample SD remains as an audit field.
Significance: paired seed-matched Wilcoxon signed-rank tests, paired bootstrap 95% CI,
rank-biserial effect size, and Holm correction within each environment.
Outputs: communication-round and counted-interaction views only; normalized progress is disabled.
EOF

cat > "$RUN_ROOT/RUN_CONFIG.env" <<EOF
experiment_id=$EXPERIMENT_ID
git_commit=$(git rev-parse HEAD)
environments=$ENVS
seed_base=$SEED_BASE
num_seeds=$NUM_SEEDS
parallel_seeds=$PARALLEL_SEEDS
fed_variants=$FED_VARIANTS
sac_baselines=$SAC_BASELINES
target_interactions_per_run=$TARGET_ENV_STEPS
confidence_interval=two-sided_95pct_student_t
significance_test=paired_wilcoxon_signed_rank_holm
training_progress_plots=disabled
EOF

run_seed() {
  local seed=$1
  local fed_dir="$RUN_ROOT/fedevosac/seed_$seed"
  local baseline_dir="$RUN_ROOT/baselines/seed_$seed"
  mkdir -p "$fed_dir" "$baseline_dir"
  echo "Starting formal seed $seed"
  ENVS="$ENVS" \
  SEEDS="$seed" \
  REPEAT_ID="formal30" \
  FED_VARIANTS="$FED_VARIANTS" \
  SAC_BASELINES="$SAC_BASELINES" \
  BUDGET_PRESET=converged \
  TARGET_ENV_STEPS="$TARGET_ENV_STEPS" \
  LOG_DIR="$fed_dir" \
  SAC_LOG_DIR="$baseline_dir" \
  SKIP_EXISTING=1 \
  SKIP_PLOTS=1 \
  bash run_continuous_fedevosac_suite.sh
}

render_aggregate() {
  FED_LOG_DIR="$RUN_ROOT/fedevosac" \
  SAC_LOG_DIR="$RUN_ROOT/baselines" \
  OUT_ROOT="$PLOT_ROOT/aggregate" \
  ENVS="$ENVS" \
  PLOT_VARIANCE=ci95 \
  INCLUDE_PROGRESS=0 \
  bash scripts/render_fedrl_paper_bundle.sh
}

for ((batch_start = START_INDEX; batch_start <= END_INDEX; batch_start += PARALLEL_SEEDS)); do
  pids=()
  labels=()
  for ((offset = 0; offset < PARALLEL_SEEDS; offset++)); do
    index=$((batch_start + offset))
    (( index <= END_INDEX )) || break
    seed=$((SEED_BASE + index))
    seed_log="$RUN_ROOT/seed_${seed}.log"
    echo "Launch seed $seed -> $seed_log"
    run_seed "$seed" >"$seed_log" 2>&1 &
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
  render_aggregate
done

echo "Completed seeds $((SEED_BASE + START_INDEX))-$((SEED_BASE + END_INDEX))."
echo "Aggregate figures and tables: $PLOT_ROOT/aggregate"

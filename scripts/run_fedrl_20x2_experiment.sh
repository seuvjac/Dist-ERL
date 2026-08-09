#!/usr/bin/env bash
# Twenty independent outer repeats, with two unique seeds in each repeat.
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

EXPERIMENT_ID=${EXPERIMENT_ID:-"fedevosac_20x2_converged_$(date +%Y%m%d)"}
REPEATS=${REPEATS:-20}
START_REPEAT=${START_REPEAT:-1}
END_REPEAT=${END_REPEAT:-$REPEATS}
PARALLEL_REPEATS=${PARALLEL_REPEATS:-2}
SEED_BASE=${SEED_BASE:-0}
ENVS=${ENVS:-"Walker2d-v5 Hopper-v5"}
FED_VARIANTS=${FED_VARIANTS:-"full no_local_rl no_ea_injection no_heterogeneity"}
SAC_BASELINES=${SAC_BASELINES-"fedavg_sac fedbest_sac fedsoftmax_sac_noea fedmedian_sac"}
RUN_ROOT=${RUN_ROOT:-"logs/experiments/$EXPERIMENT_ID"}
PLOT_ROOT=${PLOT_ROOT:-"plots_new/$EXPERIMENT_ID"}
INCLUDE_LEGACY_REFERENCE=${INCLUDE_LEGACY_REFERENCE:-0}

mkdir -p "$RUN_ROOT/fedevosac" "$RUN_ROOT/baselines" "$PLOT_ROOT"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export MPLBACKEND=${MPLBACKEND:-Agg}

if [[ "$INCLUDE_LEGACY_REFERENCE" == "1" ]]; then
  REFERENCE_ROOT="$PLOT_ROOT/reference_single_seed"
  mkdir -p "$REFERENCE_ROOT/comparison"
  find plots/fedevosac_perenv_tuned_s0_comparison -maxdepth 1 -type f \
    \( -name 'Walker2d-v5*' -o -name 'Hopper-v5*' \) \
    -exec cp -t "$REFERENCE_ROOT/comparison" {} +
fi

MANIFEST="$PLOT_ROOT/repeat_manifest.csv"
printf 'repeat_id,seed_slot_0,seed_slot_1,fed_log_dir,baseline_log_dir,plot_dir\n' > "$MANIFEST"
for ((repeat = 1; repeat <= REPEATS; repeat++)); do
  repeat_id=$(printf '%02d' "$repeat")
  seed0=$((SEED_BASE + 2 * (repeat - 1)))
  seed1=$((seed0 + 1))
  printf '%s,%s,%s,%s,%s,%s\n' \
    "$repeat_id" "$seed0" "$seed1" \
    "$RUN_ROOT/fedevosac/repeat_$repeat_id" \
    "$RUN_ROOT/baselines/repeat_$repeat_id" \
    "$PLOT_ROOT/repeats/repeat_$repeat_id" >> "$MANIFEST"
done

printf '%s\n' \
  'Each outer repeat contains exactly two independent seeds.' \
  'Seed pairs are unique across repeats, so the aggregate result contains 40 independent seeds.' \
  'The formal environment matrix is Walker2d-v5 and Hopper-v5; Swimmer is excluded.' \
  'Legacy single-seed references are disabled by default and are always excluded from statistics.' \
  'Each environment renders three comparison views plus one separate ablation view (8 individual figures total).' \
  'Main evidence: current return vs communication rounds since first evaluation, with 90% CI.' \
  'Supplement display: current return vs interactions since each run first logged a real evaluation; rewards are never shifted.' \
  'Raw counted interaction totals remain in metrics.csv and the summary tables.' \
  'Diagnostic only: normalized training progress.' > "$PLOT_ROOT/PROTOCOL.txt"

{
  printf 'experiment_id=%s\n' "$EXPERIMENT_ID"
  printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
  printf 'environments=%s\n' "$ENVS"
  printf 'repeats=%s\n' "$REPEATS"
  printf 'seeds_per_repeat=2\n'
  printf 'seed_base=%s\n' "$SEED_BASE"
  printf 'parallel_repeats=%s\n' "$PARALLEL_REPEATS"
  printf 'fed_variants=%s\n' "$FED_VARIANTS"
  printf 'sac_baselines=%s\n' "$SAC_BASELINES"
  printf 'target_interactions_per_run=1200000\n'
} > "$RUN_ROOT/RUN_CONFIG.env"

run_repeat() {
  local repeat=$1
  repeat_id=$(printf '%02d' "$repeat")
  seed0=$((SEED_BASE + 2 * (repeat - 1)))
  seed1=$((seed0 + 1))
  fed_dir="$RUN_ROOT/fedevosac/repeat_$repeat_id"
  baseline_dir="$RUN_ROOT/baselines/repeat_$repeat_id"
  repeat_plot_dir="$PLOT_ROOT/repeats/repeat_$repeat_id"
  mkdir -p "$fed_dir" "$baseline_dir" "$repeat_plot_dir"

  echo "Starting repeat $repeat_id/$REPEATS with seeds $seed0 $seed1"
  ENVS="$ENVS" \
  SEEDS="$seed0 $seed1" \
  REPEAT_ID="$repeat_id" \
  FED_VARIANTS="$FED_VARIANTS" \
  SAC_BASELINES="$SAC_BASELINES" \
  BUDGET_PRESET=converged \
  LOG_DIR="$fed_dir" \
  SAC_LOG_DIR="$baseline_dir" \
  SKIP_EXISTING=1 \
  SKIP_PLOTS=1 \
  bash run_continuous_fedevosac_suite.sh

  FED_LOG_DIR="$fed_dir" \
  SAC_LOG_DIR="$baseline_dir" \
  OUT_ROOT="$repeat_plot_dir" \
  ENVS="$ENVS" \
  PLOT_VARIANCE=ci90 \
  bash scripts/render_fedrl_paper_bundle.sh
}

render_aggregate() {
  FED_LOG_DIR="$RUN_ROOT/fedevosac" \
  SAC_LOG_DIR="$RUN_ROOT/baselines" \
  OUT_ROOT="$PLOT_ROOT/aggregate" \
  ENVS="$ENVS" \
  PLOT_VARIANCE=ci90 \
  bash scripts/render_fedrl_paper_bundle.sh
}

if (( PARALLEL_REPEATS < 1 )); then
  echo "PARALLEL_REPEATS must be >= 1" >&2
  exit 2
fi
if (( END_REPEAT > REPEATS )); then
  END_REPEAT=$REPEATS
fi

for ((batch_start = START_REPEAT; batch_start <= END_REPEAT; batch_start += PARALLEL_REPEATS)); do
  pids=()
  labels=()
  for ((offset = 0; offset < PARALLEL_REPEATS; offset++)); do
    repeat=$((batch_start + offset))
    (( repeat <= END_REPEAT )) || break
    repeat_id=$(printf '%02d' "$repeat")
    repeat_log="$RUN_ROOT/repeat_${repeat_id}.log"
    echo "Launch repeat $repeat_id/$REPEATS -> $repeat_log"
    run_repeat "$repeat" >"$repeat_log" 2>&1 &
    pids+=("$!")
    labels+=("$repeat_id")
  done

  failed=0
  for idx in "${!pids[@]}"; do
    if wait "${pids[$idx]}"; then
      echo "Repeat ${labels[$idx]} completed"
    else
      echo "Repeat ${labels[$idx]} failed; inspect $RUN_ROOT/repeat_${labels[$idx]}.log" >&2
      failed=1
    fi
  done
  (( failed == 0 )) || exit 1
  render_aggregate
done

echo "Completed repeats $START_REPEAT-$END_REPEAT. Aggregate figures: $PLOT_ROOT/aggregate/paper_figures"

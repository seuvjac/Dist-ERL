#!/usr/bin/env bash
# Paired Swimmer pilot over predetermined weak/medium/strong seeds from the audit.
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PATH="${CONDA_ENV_BIN:-$HOME/anaconda3/envs/dist-erl-re2/bin}:$PATH"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export MPLBACKEND=${MPLBACKEND:-Agg}
export RAY_local_fs_capacity_threshold=${RAY_local_fs_capacity_threshold:-0.99}

MAIN_REPO=${MAIN_REPO:-/home/ywj/code/Dist-ERL}
EXPERIMENT_ID=${EXPERIMENT_ID:-fedevosac_swimmer_stability_paired8_20260912}
SEEDS=${SEEDS:-"1 2 3 6 12 14 17 22"}
MAX_PARALLEL=${MAX_PARALLEL:-6}
TARGET_ENV_STEPS=${TARGET_ENV_STEPS:-1200000}
RUN_ROOT=${RUN_ROOT:-$MAIN_REPO/logs/experiments/$EXPERIMENT_ID}
PLOT_ROOT=${PLOT_ROOT:-$MAIN_REPO/plots_2/$EXPERIMENT_ID}
FED_ROOT="$RUN_ROOT/fedevosac"
BASELINE_ROOT="$RUN_ROOT/baselines"
RUN_SUFFIX=swimstable8

mkdir -p "$FED_ROOT" "$BASELINE_ROOT" "$PLOT_ROOT"
cat > "$PLOT_ROOT/PROTOCOL.txt" <<EOF
Purpose: paired Swimmer stability pilot before a second 24-seed run
Source commit: $(git rev-parse HEAD)
Environment/method: Swimmer-v5 / FedEvoSAC-full
Predetermined audit seeds: $SEEDS
Seed strata: strong=1,2,6; medium=3; weak=12,14,17,22
Target counted interactions: $TARGET_ENV_STEPS per seed
Metric/uncertainty: current deployable policy / two-sided 95% Student-t CI
Swimmer changes: all-client aggregation, lower injection noise, archive-centered restart,
reduced disruptive mutation, and less sensitive stagnation detection
No baseline, ablation, HalfCheetah, Walker2d, Hopper, Ant, or seed selection
EOF

run_seed() {
  local seed=$1
  echo "Launch paired Swimmer seed $seed -> $RUN_ROOT/seed_${seed}.log"
  ENVS="Swimmer-v5" \
  SEEDS="$seed" \
  REPEAT_ID="$RUN_SUFFIX" \
  FED_VARIANTS="full" \
  SAC_BASELINES="" \
  BUDGET_PRESET="converged" \
  TARGET_ENV_STEPS="$TARGET_ENV_STEPS" \
  CLIENT_HETEROGENEITY="0.0" \
  CLIENT_HETEROGENEITY_MODE="none" \
  LOG_DIR="$FED_ROOT" \
  SAC_LOG_DIR="$BASELINE_ROOT" \
  SKIP_EXISTING=1 \
  SKIP_PLOTS=1 \
  bash run_continuous_fedevosac_suite.sh > "$RUN_ROOT/seed_${seed}.log" 2>&1
}

pids=()
labels=()
failed=0
for seed in $SEEDS; do
  run_seed "$seed" &
  pids+=("$!")
  labels+=("$seed")
  if (( ${#pids[@]} >= MAX_PARALLEL )); then
    for idx in "${!pids[@]}"; do
      if ! wait "${pids[$idx]}"; then
        echo "Swimmer seed ${labels[$idx]} failed" >&2
        failed=1
      fi
    done
    pids=()
    labels=()
  fi
done
for idx in "${!pids[@]}"; do
  if ! wait "${pids[$idx]}"; then
    echo "Swimmer seed ${labels[$idx]} failed" >&2
    failed=1
  fi
done
(( failed == 0 )) || exit 1

for axis in steps round; do
  python3 "$MAIN_REPO/scripts/plot_fedrl_heterogeneous.py" \
    --fed-log-dir "$FED_ROOT" \
    --paper-log-dir "" \
    --dqn-log-dir "" \
    --out-dir "$PLOT_ROOT/comparison_${axis}" \
    --plot-kind comparison \
    --x-axis "$axis" \
    --metric current \
    --variance ci95 \
    --smooth-window 7 \
    --style paper \
    --no-raw-traces \
    --align-start \
    --envs Swimmer-v5

  python3 "$MAIN_REPO/scripts/plot_fedrl_paper_panels.py" \
    --fed-log-dir "$FED_ROOT" \
    --paper-log-dir "" \
    --out-file "$PLOT_ROOT/paper_figures/swimmer_stability_${axis}_ci95.png" \
    --plot-kind comparison \
    --x-axis "$axis" \
    --metric current \
    --variance ci95 \
    --smooth-window 7 \
    --align-start \
    --envs Swimmer-v5
done

python3 "$MAIN_REPO/scripts/summarize_fedrl_results.py" \
  --fed-log-dir "$FED_ROOT" \
  --paper-log-dir "" \
  --dqn-log-dir "" \
  --out-dir "$PLOT_ROOT/tables" \
  --plot-kind comparison \
  --envs Swimmer-v5

echo "Paired Swimmer stability pilot complete: $PLOT_ROOT"

#!/usr/bin/env bash
# Three fixed seeds for the rollback-derived, variance-stabilized comparison.
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PATH="${CONDA_ENV_BIN:-$HOME/anaconda3/envs/dist-erl-re2/bin}:$PATH"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export MPLBACKEND=${MPLBACKEND:-Agg}
export RAY_local_fs_capacity_threshold=${RAY_local_fs_capacity_threshold:-0.99}

MAIN_REPO=${MAIN_REPO:-/home/ywj/code/Dist-ERL}
EXPERIMENT_ID=${EXPERIMENT_ID:-fedevosac_rollback_stabilized_3seed_20260910}
ENVS=${ENVS:-"Walker2d-v5 Hopper-v5 Swimmer-v5 HalfCheetah-v5"}
SEEDS=${SEEDS:-"0 1 2"}
TARGET_ENV_STEPS=${TARGET_ENV_STEPS:-1200000}
RUN_ROOT=${RUN_ROOT:-$MAIN_REPO/logs/experiments/$EXPERIMENT_ID}
PLOT_ROOT=${PLOT_ROOT:-$MAIN_REPO/plots_2/$EXPERIMENT_ID}
FED_ROOT="$RUN_ROOT/fedevosac"
BASELINE_ROOT="$RUN_ROOT/baselines"

mkdir -p "$FED_ROOT" "$BASELINE_ROOT" "$PLOT_ROOT"
cat > "$PLOT_ROOT/PROTOCOL.txt" <<EOF
Rollback base commit: 8891211
Stabilization branch: rollback-paper-multienv-3seed
Environments: $ENVS
Fixed training seeds: $SEEDS
Common validation seed base: 50000003
Target counted interactions: $TARGET_ENV_STEPS per method and seed
Metric: current deployable policy return
Uncertainty: two-sided 95% Student-t confidence interval
Views: communication rounds and counted environment interactions only
Ablations: disabled
EOF

pids=()
for seed in $SEEDS; do
  echo "Launch stabilized comparison seed $seed -> $RUN_ROOT/seed_${seed}.log"
  ENVS="$ENVS" \
  SEEDS="$seed" \
  REPEAT_ID="stable3" \
  FED_VARIANTS="full" \
  SAC_BASELINES="fedavg_sac fedbest_sac fedsoftmax_sac_noea fedmedian_sac" \
  BUDGET_PRESET="converged" \
  TARGET_ENV_STEPS="$TARGET_ENV_STEPS" \
  LOG_DIR="$FED_ROOT" \
  SAC_LOG_DIR="$BASELINE_ROOT" \
  SKIP_EXISTING=1 \
  SKIP_PLOTS=1 \
  bash run_continuous_fedevosac_suite.sh > "$RUN_ROOT/seed_${seed}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
(( failed == 0 )) || { echo "A seed failed; inspect $RUN_ROOT/seed_*.log" >&2; exit 1; }

for axis in round steps; do
  python3 "$MAIN_REPO/scripts/plot_fedrl_heterogeneous.py" \
    --fed-log-dir "$FED_ROOT" \
    --paper-log-dir "$BASELINE_ROOT" \
    --dqn-log-dir "" \
    --out-dir "$PLOT_ROOT/comparison_${axis}" \
    --plot-kind comparison \
    --x-axis "$axis" \
    --metric current \
    --variance ci95 \
    --smooth-window 7 \
    --style reference \
    --no-raw-traces \
    --align-start \
    --envs $ENVS

  python3 "$MAIN_REPO/scripts/plot_fedrl_paper_panels.py" \
    --fed-log-dir "$FED_ROOT" \
    --paper-log-dir "$BASELINE_ROOT" \
    --out-file "$PLOT_ROOT/paper_figures/comparison_${axis}.png" \
    --plot-kind comparison \
    --x-axis "$axis" \
    --metric current \
    --variance ci95 \
    --smooth-window 7 \
    --align-start \
    --envs $ENVS
done

python3 "$MAIN_REPO/scripts/summarize_fedrl_results.py" \
  --fed-log-dir "$FED_ROOT" \
  --paper-log-dir "$BASELINE_ROOT" \
  --dqn-log-dir "" \
  --out-dir "$PLOT_ROOT/tables" \
  --plot-kind comparison \
  --envs $ENVS

echo "Stabilized rollback comparison complete: $PLOT_ROOT"

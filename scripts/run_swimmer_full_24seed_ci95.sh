#!/usr/bin/env bash
# Estimate Swimmer FedEvoSAC training-seed uncertainty without rerunning stable tasks.
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PATH="${CONDA_ENV_BIN:-$HOME/anaconda3/envs/dist-erl-re2/bin}:$PATH"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export MPLBACKEND=${MPLBACKEND:-Agg}
export RAY_local_fs_capacity_threshold=${RAY_local_fs_capacity_threshold:-0.99}

MAIN_REPO=${MAIN_REPO:-/home/ywj/code/Dist-ERL}
EXPERIMENT_ID=${EXPERIMENT_ID:-fedevosac_swimmer_full_24seed_ci95_20260912}
SEEDS=${SEEDS:-"$(seq -s ' ' 0 23)"}
REUSE_SEEDS=${REUSE_SEEDS:-"0 1 2"}
MAX_PARALLEL=${MAX_PARALLEL:-6}
TARGET_ENV_STEPS=${TARGET_ENV_STEPS:-1200000}
SOURCE_ROOT=${SOURCE_ROOT:-$MAIN_REPO/logs/experiments/fedevosac_rollback_stabilized_3seed_v2_20260910/fedevosac}
RUN_ROOT=${RUN_ROOT:-$MAIN_REPO/logs/experiments/$EXPERIMENT_ID}
PLOT_ROOT=${PLOT_ROOT:-$MAIN_REPO/plots_2/$EXPERIMENT_ID}
FED_ROOT="$RUN_ROOT/fedevosac"
BASELINE_ROOT="$RUN_ROOT/baselines"
RUN_SUFFIX=swim24ci95

mkdir -p "$FED_ROOT" "$BASELINE_ROOT" "$PLOT_ROOT"
MANIFEST="$PLOT_ROOT/SOURCE_MANIFEST.csv"
printf 'environment,method,seed,origin,source_path\n' > "$MANIFEST"

is_reused_seed() {
  [[ " $REUSE_SEEDS " == *" $1 "* ]]
}

for seed in $REUSE_SEEDS; do
  source_dir="$SOURCE_ROOT/fedevosac_Swimmer-v5_full_s${seed}_rstable3v2"
  [[ -d "$source_dir" ]] || {
    echo "Missing reusable Swimmer run: $source_dir" >&2
    exit 1
  }
  dest="$FED_ROOT/fedevosac_Swimmer-v5_full_s${seed}_r${RUN_SUFFIX}"
  [[ -e "$dest" ]] || cp -al "$source_dir" "$dest"
  printf '%s,%s,%s,%s,%s\n' \
    Swimmer-v5 FedEvoSAC-full "$seed" reused-stable-v2 "$source_dir" >> "$MANIFEST"
done

cat > "$PLOT_ROOT/PROTOCOL.txt" <<EOF
Purpose: estimate FedEvoSAC-full Swimmer training-seed variance before further tuning
Training source commit: $(git rev-parse HEAD)
Plotting source commit: $(git -C "$MAIN_REPO" rev-parse HEAD)
Environment: Swimmer-v5 only
Method: FedEvoSAC-full only
Predetermined seeds: $SEEDS
Reused equivalent completed seeds: $REUSE_SEEDS
Target counted interactions: $TARGET_ENV_STEPS per seed
Metric: current deployable policy return
Uncertainty: two-sided 95% Student-t confidence interval
Views: counted environment interactions and communication rounds
Maximum concurrent seed jobs: $MAX_PARALLEL
No baseline, ablation, seed selection, Walker2d, Hopper, HalfCheetah, or Ant runs
EOF

run_seed() {
  local seed=$1
  echo "Launch Swimmer Full seed $seed -> $RUN_ROOT/seed_${seed}.log"
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
  if is_reused_seed "$seed"; then
    continue
  fi
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

for seed in $SEEDS; do
  metrics="$FED_ROOT/fedevosac_Swimmer-v5_full_s${seed}_r${RUN_SUFFIX}/metrics.csv"
  [[ -s "$metrics" ]] || { echo "Missing metrics for seed $seed" >&2; exit 1; }
  last_steps=$(awk -F, '
    NR == 1 { for (i = 1; i <= NF; i++) if ($i == "total_env_steps") col = i; next }
    col && $col != "" { value = $col + 0 }
    END { print value + 0 }
  ' "$metrics")
  (( last_steps >= TARGET_ENV_STEPS )) || {
    echo "Incomplete seed $seed: $last_steps < $TARGET_ENV_STEPS" >&2
    exit 1
  }
  if ! is_reused_seed "$seed"; then
    printf '%s,%s,%s,%s,%s\n' \
      Swimmer-v5 FedEvoSAC-full "$seed" new "$FED_ROOT" >> "$MANIFEST"
  fi
done

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
    --out-file "$PLOT_ROOT/paper_figures/swimmer_full_${axis}_ci95.png" \
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

echo "Swimmer 24-seed 95% CI audit complete: $PLOT_ROOT"

#!/usr/bin/env bash
# Preserve stable rollback Full runs and retrain only methods affected by fixes.
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PATH="${CONDA_ENV_BIN:-$HOME/anaconda3/envs/dist-erl-re2/bin}:$PATH"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export MPLBACKEND=${MPLBACKEND:-Agg}
export RAY_local_fs_capacity_threshold=${RAY_local_fs_capacity_threshold:-0.99}

MAIN_REPO=${MAIN_REPO:-/home/ywj/code/Dist-ERL}
EXPERIMENT_ID=${EXPERIMENT_ID:-fedevosac_rollback_stabilized_3seed_v2_20260910}
ENVS=${ENVS:-"Walker2d-v5 Hopper-v5 Swimmer-v5 HalfCheetah-v5"}
REUSED_FULL_ENVS=${REUSED_FULL_ENVS:-"Walker2d-v5 Hopper-v5"}
SEEDS=${SEEDS:-"0 1 2"}
TARGET_ENV_STEPS=${TARGET_ENV_STEPS:-1200000}
SOURCE_ROOT=${SOURCE_ROOT:-$MAIN_REPO/logs/experiments/fedevosac_20x2_converged_20260714}
RUN_ROOT=${RUN_ROOT:-$MAIN_REPO/logs/experiments/$EXPERIMENT_ID}
PLOT_ROOT=${PLOT_ROOT:-$MAIN_REPO/plots_2/$EXPERIMENT_ID}
FED_ROOT="$RUN_ROOT/fedevosac"
BASELINE_ROOT="$RUN_ROOT/baselines"
RUN_SUFFIX=stable3v2

mkdir -p "$FED_ROOT" "$BASELINE_ROOT" "$PLOT_ROOT"
MANIFEST="$PLOT_ROOT/SOURCE_MANIFEST.csv"
printf 'environment,method,seed,origin,source_path\n' > "$MANIFEST"

for env_name in $REUSED_FULL_ENVS; do
  for seed in $SEEDS; do
    source_dir=$(find "$SOURCE_ROOT/fedevosac" -mindepth 2 -maxdepth 2 -type d \
      -name "fedevosac_${env_name}_full_s${seed}_r*" -print -quit)
    [[ -n "$source_dir" ]] || { echo "Missing rollback Full run: $env_name seed $seed" >&2; exit 1; }
    dest="$FED_ROOT/fedevosac_${env_name}_full_s${seed}_r${RUN_SUFFIX}"
    [[ -e "$dest" ]] || cp -al "$source_dir" "$dest"
    printf '%s,%s,%s,%s,%s\n' \
      "$env_name" FedEvoSAC-full "$seed" reused-rollback "$source_dir" >> "$MANIFEST"
  done
done

cat > "$PLOT_ROOT/PROTOCOL.txt" <<EOF
Rollback base commit: 8891211
Stabilization branch source: $(git rev-parse HEAD)
Environments: $ENVS
Removed environment: Ant-v5
Fixed training seeds: $SEEDS
Target counted interactions: $TARGET_ENV_STEPS per method and seed
Walker2d/Hopper FedEvoSAC-full: reused stable rollback runs for seeds 0,1,2
Walker2d/Hopper baselines: retrained with corrected SAC update budget and validation
Swimmer/HalfCheetah: retrained for every method with variance-stabilized EA/SAC settings
Metric: current deployable policy return
Uncertainty: two-sided 95% Student-t confidence interval
Views: communication rounds and counted environment interactions only
Ablations: disabled
EOF

pids=()
for seed in $SEEDS; do
  echo "Launch environment-isolated stabilization seed $seed -> $RUN_ROOT/seed_${seed}.log"
  ENVS="$ENVS" \
  SEEDS="$seed" \
  REPEAT_ID="$RUN_SUFFIX" \
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

for env_name in $ENVS; do
  for seed in $SEEDS; do
    if [[ " $REUSED_FULL_ENVS " != *" $env_name "* ]]; then
      printf '%s,%s,%s,%s,%s\n' \
        "$env_name" FedEvoSAC-full "$seed" new-stabilized "$FED_ROOT" >> "$MANIFEST"
    fi
    for mode in fedavg_sac fedbest_sac fedsoftmax_sac_noea fedmedian_sac; do
      printf '%s,%s,%s,%s,%s\n' \
        "$env_name" "$mode" "$seed" new-stabilized "$BASELINE_ROOT" >> "$MANIFEST"
    done
  done
done

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

echo "Environment-isolated stabilization complete: $PLOT_ROOT"

#!/usr/bin/env bash
# Reuse fixed historical seeds for three environments and train only missing tasks.
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PATH="${CONDA_ENV_BIN:-$HOME/anaconda3/envs/dist-erl-re2/bin}:$PATH"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export MPLBACKEND=${MPLBACKEND:-Agg}
export RAY_local_fs_capacity_threshold=${RAY_local_fs_capacity_threshold:-0.99}

MAIN_REPO=${MAIN_REPO:-/home/ywj/code/Dist-ERL}
EXPERIMENT_ID=${EXPERIMENT_ID:-fedevosac_rollback_multienv_3seed_20260907}
SEEDS=${SEEDS:-"0 1 2"}
REUSED_ENVS=${REUSED_ENVS:-"Walker2d-v5 Hopper-v5 Swimmer-v5"}
NEW_ENVS=${NEW_ENVS:-"HalfCheetah-v5"}
ALL_ENVS=${ALL_ENVS:-"Walker2d-v5 Hopper-v5 Swimmer-v5 HalfCheetah-v5"}
SOURCE_ROOT=${SOURCE_ROOT:-$MAIN_REPO/logs/experiments/fedevosac_20x2_converged_20260714}
RUN_ROOT=${RUN_ROOT:-$MAIN_REPO/logs/experiments/$EXPERIMENT_ID}
PLOT_ROOT=${PLOT_ROOT:-$MAIN_REPO/plots_2/$EXPERIMENT_ID}
STAGE_ONLY=${STAGE_ONLY:-0}
FED_ROOT="$RUN_ROOT/fedevosac"
BASELINE_ROOT="$RUN_ROOT/baselines"

mkdir -p "$FED_ROOT" "$BASELINE_ROOT" "$PLOT_ROOT"

REFERENCE_SOURCE="$MAIN_REPO/plots_new/selected_best_converged_20260721"
REFERENCE_DEST="$PLOT_ROOT/historical_24seed_reference"
mkdir -p "$REFERENCE_DEST/comparison_round" "$REFERENCE_DEST/comparison_steps"
cp -a "$REFERENCE_SOURCE/comparison_round/." "$REFERENCE_DEST/comparison_round/"
cp -a "$REFERENCE_SOURCE/comparison_steps/." "$REFERENCE_DEST/comparison_steps/"
cp -a "$REFERENCE_SOURCE/README.md" "$REFERENCE_SOURCE/selection_manifest.csv" "$REFERENCE_DEST/"

MANIFEST="$PLOT_ROOT/SOURCE_MANIFEST.csv"
printf 'environment,method,seed,origin,source_path\n' > "$MANIFEST"

stage_run() {
  local source_dir=$1
  local dest_root=$2
  local dest="$dest_root/$(basename "$source_dir")"
  [[ -e "$dest" ]] || cp -al "$source_dir" "$dest"
}

for env_name in $REUSED_ENVS; do
  for seed in $SEEDS; do
    fed_match=$(find "$SOURCE_ROOT/fedevosac" -mindepth 2 -maxdepth 2 -type d \
      -name "fedevosac_${env_name}_full_s${seed}_r*" -print -quit)
    [[ -n "$fed_match" ]] || { echo "Missing historical Full run: $env_name seed $seed" >&2; exit 1; }
    stage_run "$fed_match" "$FED_ROOT"
    printf '%s,%s,%s,%s,%s\n' "$env_name" FedEvoSAC-full "$seed" reused "$fed_match" >> "$MANIFEST"

    for mode in fedavg_sac fedbest_sac fedsoftmax_sac_noea fedmedian_sac; do
      baseline_match=$(find "$SOURCE_ROOT/baselines" -mindepth 2 -maxdepth 2 -type d \
        -name "${mode}_${env_name}_s${seed}_r*" -print -quit)
      [[ -n "$baseline_match" ]] || { echo "Missing historical baseline: $mode $env_name seed $seed" >&2; exit 1; }
      stage_run "$baseline_match" "$BASELINE_ROOT"
      printf '%s,%s,%s,%s,%s\n' "$env_name" "$mode" "$seed" reused "$baseline_match" >> "$MANIFEST"
    done
  done
done

cat > "$PLOT_ROOT/PROTOCOL.txt" <<EOF
Rollback base commit: 8891211
Fixed seeds: $SEEDS
Reused environments: $REUSED_ENVS
Newly trained environments: $NEW_ENVS
Historical source: $SOURCE_ROOT
Metric: current deployable policy return
Uncertainty: two-sided 95% Student-t confidence interval
Views: communication rounds and counted environment interactions only
Status: exploratory post-hoc comparison; not a replacement for formal aggregate
EOF

render_comparison() {
  local envs=$1
  local out_root=$2
  for axis in round steps; do
    python3 "$MAIN_REPO/scripts/plot_fedrl_heterogeneous.py" \
      --fed-log-dir "$FED_ROOT" \
      --paper-log-dir "$BASELINE_ROOT" \
      --dqn-log-dir "" \
      --out-dir "$out_root/comparison_${axis}" \
      --plot-kind comparison \
      --x-axis "$axis" \
      --metric current \
      --variance ci95 \
      --smooth-window 7 \
      --style reference \
      --no-raw-traces \
      --align-start \
      --envs $envs

    python3 "$MAIN_REPO/scripts/plot_fedrl_paper_panels.py" \
      --fed-log-dir "$FED_ROOT" \
      --paper-log-dir "$BASELINE_ROOT" \
      --out-file "$out_root/paper_figures/comparison_${axis}.png" \
      --plot-kind comparison \
      --x-axis "$axis" \
      --metric current \
      --variance ci95 \
      --smooth-window 7 \
      --align-start \
      --envs $envs
  done
}

# Make the reused subset inspectable immediately, before the two new tasks run.
render_comparison "$REUSED_ENVS" "$PLOT_ROOT/reused_3seed"

if [[ "$STAGE_ONLY" == "1" ]]; then
  echo "Historical three-seed subset staged: $PLOT_ROOT/reused_3seed"
  exit 0
fi

pids=()
for seed in $SEEDS; do
  echo "Launch missing environments for seed $seed -> $RUN_ROOT/seed_${seed}.log"
  ENVS="$NEW_ENVS" \
  SEEDS="$seed" \
  REPEAT_ID="rollback3" \
  FED_VARIANTS="full" \
  SAC_BASELINES="fedavg_sac fedbest_sac fedsoftmax_sac_noea fedmedian_sac" \
  BUDGET_PRESET="converged" \
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

for env_name in $NEW_ENVS; do
  for seed in $SEEDS; do
    printf '%s,%s,%s,%s,%s\n' "$env_name" FedEvoSAC-full "$seed" new "$FED_ROOT" >> "$MANIFEST"
    for mode in fedavg_sac fedbest_sac fedsoftmax_sac_noea fedmedian_sac; do
      printf '%s,%s,%s,%s,%s\n' "$env_name" "$mode" "$seed" new "$BASELINE_ROOT" >> "$MANIFEST"
    done
  done
done

render_comparison "$ALL_ENVS" "$PLOT_ROOT"

python3 "$MAIN_REPO/scripts/summarize_fedrl_results.py" \
  --fed-log-dir "$FED_ROOT" \
  --paper-log-dir "$BASELINE_ROOT" \
  --dqn-log-dir "" \
  --out-dir "$PLOT_ROOT/tables" \
  --plot-kind comparison \
  --envs $ALL_ENVS

echo "Rollback multi-environment comparison complete: $PLOT_ROOT"

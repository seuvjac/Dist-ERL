#!/usr/bin/env bash
# Three-environment, five-method pilot before committing to high-seed experiments.
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PATH="${CONDA_ENV_BIN:-$HOME/anaconda3/envs/dist-erl-re2/bin}:$PATH"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export MPLBACKEND=${MPLBACKEND:-Agg}
export RAY_local_fs_capacity_threshold=${RAY_local_fs_capacity_threshold:-0.99}

MAIN_REPO=${MAIN_REPO:-/home/ywj/code/Dist-ERL}
EXPERIMENT_ID=${EXPERIMENT_ID:-fedevosac_candidate3_comparison_s012_20260912}
ENVS=${ENVS:-"InvertedPendulum-v5 Reacher-v5 HumanoidStandup-v5"}
SEEDS=${SEEDS:-"0 1 2"}
MAX_PARALLEL=${MAX_PARALLEL:-6}
RUN_ROOT=${RUN_ROOT:-$MAIN_REPO/logs/experiments/$EXPERIMENT_ID}
PLOT_ROOT=${PLOT_ROOT:-$MAIN_REPO/plots_2/$EXPERIMENT_ID}
FED_ROOT="$RUN_ROOT/fedevosac"
BASELINE_ROOT="$RUN_ROOT/baselines"
RUN_SUFFIX=candidate3s012

mkdir -p "$FED_ROOT" "$BASELINE_ROOT" "$PLOT_ROOT"
cat > "$PLOT_ROOT/PROTOCOL.txt" <<EOF
Purpose: assess three continuous-control replacements for HalfCheetah
Source commit: $(git rev-parse HEAD)
Environments: $ENVS
Methods: FedEvoSAC-full, FedAvg-SAC, FedBest-SAC, FedSoftmax-SAC-noEA,
RobustFed-SAC-Median
Predetermined seeds: $SEEDS
Budgets: InvertedPendulum=300k, Reacher=300k, HumanoidStandup=600k counted interactions
Clients: 3
Heterogeneity: dynamics only; no reward scaling
Metric/uncertainty: current deployable policy / two-sided 95% Student-t CI
Views: counted environment interactions and communication rounds; no progress view
Ablations: disabled
EOF

run_env_seed() {
  local env_name=$1
  local seed=$2
  local safe_env=${env_name//[^A-Za-z0-9]/_}
  echo "Launch $env_name seed $seed -> $RUN_ROOT/${safe_env}_seed_${seed}.log"
  ENVS="$env_name" \
  SEEDS="$seed" \
  REPEAT_ID="$RUN_SUFFIX" \
  FED_VARIANTS="full" \
  SAC_BASELINES="fedavg_sac fedbest_sac fedsoftmax_sac_noea fedmedian_sac" \
  BUDGET_PRESET="converged" \
  CLIENT_HETEROGENEITY="0.0" \
  CLIENT_HETEROGENEITY_MODE="none" \
  LOG_DIR="$FED_ROOT" \
  SAC_LOG_DIR="$BASELINE_ROOT" \
  SKIP_EXISTING=1 \
  SKIP_PLOTS=1 \
  bash run_continuous_fedevosac_suite.sh > "$RUN_ROOT/${safe_env}_seed_${seed}.log" 2>&1
}

pids=()
labels=()
failed=0
for env_name in $ENVS; do
  for seed in $SEEDS; do
    run_env_seed "$env_name" "$seed" &
    pids+=("$!")
    labels+=("$env_name/$seed")
    if (( ${#pids[@]} >= MAX_PARALLEL )); then
      for idx in "${!pids[@]}"; do
        if ! wait "${pids[$idx]}"; then
          echo "Candidate pilot ${labels[$idx]} failed" >&2
          failed=1
        fi
      done
      pids=()
      labels=()
    fi
  done
done
for idx in "${!pids[@]}"; do
  if ! wait "${pids[$idx]}"; then
    echo "Candidate pilot ${labels[$idx]} failed" >&2
    failed=1
  fi
done
(( failed == 0 )) || exit 1

for axis in steps round; do
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
    --style paper \
    --no-raw-traces \
    --align-start \
    --envs $ENVS

  python3 "$MAIN_REPO/scripts/plot_fedrl_paper_panels.py" \
    --fed-log-dir "$FED_ROOT" \
    --paper-log-dir "$BASELINE_ROOT" \
    --out-file "$PLOT_ROOT/paper_figures/candidate3_comparison_${axis}_ci95.png" \
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

python3 - "$PLOT_ROOT/tables/comparison_summary.csv" <<'PY'
import csv
import sys

path = sys.argv[1]
expected_envs = {'InvertedPendulum-v5', 'Reacher-v5', 'HumanoidStandup-v5'}
expected_methods = {
    'FedEvoSAC-full', 'FedAvg-SAC', 'FedBest-SAC',
    'FedSoftmax-SAC-noEA', 'RobustFed-SAC-Median',
}
with open(path, newline='', encoding='utf-8') as handle:
    rows = list(csv.DictReader(handle))
seen = {(row['env'], row['method']): int(row['n']) for row in rows}
missing = [
    (env, method)
    for env in expected_envs
    for method in expected_methods
    if seen.get((env, method)) != 3
]
if missing:
    raise SystemExit(f'Incomplete comparison bundle: {missing}')
print('Verified all 15 environment/method curves with n=3.')
PY

echo "Three-environment comparison pilot complete: $PLOT_ROOT"

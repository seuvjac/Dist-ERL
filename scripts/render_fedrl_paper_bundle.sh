#!/usr/bin/env bash
# Render paper-facing figures and diagnostics from one or many repeated runs.
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

FED_LOG_DIR=${FED_LOG_DIR:?FED_LOG_DIR is required}
SAC_LOG_DIR=${SAC_LOG_DIR:-""}
OUT_ROOT=${OUT_ROOT:?OUT_ROOT is required}
ENVS=${ENVS:-"Walker2d-v5 Hopper-v5 Ant-v5 HalfCheetah-v5 Swimmer-v5"}
VARIANCE=${PLOT_VARIANCE:-"ci95"}
SMOOTH_WINDOW=${PLOT_SMOOTH_WINDOW:-"7"}
INCLUDE_PROGRESS=${INCLUDE_PROGRESS:-"0"}

render_individual() {
  local axis=$1
  local out_dir=$2
  python3 scripts/plot_fedrl_heterogeneous.py \
    --fed-log-dir "$FED_LOG_DIR" \
    --paper-log-dir "$SAC_LOG_DIR" \
    --dqn-log-dir "" \
    --out-dir "$out_dir" \
    --plot-kind comparison \
    --x-axis "$axis" \
    --metric current \
    --variance "$VARIANCE" \
    --smooth-window "$SMOOTH_WINDOW" \
    --style reference \
    --no-raw-traces \
    --align-start \
    --envs $ENVS
}

render_individual round "$OUT_ROOT/main/comparison_round"
render_individual steps "$OUT_ROOT/supplement/comparison_steps"
if [[ "$INCLUDE_PROGRESS" == "1" ]]; then
  render_individual progress "$OUT_ROOT/diagnostics/comparison_progress"
fi

python3 scripts/plot_fedrl_heterogeneous.py \
  --fed-log-dir "$FED_LOG_DIR" \
  --paper-log-dir "" \
  --dqn-log-dir "" \
  --out-dir "$OUT_ROOT/ablation/round" \
  --plot-kind ablation \
  --x-axis round \
  --metric current \
  --variance "$VARIANCE" \
  --smooth-window "$SMOOTH_WINDOW" \
  --style reference \
  --no-raw-traces \
  --align-start \
  --envs $ENVS

mkdir -p "$OUT_ROOT/paper_figures" "$OUT_ROOT/tables"

for AXIS in round steps; do
  python3 scripts/plot_fedrl_paper_panels.py \
    --fed-log-dir "$FED_LOG_DIR" \
    --paper-log-dir "$SAC_LOG_DIR" \
    --out-file "$OUT_ROOT/paper_figures/comparison_${AXIS}.png" \
    --plot-kind comparison \
    --x-axis "$AXIS" \
    --metric current \
    --variance "$VARIANCE" \
    --smooth-window "$SMOOTH_WINDOW" \
    --align-start \
    --envs $ENVS
done

if [[ "$INCLUDE_PROGRESS" == "1" ]]; then
  python3 scripts/plot_fedrl_paper_panels.py \
    --fed-log-dir "$FED_LOG_DIR" \
    --paper-log-dir "$SAC_LOG_DIR" \
    --out-file "$OUT_ROOT/paper_figures/comparison_progress.png" \
    --plot-kind comparison \
    --x-axis progress \
    --metric current \
    --variance "$VARIANCE" \
    --smooth-window "$SMOOTH_WINDOW" \
    --align-start \
    --envs $ENVS
fi

python3 scripts/plot_fedrl_paper_panels.py \
  --fed-log-dir "$FED_LOG_DIR" \
  --paper-log-dir "" \
  --out-file "$OUT_ROOT/paper_figures/ablation_round.png" \
  --plot-kind ablation \
  --x-axis round \
  --metric current \
  --variance "$VARIANCE" \
  --smooth-window "$SMOOTH_WINDOW" \
  --align-start \
  --envs $ENVS

python3 scripts/summarize_fedrl_results.py \
  --fed-log-dir "$FED_LOG_DIR" \
  --paper-log-dir "$SAC_LOG_DIR" \
  --dqn-log-dir "" \
  --out-dir "$OUT_ROOT/tables" \
  --plot-kind comparison \
  --envs $ENVS

python3 scripts/summarize_fedrl_results.py \
  --fed-log-dir "$FED_LOG_DIR" \
  --paper-log-dir "" \
  --dqn-log-dir "" \
  --out-dir "$OUT_ROOT/tables" \
  --plot-kind ablation \
  --envs $ENVS

LOG_DIRS=("$FED_LOG_DIR")
if [[ -n "$SAC_LOG_DIR" ]]; then
  LOG_DIRS+=("$SAC_LOG_DIR")
fi
python3 scripts/check_fedrl_convergence.py \
  --log-dirs "${LOG_DIRS[@]}" \
  --out-file "$OUT_ROOT/tables/convergence_report.csv"

python3 scripts/test_fedrl_significance.py \
  --log-dirs "${LOG_DIRS[@]}" \
  --out-file "$OUT_ROOT/tables/comparison_wilcoxon.csv" \
  --plot-kind comparison \
  --envs $ENVS

python3 scripts/test_fedrl_significance.py \
  --log-dirs "$FED_LOG_DIR" \
  --out-file "$OUT_ROOT/tables/ablation_wilcoxon.csv" \
  --plot-kind ablation \
  --envs $ENVS

echo "Rendered paper bundle under $OUT_ROOT"

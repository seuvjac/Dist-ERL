#!/usr/bin/env bash
# Render the paper-style formal environment figure set.
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

RUN_ID=${RUN_ID:-"perenv_tuned_3seed_20260711"}
FED_LOG_DIR=${FED_LOG_DIR:-"logs/logs_fedevosac_${RUN_ID}"}
SAC_LOG_DIR=${SAC_LOG_DIR:-"logs/logs_sac_${RUN_ID}"}
OUT_PREFIX=${OUT_PREFIX:-"plots/fedevosac_${RUN_ID}"}
ENVS=${ENVS:-"Walker2d-v5 Hopper-v5"}
VARIANCE=${PLOT_VARIANCE:-"seed"}
SMOOTH_WINDOW=${PLOT_SMOOTH_WINDOW:-"5"}
STYLE=${PLOT_STYLE:-"reference"}

for AXIS in steps progress round; do
  python3 scripts/plot_fedrl_heterogeneous.py \
    --fed-log-dir "$FED_LOG_DIR" \
    --paper-log-dir "$SAC_LOG_DIR" \
    --dqn-log-dir "" \
    --out-dir "${OUT_PREFIX}_comparison_${AXIS}" \
    --plot-kind comparison \
    --x-axis "$AXIS" \
    --metric current \
    --variance "$VARIANCE" \
    --smooth-window "$SMOOTH_WINDOW" \
    --style "$STYLE" \
    --envs $ENVS
done

python3 scripts/plot_fedrl_heterogeneous.py \
  --fed-log-dir "$FED_LOG_DIR" \
  --paper-log-dir "" \
  --dqn-log-dir "" \
  --out-dir "${OUT_PREFIX}_ablations_round" \
  --plot-kind ablation \
  --x-axis round \
  --metric current \
  --variance "$VARIANCE" \
  --smooth-window "$SMOOTH_WINDOW" \
  --style "$STYLE" \
  --envs $ENVS

python3 scripts/summarize_fedrl_results.py \
  --fed-log-dir "$FED_LOG_DIR" \
  --paper-log-dir "$SAC_LOG_DIR" \
  --dqn-log-dir "" \
  --out-dir "${OUT_PREFIX}_tables" \
  --plot-kind comparison \
  --envs $ENVS

python3 scripts/summarize_fedrl_results.py \
  --fed-log-dir "$FED_LOG_DIR" \
  --paper-log-dir "" \
  --dqn-log-dir "" \
  --out-dir "${OUT_PREFIX}_ablation_tables" \
  --plot-kind ablation \
  --envs $ENVS

echo "Rendered 12 main plots under ${OUT_PREFIX}_{comparison_steps,comparison_progress,comparison_round,ablations_round}"

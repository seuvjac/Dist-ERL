#!/usr/bin/env bash
# Pre-registered five-seed heterogeneity sensitivity analysis.
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export MPLBACKEND=${MPLBACKEND:-Agg}
export RAY_local_fs_capacity_threshold=${RAY_local_fs_capacity_threshold:-0.99}

EXPERIMENT_ID=${EXPERIMENT_ID:-"fedevosac_sensitivity_5seed_$(date +%Y%m%d)"}
ENVS=${ENVS:-"Walker2d-v5 Hopper-v5 Ant-v5 HalfCheetah-v5 Swimmer-v5"}
SEEDS=${SEEDS:-"200 201 202 203 204"}
LEVELS=${LEVELS:-"0.0 0.075 0.15 0.225 0.30"}
TARGET_ENV_STEPS=${TARGET_ENV_STEPS:-1200000}
RUN_ROOT=${RUN_ROOT:-"logs/experiments/$EXPERIMENT_ID"}
PLOT_ROOT=${PLOT_ROOT:-"plots_2/$EXPERIMENT_ID"}

mkdir -p "$RUN_ROOT" "$PLOT_ROOT"
cat > "$PLOT_ROOT/PROTOCOL.txt" <<EOF
Sensitivity factor: client dynamics heterogeneity strength.
Levels fixed before execution: $LEVELS.
Seeds fixed before execution: $SEEDS.
All levels use FedEvoSAC-full and $TARGET_ENV_STEPS counted interactions.
Error bars are two-sided 95% Student-t confidence intervals.
This analysis is separate from component ablation and main-method comparison.
EOF

for level in $LEVELS; do
  tag=${level//./p}
  echo "Sensitivity level $level"
  FED_WALKER2D_CLIENT_HETEROGENEITY="$level" \
  FED_HOPPER_CLIENT_HETEROGENEITY="$level" \
  FED_ANT_CLIENT_HETEROGENEITY="$level" \
  FED_HALFCHEETAH_CLIENT_HETEROGENEITY="$level" \
  FED_SWIMMER_CLIENT_HETEROGENEITY="$level" \
  ENVS="$ENVS" \
  SEEDS="$SEEDS" \
  FED_VARIANTS="full" \
  SAC_BASELINES="" \
  BUDGET_PRESET=converged \
  TARGET_ENV_STEPS="$TARGET_ENV_STEPS" \
  LOG_DIR="$RUN_ROOT/level_$tag" \
  SAC_LOG_DIR="$RUN_ROOT/unused_baselines" \
  SKIP_EXISTING=1 \
  SKIP_PLOTS=1 \
  bash run_continuous_fedevosac_suite.sh
done

python3 scripts/plot_fedrl_sensitivity.py \
  --log-dir "$RUN_ROOT" \
  --out-dir "$PLOT_ROOT/heterogeneity" \
  --envs $ENVS

echo "Sensitivity results: $PLOT_ROOT/heterogeneity"

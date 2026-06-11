#!/usr/bin/env bash
# Quick continuous-control pilot: 3 workers, single seed, no client heterogeneity.
set -euo pipefail

cd "$(dirname "$0")"

ENVS=${ENVS:-"BipedalWalker-v3"}
SEEDS=${SEEDS:-"0"}
FED_NUM_CLIENTS=${FED_NUM_CLIENTS:-"3"}
FED_NUM_WORKERS=${FED_NUM_WORKERS:-"3"}
CLIENT_HETEROGENEITY=${CLIENT_HETEROGENEITY:-"0.0"}
CLIENT_HETEROGENEITY_MODE=${CLIENT_HETEROGENEITY_MODE:-"none"}
BUDGET_PRESET=${BUDGET_PRESET:-"reduced"}

LOG_DIR=${LOG_DIR:-"logs_fedevosac_continuous_nohetero_pilot"}
SAC_LOG_DIR=${SAC_LOG_DIR:-"logs_sac_continuous_nohetero_pilot"}
COMPARISON_OUT_DIR=${COMPARISON_OUT_DIR:-"plots/fedevosac_continuous_nohetero_pilot"}
ABLATION_OUT_DIR=${ABLATION_OUT_DIR:-"plots/fedevosac_continuous_nohetero_pilot_ablations"}
SUMMARY_OUT_DIR=${SUMMARY_OUT_DIR:-"plots/fedevosac_continuous_nohetero_pilot_tables"}
PLOT_X_AXIS=${PLOT_X_AXIS:-"round"}
PLOT_METRIC=${PLOT_METRIC:-"current"}

export ENVS SEEDS FED_NUM_CLIENTS FED_NUM_WORKERS CLIENT_HETEROGENEITY CLIENT_HETEROGENEITY_MODE
export BUDGET_PRESET LOG_DIR SAC_LOG_DIR COMPARISON_OUT_DIR ABLATION_OUT_DIR SUMMARY_OUT_DIR
export PLOT_X_AXIS PLOT_METRIC

bash ./run_continuous_fedevosac_suite.sh

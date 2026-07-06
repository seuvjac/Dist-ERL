#!/usr/bin/env bash
# Continuous FedEvoSAC benchmark suite.
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

ENVS=${ENVS:-"Swimmer-v5 Walker2d-v5 Hopper-v5"}
SEEDS=${SEEDS:-"0"}
FED_VARIANTS=${FED_VARIANTS:-"full"}
SAC_BASELINES=${SAC_BASELINES:-"fedavg_sac fedbest_sac fedsoftmax_sac_noea fedmedian_sac"}
LOG_DIR=${LOG_DIR:-"logs/logs_fedevosac_continuous_mixed"}
SAC_LOG_DIR=${SAC_LOG_DIR:-"logs/logs_sac_continuous_mixed"}
COMPARISON_OUT_DIR=${COMPARISON_OUT_DIR:-"plots/fedevosac_continuous_comparison_round"}
CANDIDATE_OUT_DIR=${CANDIDATE_OUT_DIR:-"plots/fedevosac_continuous_candidate_round"}
ABLATION_OUT_DIR=${ABLATION_OUT_DIR:-"plots/fedevosac_continuous_ablations_round"}
SUMMARY_OUT_DIR=${SUMMARY_OUT_DIR:-"plots/fedevosac_continuous_tables"}
PLOT_X_AXIS=${PLOT_X_AXIS:-"round"}
PLOT_METRIC=${PLOT_METRIC:-"current"}
PLOT_VARIANCE=${PLOT_VARIANCE:-"none"}
PLOT_SMOOTH_WINDOW=${PLOT_SMOOTH_WINDOW:-"7"}
CLIENT_HETEROGENEITY=${CLIENT_HETEROGENEITY:-"0.0"}
CLIENT_HETEROGENEITY_MODE=${CLIENT_HETEROGENEITY_MODE:-"none"}
BUDGET_PRESET=${BUDGET_PRESET:-"reduced"}
TARGET_ENV_STEPS=${TARGET_ENV_STEPS:-"459000"}
BASELINE_UPDATE_TO_DATA_RATIO=${BASELINE_UPDATE_TO_DATA_RATIO:-"0.05"}
BASELINE_MAX_UPDATES_PER_ROUND=${BASELINE_MAX_UPDATES_PER_ROUND:-"20"}
BASELINE_BASE_UPDATES=${BASELINE_BASE_UPDATES:-"4"}
BASELINE_AGGREGATION_INTERVAL=${BASELINE_AGGREGATION_INTERVAL:-"5"}
BASELINE_SERVER_LEARNING_RATE=${BASELINE_SERVER_LEARNING_RATE:-"0.25"}
FED_INJECT_MARGIN=${FED_INJECT_MARGIN:-"-0.05"}
FED_EA_MUTATION_PROB=${FED_EA_MUTATION_PROB:-"0.90"}
FED_EA_RESET_PROB=${FED_EA_RESET_PROB:-"0.05"}
FED_DELTA_CLIP_NORM=${FED_DELTA_CLIP_NORM:-"5"}
FED_AGGREGATION_TEMPERATURE=${FED_AGGREGATION_TEMPERATURE:-"75"}
FED_SCORE_NORMALIZATION=${FED_SCORE_NORMALIZATION:-"relative_gain"}
FED_SCORE_EMA_BETA=${FED_SCORE_EMA_BETA:-"0.90"}
FED_SCORE_MIN_STD=${FED_SCORE_MIN_STD:-"1.0"}
FED_SCORE_WARMUP_ROUNDS=${FED_SCORE_WARMUP_ROUNDS:-"0"}
FED_SCORE_WARMUP_NORMALIZATION=${FED_SCORE_WARMUP_NORMALIZATION:-"batch_zscore"}
FED_INJECTION_WARMUP_ROUNDS=${FED_INJECTION_WARMUP_ROUNDS:-"0"}

for ENV_NAME in $ENVS; do
  read POP CLIENTS GENS STEPS <<<"$(python3 - <<PY
from src.config import env_run_preset
p = env_run_preset('$ENV_NAME')
print(p['population_size'], p['num_workers'], p['max_generations'], p['max_episode_steps'])
PY
)"
  if [[ "$BUDGET_PRESET" == "reduced" ]]; then
    POP=${FED_POPULATION_SIZE:-8}
    GENS=${FED_MAX_GENERATIONS:-$(( GENS < 15 ? GENS : 15 ))}
    EVAL_EPISODES=${FED_EVAL_EPISODES:-2}
    ARCHIVE_EVAL_CANDIDATES=${FED_ARCHIVE_EVAL_CANDIDATES:-2}
    ARCHIVE_EVAL_EPISODES=${FED_ARCHIVE_EVAL_EPISODES:-2}
  else
    POP=${FED_POPULATION_SIZE:-$POP}
    GENS=${FED_MAX_GENERATIONS:-$GENS}
    EVAL_EPISODES=${FED_EVAL_EPISODES:-3}
    CLIENT_ROLLOUTS=${FED_CLIENT_ROLLOUTS:-2}
    CLIENT_UPDATES=${FED_CLIENT_UPDATES:-8}
    ARCHIVE_EVAL_CANDIDATES=${FED_ARCHIVE_EVAL_CANDIDATES:-3}
    ARCHIVE_EVAL_EPISODES=${FED_ARCHIVE_EVAL_EPISODES:-3}
  fi
  SCORE_WARMUP_ROUNDS="$FED_SCORE_WARMUP_ROUNDS"
  SCORE_WARMUP_NORMALIZATION="$FED_SCORE_WARMUP_NORMALIZATION"
  INJECTION_WARMUP_ROUNDS="$FED_INJECTION_WARMUP_ROUNDS"
  EA_MUTATION_PROB="$FED_EA_MUTATION_PROB"
  EA_RESET_PROB="$FED_EA_RESET_PROB"
  INJECT_MARGIN="$FED_INJECT_MARGIN"
  DELTA_CLIP_NORM="$FED_DELTA_CLIP_NORM"
  AGGREGATION_TEMPERATURE="$FED_AGGREGATION_TEMPERATURE"
  case "$ENV_NAME" in
    Reacher-v5)
      STEPS=${FED_MAX_EPISODE_STEPS:-50}
      FED_AGG_INTERVAL=${FED_AGGREGATION_INTERVAL:-1}
      CLIENT_ROLLOUTS=${FED_CLIENT_ROLLOUTS:-4}
      CLIENT_UPDATES=${FED_CLIENT_UPDATES:-12}
      CLIENT_CRITIC_WARMUP=${FED_CLIENT_CRITIC_WARMUP:-6}
      ;;
    Swimmer-v5)
      POP=${FED_POPULATION_SIZE:-10}
      STEPS=${FED_MAX_EPISODE_STEPS:-1000}
      FED_AGG_INTERVAL=${FED_AGGREGATION_INTERVAL:-5}
      CLIENT_ROLLOUTS=${FED_CLIENT_ROLLOUTS:-1}
      CLIENT_UPDATES=${FED_CLIENT_UPDATES:-4}
      CLIENT_CRITIC_WARMUP=${FED_CLIENT_CRITIC_WARMUP:-2}
      ARCHIVE_EVAL_CANDIDATES=${FED_ARCHIVE_EVAL_CANDIDATES:-2}
      ARCHIVE_EVAL_EPISODES=${FED_ARCHIVE_EVAL_EPISODES:-1}
      SCORE_WARMUP_ROUNDS=${FED_SWIMMER_SCORE_WARMUP_ROUNDS:-2}
      SCORE_WARMUP_NORMALIZATION=${FED_SWIMMER_SCORE_WARMUP_NORMALIZATION:-batch_zscore}
      INJECTION_WARMUP_ROUNDS=${FED_SWIMMER_INJECTION_WARMUP_ROUNDS:-2}
      ;;
    Hopper-v5)
      POP=${FED_HOPPER_POPULATION_SIZE:-${FED_POPULATION_SIZE:-12}}
      STEPS=${FED_MAX_EPISODE_STEPS:-1000}
      FED_AGG_INTERVAL=${FED_AGGREGATION_INTERVAL:-4}
      CLIENT_ROLLOUTS=${FED_CLIENT_ROLLOUTS:-2}
      CLIENT_UPDATES=${FED_CLIENT_UPDATES:-8}
      CLIENT_CRITIC_WARMUP=${FED_CLIENT_CRITIC_WARMUP:-4}
      ARCHIVE_EVAL_CANDIDATES=${FED_ARCHIVE_EVAL_CANDIDATES:-3}
      ARCHIVE_EVAL_EPISODES=${FED_ARCHIVE_EVAL_EPISODES:-2}
      EA_MUTATION_PROB=${FED_HOPPER_EA_MUTATION_PROB:-0.82}
      EA_RESET_PROB=${FED_HOPPER_EA_RESET_PROB:-0.02}
      INJECT_MARGIN=${FED_HOPPER_INJECT_MARGIN:-"-0.02"}
      DELTA_CLIP_NORM=${FED_HOPPER_DELTA_CLIP_NORM:-4}
      AGGREGATION_TEMPERATURE=${FED_HOPPER_AGGREGATION_TEMPERATURE:-50}
      ;;
    Walker2d-v5)
      POP=${FED_WALKER2D_POPULATION_SIZE:-${FED_POPULATION_SIZE:-12}}
      STEPS=${FED_MAX_EPISODE_STEPS:-1000}
      FED_AGG_INTERVAL=${FED_WALKER2D_AGGREGATION_INTERVAL:-5}
      CLIENT_ROLLOUTS=${FED_WALKER2D_CLIENT_ROLLOUTS:-1}
      CLIENT_UPDATES=${FED_WALKER2D_CLIENT_UPDATES:-6}
      CLIENT_CRITIC_WARMUP=${FED_WALKER2D_CLIENT_CRITIC_WARMUP:-3}
      ARCHIVE_EVAL_CANDIDATES=${FED_WALKER2D_ARCHIVE_EVAL_CANDIDATES:-2}
      ARCHIVE_EVAL_EPISODES=${FED_WALKER2D_ARCHIVE_EVAL_EPISODES:-1}
      EA_MUTATION_PROB=${FED_WALKER2D_EA_MUTATION_PROB:-0.86}
      EA_RESET_PROB=${FED_WALKER2D_EA_RESET_PROB:-0.03}
      INJECT_MARGIN=${FED_WALKER2D_INJECT_MARGIN:-"-0.04"}
      DELTA_CLIP_NORM=${FED_WALKER2D_DELTA_CLIP_NORM:-4}
      AGGREGATION_TEMPERATURE=${FED_WALKER2D_AGGREGATION_TEMPERATURE:-60}
      ;;
  esac
  CLIENTS=${FED_NUM_CLIENTS:-3}
  WORKERS=${FED_NUM_WORKERS:-$CLIENTS}
  TARGET_STEPS=$TARGET_ENV_STEPS
  if [[ -n "${BASELINE_ROUNDS:-}" ]]; then
    THIS_BASELINE_ROUNDS="$BASELINE_ROUNDS"
  else
    THIS_BASELINE_ROUNDS=$(python3 - <<PY
import math
print(max(1, math.ceil(int('$TARGET_STEPS') / max(1, int('$CLIENTS') * int('$STEPS')))))
PY
)
  fi
  echo "$ENV_NAME continuous budget=$BUDGET_PRESET target_env_steps=$TARGET_STEPS baseline_rounds>=$THIS_BASELINE_ROUNDS"

  for SEED in $SEEDS; do
    for VARIANT in $FED_VARIANTS; do
      EXP="fedevosac_${ENV_NAME}_${VARIANT}_s${SEED}"
      python3 -m src.main \
        --env "$ENV_NAME" \
        --mode fed_evo_rl \
        --fed-ablation "$VARIANT" \
        --algorithm SAC \
        --population-size "$POP" \
        --num-workers "$WORKERS" \
        --num-clients "$CLIENTS" \
        --max-generations "$GENS" \
        --target-env-steps "$TARGET_STEPS" \
        --max-episode-steps "$STEPS" \
        --client-heterogeneity "$CLIENT_HETEROGENEITY" \
        --client-heterogeneity-mode "$CLIENT_HETEROGENEITY_MODE" \
        --fed-aggregation softmax \
        --fed-aggregation-interval "$FED_AGG_INTERVAL" \
        --fed-aggregation-temperature "$AGGREGATION_TEMPERATURE" \
        --fed-score-normalization "$FED_SCORE_NORMALIZATION" \
        --fed-score-warmup-rounds "$SCORE_WARMUP_ROUNDS" \
        --fed-score-warmup-normalization "$SCORE_WARMUP_NORMALIZATION" \
        --fed-score-ema-beta "$FED_SCORE_EMA_BETA" \
        --fed-score-min-std "$FED_SCORE_MIN_STD" \
        --fed-injection-warmup-rounds "$INJECTION_WARMUP_ROUNDS" \
        --fed-inject-margin "$INJECT_MARGIN" \
        --fed-delta-clip-norm "$DELTA_CLIP_NORM" \
        --ea-mutation-prob "$EA_MUTATION_PROB" \
        --ea-prob-reset-and-super "$EA_RESET_PROB" \
        --ea-weight-clip 5 \
        --elite-archive-size 5 \
        --elite-archive-restore-copies 1 \
        --archive-eval-candidates "$ARCHIVE_EVAL_CANDIDATES" \
        --archive-eval-episodes "$ARCHIVE_EVAL_EPISODES" \
        --client-rollouts "$CLIENT_ROLLOUTS" \
        --client-updates "$CLIENT_UPDATES" \
        --client-critic-warmup-updates "$CLIENT_CRITIC_WARMUP" \
        --batch-size 64 \
        --eval-episodes "$EVAL_EPISODES" \
        --seed "$SEED" \
        --log-dir "$LOG_DIR" \
        --exp-name "$EXP"
    done
    for MODE in $SAC_BASELINES; do
      EXP="${MODE}_${ENV_NAME}_s${SEED}"
      python3 scripts/train_continuous_sac_baseline.py \
        --env "$ENV_NAME" \
        --seed "$SEED" \
        --rounds "$THIS_BASELINE_ROUNDS" \
        --target-env-steps "$TARGET_STEPS" \
        --num-workers "$CLIENTS" \
        --updates "$BASELINE_BASE_UPDATES" \
        --update-to-data-ratio "$BASELINE_UPDATE_TO_DATA_RATIO" \
        --max-updates-per-round "$BASELINE_MAX_UPDATES_PER_ROUND" \
        --aggregation-interval "$BASELINE_AGGREGATION_INTERVAL" \
        --server-learning-rate "$BASELINE_SERVER_LEARNING_RATE" \
        --aggregation-eval-episodes "$ARCHIVE_EVAL_EPISODES" \
        --eval-interval "$BASELINE_AGGREGATION_INTERVAL" \
        --batch-size 64 \
        --eval-episodes "$EVAL_EPISODES" \
        --max-episode-steps "$STEPS" \
        --client-heterogeneity "$CLIENT_HETEROGENEITY" \
        --client-heterogeneity-mode "$CLIENT_HETEROGENEITY_MODE" \
        --log-dir "$SAC_LOG_DIR" \
        --exp-name "$EXP" \
        --baseline-mode "$MODE"
    done
  done
done

python3 scripts/plot_fedrl_heterogeneous.py \
  --fed-log-dir "$LOG_DIR" \
  --paper-log-dir "$SAC_LOG_DIR" \
  --dqn-log-dir "" \
  --out-dir "$COMPARISON_OUT_DIR" \
  --plot-kind comparison \
  --x-axis "$PLOT_X_AXIS" \
  --metric "$PLOT_METRIC" \
  --variance "$PLOT_VARIANCE" \
  --smooth-window "$PLOT_SMOOTH_WINDOW"

python3 scripts/plot_fedrl_heterogeneous.py \
  --fed-log-dir "$LOG_DIR" \
  --paper-log-dir "$SAC_LOG_DIR" \
  --dqn-log-dir "" \
  --out-dir "$CANDIDATE_OUT_DIR" \
  --plot-kind comparison \
  --x-axis "$PLOT_X_AXIS" \
  --metric candidate \
  --variance "$PLOT_VARIANCE" \
  --smooth-window "$PLOT_SMOOTH_WINDOW"

python3 scripts/plot_fedrl_heterogeneous.py \
  --fed-log-dir "$LOG_DIR" \
  --paper-log-dir "" \
  --dqn-log-dir "" \
  --out-dir "$ABLATION_OUT_DIR" \
  --plot-kind ablation \
  --x-axis "$PLOT_X_AXIS" \
  --metric "$PLOT_METRIC" \
  --variance "$PLOT_VARIANCE" \
  --smooth-window "$PLOT_SMOOTH_WINDOW"

python3 scripts/summarize_fedrl_results.py \
  --fed-log-dir "$LOG_DIR" \
  --paper-log-dir "$SAC_LOG_DIR" \
  --dqn-log-dir "" \
  --out-dir "$SUMMARY_OUT_DIR" \
  --plot-kind comparison \
  --envs $ENVS

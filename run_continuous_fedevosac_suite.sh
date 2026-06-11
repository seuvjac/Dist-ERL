#!/usr/bin/env bash
# Continuous FedEvoSAC benchmark suite.
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

ENVS=${ENVS:-"MountainCarContinuous-v0 LunarLanderContinuous-v3 HalfCheetah-v5"}
SEEDS=${SEEDS:-"0 1 2"}
FED_VARIANTS=${FED_VARIANTS:-"full"}
SAC_BASELINES=${SAC_BASELINES:-"fedavg_sac fedbest_sac fedsoftmax_sac_noea fedmedian_sac"}
LOG_DIR=${LOG_DIR:-"logs_fedevosac_continuous_mixed"}
SAC_LOG_DIR=${SAC_LOG_DIR:-"logs_sac_continuous_mixed"}
COMPARISON_OUT_DIR=${COMPARISON_OUT_DIR:-"plots/fedevosac_continuous_comparison_round"}
ABLATION_OUT_DIR=${ABLATION_OUT_DIR:-"plots/fedevosac_continuous_ablations_round"}
SUMMARY_OUT_DIR=${SUMMARY_OUT_DIR:-"plots/fedevosac_continuous_tables"}
PLOT_X_AXIS=${PLOT_X_AXIS:-"round"}
PLOT_METRIC=${PLOT_METRIC:-"current"}
CLIENT_HETEROGENEITY=${CLIENT_HETEROGENEITY:-"0.60"}
CLIENT_HETEROGENEITY_MODE=${CLIENT_HETEROGENEITY_MODE:-"mixed"}
BUDGET_PRESET=${BUDGET_PRESET:-"reduced"}

for ENV_NAME in $ENVS; do
  read POP CLIENTS GENS STEPS <<<"$(python3 - <<PY
from src.config import env_run_preset
p = env_run_preset('$ENV_NAME')
print(p['population_size'], p['num_workers'], p['max_generations'], p['max_episode_steps'])
PY
)"
  if [[ "$BUDGET_PRESET" == "reduced" ]]; then
    POP=${FED_POPULATION_SIZE:-$(( POP < 10 ? POP : 10 ))}
    GENS=${FED_MAX_GENERATIONS:-$(( GENS < 15 ? GENS : 15 ))}
    EVAL_EPISODES=${FED_EVAL_EPISODES:-2}
    CLIENT_ROLLOUTS=${FED_CLIENT_ROLLOUTS:-1}
    CLIENT_UPDATES=${FED_CLIENT_UPDATES:-4}
  else
    POP=${FED_POPULATION_SIZE:-$POP}
    GENS=${FED_MAX_GENERATIONS:-$GENS}
    EVAL_EPISODES=${FED_EVAL_EPISODES:-3}
    CLIENT_ROLLOUTS=${FED_CLIENT_ROLLOUTS:-2}
    CLIENT_UPDATES=${FED_CLIENT_UPDATES:-8}
  fi
  CLIENTS=${FED_NUM_CLIENTS:-4}
  WORKERS=${FED_NUM_WORKERS:-$CLIENTS}
  TARGET_STEPS=${TARGET_ENV_STEPS:-$(python3 - <<PY
pop=int('$POP'); clients=int('$CLIENTS'); gens=int('$GENS'); steps=int('$STEPS')
eval_episodes=int('$EVAL_EPISODES'); rollouts=int('$CLIENT_ROLLOUTS')
interval=5
eval_half=max(1, eval_episodes//2)
fed_rounds=sum(1 for g in range(gens) if g % interval == 0)
print(gens*pop*clients*steps*eval_half + fed_rounds*clients*rollouts*steps)
PY
)}
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
        --fed-aggregation-interval 5 \
        --fed-aggregation-temperature 75 \
        --fed-delta-clip-norm 5 \
        --ea-weight-clip 5 \
        --elite-archive-size 5 \
        --elite-archive-restore-copies 1 \
        --client-rollouts "$CLIENT_ROLLOUTS" \
        --client-updates "$CLIENT_UPDATES" \
        --batch-size 128 \
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
        --updates "$CLIENT_UPDATES" \
        --batch-size 128 \
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
  --metric "$PLOT_METRIC"

python3 scripts/plot_fedrl_heterogeneous.py \
  --fed-log-dir "$LOG_DIR" \
  --paper-log-dir "" \
  --dqn-log-dir "" \
  --out-dir "$ABLATION_OUT_DIR" \
  --plot-kind ablation \
  --x-axis "$PLOT_X_AXIS" \
  --metric "$PLOT_METRIC"

python3 scripts/summarize_fedrl_results.py \
  --fed-log-dir "$LOG_DIR" \
  --paper-log-dir "$SAC_LOG_DIR" \
  --dqn-log-dir "" \
  --out-dir "$SUMMARY_OUT_DIR" \
  --plot-kind comparison \
  --envs $ENVS

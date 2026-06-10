#!/usr/bin/env bash
# FedEvoFSAC runs for the three SAC/FSAC comparison environments.
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

ENVS=${ENVS:-"CartPole-v1 Acrobot-v1 LunarLander-v3"}
SEEDS=${SEEDS:-"0"}
FED_VARIANTS=${FED_VARIANTS:-"full uniform_aggregation no_local_rl no_ea_injection no_heterogeneity"}
PAPER_MODES=${PAPER_MODES:-"paper_sac paper_fsac fedavg_sac fedsoftmax_sac_noea fedbest_sac fedmedian_sac fedtrimmedmean_sac attention_sac_lite"}
EVO_BASELINES=${EVO_BASELINES:-""}
DQN_BASELINES=${DQN_BASELINES:-"fedavg_dqn"}
LOG_DIR=${LOG_DIR:-"logs_fedrl_hetero_mixed"}
PAPER_LOG_DIR=${PAPER_LOG_DIR:-"logs_fsac_paper_mixed"}
DQN_LOG_DIR=${DQN_LOG_DIR:-"logs_dqn_fedrl_mixed"}
COMPARISON_OUT_DIR=${COMPARISON_OUT_DIR:-"plots/fedrl_comparison_mixed"}
ABLATION_OUT_DIR=${ABLATION_OUT_DIR:-"plots/fedrl_ablations_mixed"}
SUMMARY_OUT_DIR=${SUMMARY_OUT_DIR:-"plots/fedrl_tables_mixed"}
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
    POP=${FED_POPULATION_SIZE:-$(( POP < 12 ? POP : 12 ))}
    GENS=${FED_MAX_GENERATIONS:-$(( GENS < 20 ? GENS : 20 ))}
    EVAL_EPISODES=${FED_EVAL_EPISODES:-2}
    CLIENT_ROLLOUTS=${FED_CLIENT_ROLLOUTS:-1}
    CLIENT_UPDATES=${FED_CLIENT_UPDATES:-2}
  else
    POP=${FED_POPULATION_SIZE:-$POP}
    GENS=${FED_MAX_GENERATIONS:-$GENS}
    EVAL_EPISODES=${FED_EVAL_EPISODES:-2}
    CLIENT_ROLLOUTS=${FED_CLIENT_ROLLOUTS:-1}
    CLIENT_UPDATES=${FED_CLIENT_UPDATES:-2}
  fi
  CLIENTS=${FED_NUM_CLIENTS:-4}
  ALG=${FED_ALGORITHM:-"FSAC"}
  if [[ -n "${TARGET_ENV_STEPS:-}" ]]; then
    TARGET_STEPS="$TARGET_ENV_STEPS"
  else
    TARGET_STEPS=$(python3 - <<PY
import math
pop = int('$POP')
clients = int('$CLIENTS')
gens = int('$GENS')
steps = int('$STEPS')
eval_episodes = int('$EVAL_EPISODES')
rollouts = int('$CLIENT_ROLLOUTS')
interval = 5
eval_half = max(1, eval_episodes // 2)
fed_rounds = sum(1 for g in range(gens) if g % interval == 0)
target = gens * pop * clients * steps * eval_half + fed_rounds * clients * rollouts * steps
print(target)
PY
)
  fi
  if [[ -n "${BASELINE_ROUNDS:-}" ]]; then
    THIS_BASELINE_ROUNDS="$BASELINE_ROUNDS"
  else
    THIS_BASELINE_ROUNDS=$(python3 - <<PY
import math
target = int('$TARGET_STEPS')
clients = int('$CLIENTS')
steps = int('$STEPS')
print(max(1, math.ceil(target / max(1, clients * steps))))
PY
)
  fi
  echo "$ENV_NAME budget=$BUDGET_PRESET target_env_steps=$TARGET_STEPS baseline_rounds>=$THIS_BASELINE_ROUNDS"

  for SEED in $SEEDS; do
    for VARIANT in $FED_VARIANTS; do
      EXP="fedrlhet_${ENV_NAME}_${VARIANT}_s${SEED}"
      python3 -m src.main \
        --env "$ENV_NAME" \
        --mode fed_evo_rl \
        --fed-ablation "$VARIANT" \
        --algorithm "$ALG" \
        --population-size "$POP" \
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
        --batch-size 32 \
        --eval-episodes "$EVAL_EPISODES" \
        --seed "$SEED" \
        --log-dir "$LOG_DIR" \
        --exp-name "$EXP"
    done
    for EVO_MODE in $EVO_BASELINES; do
      if [[ "$EVO_MODE" != "evosac_nofed" ]]; then
        echo "Unknown EVO baseline: $EVO_MODE" >&2
        exit 1
      fi
      EXP="${EVO_MODE}_${ENV_NAME}_s${SEED}"
      python3 -m src.main \
        --env "$ENV_NAME" \
        --mode standard_erl \
        --algorithm "$ALG" \
        --population-size "$POP" \
        --num-workers "$CLIENTS" \
        --max-generations "$GENS" \
        --target-env-steps "$TARGET_STEPS" \
        --max-episode-steps "$STEPS" \
        --rl-rollouts "$CLIENT_ROLLOUTS" \
        --rl-updates "$CLIENT_UPDATES" \
        --batch-size 32 \
        --eval-episodes "$EVAL_EPISODES" \
        --seed "$SEED" \
        --log-dir "$LOG_DIR" \
        --exp-name "$EXP"
    done
    for PAPER_MODE in $PAPER_MODES; do
      EXP="${PAPER_MODE}_${ENV_NAME}_s${SEED}"
      python3 scripts/train_fsac_paper_baseline.py \
        --env "$ENV_NAME" \
        --seed "$SEED" \
        --rounds "$THIS_BASELINE_ROUNDS" \
        --target-env-steps "$TARGET_STEPS" \
        --num-workers "$CLIENTS" \
        --updates "$CLIENT_UPDATES" \
        --batch-size 32 \
        --eval-episodes "$EVAL_EPISODES" \
        --max-episode-steps "$STEPS" \
        --client-heterogeneity "$CLIENT_HETEROGENEITY" \
        --client-heterogeneity-mode "$CLIENT_HETEROGENEITY_MODE" \
        --log-dir "$PAPER_LOG_DIR" \
        --exp-name "$EXP" \
        --baseline-mode "$PAPER_MODE"
    done
    for DQN_MODE in $DQN_BASELINES; do
      if [[ "$DQN_MODE" != "fedavg_dqn" ]]; then
        echo "Unknown DQN baseline: $DQN_MODE" >&2
        exit 1
      fi
      EXP="${DQN_MODE}_${ENV_NAME}_s${SEED}"
      python3 scripts/train_fedavg_dqn_baseline.py \
        --env "$ENV_NAME" \
        --seed "$SEED" \
        --rounds "$THIS_BASELINE_ROUNDS" \
        --target-env-steps "$TARGET_STEPS" \
        --num-workers "$CLIENTS" \
        --updates "$CLIENT_UPDATES" \
        --batch-size 32 \
        --eval-episodes "$EVAL_EPISODES" \
        --max-episode-steps "$STEPS" \
        --client-heterogeneity "$CLIENT_HETEROGENEITY" \
        --client-heterogeneity-mode "$CLIENT_HETEROGENEITY_MODE" \
        --log-dir "$DQN_LOG_DIR" \
        --exp-name "$EXP"
    done
  done
done

python3 scripts/plot_fedrl_heterogeneous.py \
  --fed-log-dir "$LOG_DIR" \
  --paper-log-dir "$PAPER_LOG_DIR" \
  --dqn-log-dir "$DQN_LOG_DIR" \
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
  --paper-log-dir "$PAPER_LOG_DIR" \
  --dqn-log-dir "$DQN_LOG_DIR" \
  --out-dir "$SUMMARY_OUT_DIR" \
  --plot-kind comparison

python3 scripts/summarize_fedrl_results.py \
  --fed-log-dir "$LOG_DIR" \
  --paper-log-dir "" \
  --dqn-log-dir "" \
  --out-dir "$SUMMARY_OUT_DIR" \
  --plot-kind ablation

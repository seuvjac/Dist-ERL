#!/usr/bin/env bash
# FedEvoFSAC runs for the three SAC/FSAC comparison environments.
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

ENVS=${ENVS:-"CartPole-v1 Acrobot-v1 LunarLander-v3"}
SEEDS=${SEEDS:-"0"}
FED_VARIANTS=${FED_VARIANTS:-"full uniform_aggregation no_local_rl no_ea_injection no_heterogeneity"}
PAPER_MODES=${PAPER_MODES:-"paper_sac paper_fsac fedavg_sac fedsoftmax_sac_noea fedbest_sac fedmedian_sac fedtrimmedmean_sac attention_sac_lite"}
EVO_BASELINES=${EVO_BASELINES:-"evosac_nofed"}
DQN_BASELINES=${DQN_BASELINES:-"fedavg_dqn"}
LOG_DIR=${LOG_DIR:-"logs_fedrl_hetero"}
PAPER_LOG_DIR=${PAPER_LOG_DIR:-"logs_fsac_paper"}
DQN_LOG_DIR=${DQN_LOG_DIR:-"logs_dqn_fedrl"}
OUT_DIR=${OUT_DIR:-"plots/fedrl_heterogeneous"}

for ENV_NAME in $ENVS; do
  read POP CLIENTS GENS STEPS <<<"$(python3 - <<PY
from src.config import env_run_preset
p = env_run_preset('$ENV_NAME')
print(p['population_size'], p['num_workers'], p['max_generations'], p['max_episode_steps'])
PY
)"

  # Keep the first all-environment FedEvoRL pass tractable; raise overrides for final paper runs.
  POP=${FED_POPULATION_SIZE:-$(( POP < 12 ? POP : 12 ))}
  CLIENTS=${FED_NUM_CLIENTS:-$(( CLIENTS < 2 ? CLIENTS : 2 ))}
  GENS=${FED_MAX_GENERATIONS:-$(( GENS < 20 ? GENS : 20 ))}
  EVAL_EPISODES=${FED_EVAL_EPISODES:-2}
  CLIENT_ROLLOUTS=${FED_CLIENT_ROLLOUTS:-1}
  CLIENT_UPDATES=${FED_CLIENT_UPDATES:-2}
  ALG=${FED_ALGORITHM:-"FSAC"}

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
        --max-episode-steps "$STEPS" \
        --client-heterogeneity 0.35 \
        --client-heterogeneity-mode env_params \
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
        --rounds "$GENS" \
        --num-workers "$CLIENTS" \
        --updates "$CLIENT_UPDATES" \
        --batch-size 32 \
        --eval-episodes "$EVAL_EPISODES" \
        --max-episode-steps "$STEPS" \
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
        --rounds "$GENS" \
        --num-workers "$CLIENTS" \
        --updates "$CLIENT_UPDATES" \
        --batch-size 32 \
        --eval-episodes "$EVAL_EPISODES" \
        --max-episode-steps "$STEPS" \
        --log-dir "$DQN_LOG_DIR" \
        --exp-name "$EXP"
    done
  done
done

python3 scripts/plot_fedrl_heterogeneous.py \
  --fed-log-dir "$LOG_DIR" \
  --paper-log-dir "$PAPER_LOG_DIR" \
  --dqn-log-dir "$DQN_LOG_DIR" \
  --out-dir "$OUT_DIR"

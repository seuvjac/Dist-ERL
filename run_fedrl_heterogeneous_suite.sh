#!/usr/bin/env bash
# FedEvoFSAC heterogeneous benchmark suite with SAC/FSAC baselines.
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

ENVS=${ENVS:-"CartPole-v1 Acrobot-v1 LunarLander-v3"}
SEEDS=${SEEDS:-"0 1 2"}
FED_VARIANTS=${FED_VARIANTS:-"full uniform_aggregation no_local_rl no_ea_injection no_heterogeneity"}
PAPER_MODES=${PAPER_MODES:-"paper_sac paper_fsac fedavg_fsac fedsoftmax_fsac_noea fedbest_fsac"}
EVO_BASELINES=${EVO_BASELINES:-"evosac_nofed"}
LOG_DIR=${LOG_DIR:-"logs_fedrl_hetero"}
PAPER_LOG_DIR=${PAPER_LOG_DIR:-"logs_fsac_paper"}

for ENV_NAME in $ENVS; do
  read POP CLIENTS GENS STEPS <<<"$(python3 - <<PY
from src.config import env_run_preset
p = env_run_preset('$ENV_NAME')
print(p['population_size'], p['num_workers'], p['max_generations'], p['max_episode_steps'])
PY
)"
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
        --client-rollouts 2 \
        --client-updates 8 \
        --eval-episodes 4 \
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
        --eval-episodes 4 \
        --max-episode-steps "$STEPS" \
        --log-dir "$PAPER_LOG_DIR" \
        --exp-name "$EXP" \
        --baseline-mode "$PAPER_MODE"
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
        --rl-rollouts 2 \
        --rl-updates 8 \
        --eval-episodes 4 \
        --seed "$SEED" \
        --log-dir "$LOG_DIR" \
        --exp-name "$EXP"
    done
  done
done

python3 scripts/plot_fedrl_heterogeneous.py \
  --fed-log-dir "$LOG_DIR" \
  --paper-log-dir "$PAPER_LOG_DIR"

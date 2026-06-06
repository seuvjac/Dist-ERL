#!/bin/bash
# Long MuJoCo comparison for FedEvoRL against related ERL/RL baselines.
set -uo pipefail
cd "$(dirname "$0")"

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTHONUNBUFFERED=1
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

source /home/ywj/anaconda3/bin/activate dist-erl-re2

ENVS=${ENVS:-"Hopper-v2"}
SEEDS=${SEEDS:-"0 1 2"}
MODES=${MODES:-"fed_evo_rl dist_erl erl_re2 standard_erl pure_ea pure_rl"}
MAX_GENERATIONS_OVERRIDE=${MAX_GENERATIONS_OVERRIDE:-""}
LOG_DIR=${LOG_DIR:-"logs"}

echo "Long MuJoCo comparison"
echo "  envs=$ENVS"
echo "  seeds=$SEEDS"
echo "  modes=$MODES"
echo "  log_dir=$LOG_DIR"

failures=0

for ENV_NAME in $ENVS; do
  ENV_SLUG=$(echo "$ENV_NAME" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/_/g')
  read -r POP NW GENS STEPS <<< "$(python3 -c "
from src.config import env_run_preset
p = env_run_preset('$ENV_NAME')
print(p['population_size'], p['num_workers'], p['max_generations'], p['max_episode_steps'])
")"
  if [ -n "$MAX_GENERATIONS_OVERRIDE" ]; then
    GENS="$MAX_GENERATIONS_OVERRIDE"
  fi

  for SEED in $SEEDS; do
    for MODE in $MODES; do
      EXTRA=()
      case "$MODE" in
        pure_rl) EXTRA=(--population-size 10 --num-workers 1 --rl-rollouts 2 --rl-updates 10) ;;
        pure_ea) EXTRA=(--population-size "$POP" --num-workers 2) ;;
        standard_erl|erl_re2) EXTRA=(--population-size "$POP" --num-workers 1) ;;
        dist_erl) EXTRA=(--population-size "$POP" --num-workers "$NW") ;;
        fed_evo_rl) EXTRA=(--population-size "$POP" --num-clients "$NW" --client-fraction 1.0 --fed-ablation full) ;;
      esac

      EXP="long_${ENV_SLUG}_${MODE}_s${SEED}"
      echo ">>> $EXP"
      MODE="$MODE" ENV_NAME="$ENV_NAME" ./run_dist_erl.sh \
        --env "$ENV_NAME" \
        --mode "$MODE" \
        --exp-name "$EXP" \
        --log-dir "$LOG_DIR" \
        --max-generations "$GENS" \
        --max-episode-steps "$STEPS" \
        --seed "$SEED" \
        "${EXTRA[@]}"
      rc=$?
      if [ "$rc" -ne 0 ]; then
        echo "FAILED: $EXP rc=$rc"
        failures=$((failures + 1))
      fi
    done
  done
done

python3 generate_plots.py --log-dir "$LOG_DIR" --require-real
plot_rc=$?
if [ "$plot_rc" -ne 0 ]; then
  echo "Plot generation failed rc=$plot_rc"
  failures=$((failures + 1))
fi

echo "Long MuJoCo comparison finished with failures=$failures"
exit "$failures"

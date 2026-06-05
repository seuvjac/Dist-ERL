#!/bin/bash
# Multi-seed benchmark: 6x MuJoCo-v2 x 5 modes (see src/config.py MUJOCO_V2_ENVS)
set -e
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd):$PYTHONPATH"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTHONUNBUFFERED=1
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}
ENVS=${ENVS:-$(python -c "from src.config import MUJOCO_V2_ENVS; print(' '.join(MUJOCO_V2_ENVS))")}
MODES=${MODES:-"pure_rl pure_ea standard_erl erl_re2 dist_erl"}

echo "Multi-seed run: seeds=$SEEDS"
echo "MuJoCo-v2 envs: $ENVS"

for ENV_NAME in $ENVS; do
  ENV_SLUG=$(echo "$ENV_NAME" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/_/g')
  read -r POP NW GENS STEPS <<< "$(python -c "
from src.config import env_run_preset
p = env_run_preset('$ENV_NAME')
print(p['population_size'], p['num_workers'], p['max_generations'], p['max_episode_steps'])
")"

  for SEED in $SEEDS; do
    for MODE in $MODES; do
      EXTRA=()
      case "$MODE" in
        pure_rl) EXTRA=(--population-size 10 --num-workers 1) ;;
        pure_ea) EXTRA=(--population-size "$POP" --num-workers 2) ;;
        standard_erl|erl_re2) EXTRA=(--population-size "$POP" --num-workers 1) ;;
        dist_erl) EXTRA=(--population-size "$POP" --num-workers "$NW") ;;
      esac

      EXP="paper_${ENV_SLUG}_${MODE}_s${SEED}"
      echo ">>> $EXP env=$ENV_NAME mode=$MODE seed=$SEED (preset pop=$POP nw=$NW)"
      MODE="$MODE" ENV_NAME="$ENV_NAME" ./run_dist_erl.sh \
        --env "$ENV_NAME" \
        --mode "$MODE" \
        --exp-name "$EXP" \
        --max-generations "$GENS" \
        --max-episode-steps "$STEPS" \
        --seed "$SEED" \
        "${EXTRA[@]}"
    done
  done
done

echo "Done. Plot: python3 generate_plots.py --log-dir logs --require-real"

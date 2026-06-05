#!/bin/bash
# Six MuJoCo-v2 tasks x 5 modes (src/config.py MUJOCO_V2_ENVS)
set -e
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd):$PYTHONPATH"

SEED=${SEED:-42}
MODES=${MODES:-"fed_evo_rl dist_erl erl_re2 standard_erl pure_rl pure_ea"}

echo "Multi-environment MuJoCo-v2 benchmark, seed=$SEED"
echo "Modes: $MODES"

for ENV_NAME in $(python -c "from src.config import MUJOCO_V2_ENVS; print(' '.join(MUJOCO_V2_ENVS))"); do
  ENV_SLUG=$(echo "$ENV_NAME" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/_/g')
  read -r POP NW GENS STEPS <<< "$(python -c "
from src.config import env_run_preset
p = env_run_preset('$ENV_NAME')
print(p['population_size'], p['num_workers'], p['max_generations'], p['max_episode_steps'])
")"

  echo ""
  echo "========== Environment: $ENV_NAME =========="

  for MODE in $MODES; do
    EXTRA=()
    case "$MODE" in
      pure_rl) EXTRA=(--population-size 10 --num-workers 1) ;;
      pure_ea) EXTRA=(--num-workers 2) ;;
      standard_erl|erl_re2) EXTRA=(--population-size "$POP" --num-workers 1) ;;
      dist_erl) EXTRA=(--population-size "$POP" --num-workers "$NW") ;;
      fed_evo_rl) EXTRA=(--population-size "$POP" --num-clients "$NW" --client-fraction 1.0) ;;
    esac

    ./run_dist_erl.sh \
      --env "$ENV_NAME" \
      --mode "$MODE" \
      --exp-name "multi_${ENV_SLUG}_${MODE}_s${SEED}" \
      --max-generations "$GENS" \
      --max-episode-steps "$STEPS" \
      --population-size "$POP" \
      --seed "$SEED" \
      "${EXTRA[@]}"
  done
done

echo ""
echo "All MuJoCo-v2 environments finished."
echo "  python3 generate_plots.py --log-dir logs --require-real"

#!/bin/bash
# FedEvoRL component ablation on MuJoCo. Produces >=4 ablation curves.
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
FED_ABLATIONS=${FED_ABLATIONS:-"full no_local_rl no_ea_injection no_heterogeneity"}
MAX_GENERATIONS_OVERRIDE=${MAX_GENERATIONS_OVERRIDE:-""}
LOG_DIR=${LOG_DIR:-"logs"}

echo "FedEvoRL MuJoCo ablation"
echo "  envs=$ENVS"
echo "  seeds=$SEEDS"
echo "  fed_ablations=$FED_ABLATIONS"
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
    for FED_ABLATION in $FED_ABLATIONS; do
      EXP="abl_fed_${ENV_SLUG}_${FED_ABLATION}_s${SEED}"
      echo ">>> $EXP"
      MODE="fed_evo_rl" ENV_NAME="$ENV_NAME" ./run_dist_erl.sh \
        --env "$ENV_NAME" \
        --mode fed_evo_rl \
        --fed-ablation "$FED_ABLATION" \
        --exp-name "$EXP" \
        --log-dir "$LOG_DIR" \
        --population-size "$POP" \
        --num-clients "$NW" \
        --client-fraction 1.0 \
        --max-generations "$GENS" \
        --max-episode-steps "$STEPS" \
        --seed "$SEED"
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

echo "FedEvoRL MuJoCo ablation finished with failures=$failures"
exit "$failures"

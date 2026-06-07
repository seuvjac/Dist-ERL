#!/bin/bash

# FedEvoRL Launcher Script
# Sets up environment variables and launches federated/evolutionary training

# Headless MuJoCo (tmux/SSH/Ray workers — no GL window)
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTHONUNBUFFERED=1
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

# Set CUDA device (modify as needed)
export CUDA_VISIBLE_DEVICES=0

# Set Python path
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Activate conda environment (adjust path as needed)
source /home/ywj/anaconda3/bin/activate dist-erl-re2

# Default: MuJoCo-v2 main task (override with ENV_NAME=...)
ENV_NAME=${ENV_NAME:-"HalfCheetah-v2"}
POPULATION_SIZE=${POPULATION_SIZE:-50}
NUM_WORKERS=${NUM_WORKERS:-4}
MAX_GENERATIONS=${MAX_GENERATIONS:-200}
ALGORITHM=${ALGORITHM:-"FSAC"}
NUM_CLIENTS=${NUM_CLIENTS:-4}
MODE=${MODE:-"fed_evo_rl"}
USE_WANDB=${USE_WANDB:-0}
WANDB_API_KEY=${WANDB_API_KEY:-""}

ABLATION=${ABLATION:-""}

echo "Starting FedEvoRL:"
echo "  Environment: $ENV_NAME"
echo "  Mode: $MODE"
[ -n "$ABLATION" ] && echo "  Ablation: $ABLATION"
echo "  Population Size: $POPULATION_SIZE"
echo "  Workers: $NUM_WORKERS"
echo "  Federated Clients: $NUM_CLIENTS"
echo "  Max Generations: $MAX_GENERATIONS"
echo "  RL Algorithm: $ALGORITHM"

# Launch training (metrics saved to logs/ for generate_plots.py)
WANDB_ARGS=()
if [ "$USE_WANDB" = "1" ]; then
    WANDB_ARGS=(--wandb --wandb-key "$WANDB_API_KEY" --wandb-project "FedEvoRL-Benchmarks")
    echo "WandB logging enabled (optional; plots use local logs/)"
fi

ABLATION_ARGS=()
[ -n "$ABLATION" ] && ABLATION_ARGS=(--ablation "$ABLATION")

python -u src/main.py \
    --env "$ENV_NAME" \
    --mode "$MODE" \
    "${ABLATION_ARGS[@]}" \
    --population-size "$POPULATION_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --num-clients "$NUM_CLIENTS" \
    --max-generations "$MAX_GENERATIONS" \
    --algorithm "$ALGORITHM" \
    "${WANDB_ARGS[@]}" \
    "$@"

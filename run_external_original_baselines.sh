#!/usr/bin/env bash
# Run downloaded original-code FedRL baselines separately from same-protocol plots.
set -euo pipefail

cd "$(dirname "$0")"

EXTERNAL_PYTHON=${EXTERNAL_PYTHON:-python3}
OUT_ROOT=${OUT_ROOT:-"$PWD/external_original_logs"}
FED_DRL_REPO=${FED_DRL_REPO:-"/home/ywj/code/Federated-DRL"}
FEDPG_REPO=${FEDPG_REPO:-"/home/ywj/code/Byzantine-Federated-RL"}
RUN_FEDERATED_DRL=${RUN_FEDERATED_DRL:-1}
RUN_FEDPG_BR=${RUN_FEDPG_BR:-1}

mkdir -p "$OUT_ROOT"
export MPLBACKEND=${MPLBACKEND:-Agg}

echo "External-original baselines"
echo "  python=$EXTERNAL_PYTHON"
echo "  out=$OUT_ROOT"

if [[ "$RUN_FEDERATED_DRL" == "1" ]]; then
  if [[ ! -d "$FED_DRL_REPO" ]]; then
    echo "Missing Federated-DRL repo: $FED_DRL_REPO" >&2
    exit 1
  fi
  echo "Running Federated-DRL original code: CartPole-v1"
  (
    cd "$FED_DRL_REPO"
    "$EXTERNAL_PYTHON" main-cart.py
    mkdir -p "$OUT_ROOT/Federated-DRL/CartPole-v1"
    mv -f fed_rewards.npy single_rewards.npy "$OUT_ROOT/Federated-DRL/CartPole-v1/"
  )

  echo "Running Federated-DRL original code: LunarLander-v2"
  (
    cd "$FED_DRL_REPO"
    "$EXTERNAL_PYTHON" main-lun.py
    mkdir -p "$OUT_ROOT/Federated-DRL/LunarLander-v2"
    mv -f fed_rewards.npy single_rewards.npy "$OUT_ROOT/Federated-DRL/LunarLander-v2/"
  )
fi

if [[ "$RUN_FEDPG_BR" == "1" ]]; then
  if [[ ! -d "$FEDPG_REPO" ]]; then
    echo "Missing Byzantine-Federated-RL repo: $FEDPG_REPO" >&2
    exit 1
  fi
  echo "Running FedPG-BR original code: CartPole-v1"
  (
    cd "$FEDPG_REPO/codes"
    "$EXTERNAL_PYTHON" run.py \
      --env_name CartPole-v1 \
      --FedPG_BR \
      --num_worker 4 \
      --num_Byzantine 0 \
      --log_dir "$OUT_ROOT/FedPG-BR" \
      --multiple_run 3 \
      --run_name CartPole_FedPGBR_W4B0 \
      --no_saving
  )

  echo "Running FedPG-BR original code: LunarLander-v2"
  (
    cd "$FEDPG_REPO/codes"
    "$EXTERNAL_PYTHON" run.py \
      --env_name LunarLander-v2 \
      --FedPG_BR \
      --num_worker 4 \
      --num_Byzantine 0 \
      --log_dir "$OUT_ROOT/FedPG-BR" \
      --multiple_run 3 \
      --run_name LunarLander_FedPGBR_W4B0 \
      --no_saving
  )
fi

cat > "$OUT_ROOT/README.md" <<'EOF'
# External Original Baselines

This directory is reserved for original-code external baselines.

- `Federated-DRL`: original FedAvg DQN/DDQN scripts; supports CartPole-v1 and LunarLander-v2.
- `FedPG-BR`: original Byzantine-Federated-RL policy-gradient code; supports CartPole-v1 and LunarLander-v2.

These outputs are not same-protocol internal baselines. Keep them in a separate figure/table from the FedEvoFSAC same-protocol comparison and from FedEvoFSAC ablations.
EOF

echo "Done. Results are under $OUT_ROOT"

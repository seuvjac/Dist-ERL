#!/bin/bash
# Run from anywhere: bash /home/ywj/code/Dist-ERL/run_experiments.sh [seeds|plots|scaling|all]
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:$PYTHONPATH"

usage() {
  echo "Usage: $0 [seeds|plots|scaling|smoke|all]"
  echo "  seeds   - ./run_seeds.sh (long, MuJoCo x 5 modes x 10 seeds)"
  echo "  plots   - python3 generate_plots.py --log-dir logs --require-real"
  echo "  scaling - ./run_scaling.sh + plot_scaling_bandwidth.py"
  echo "  smoke   - short LunarLander dist_erl test (30 gen)"
  echo "  all     - smoke then plots (if logs exist)"
}

cmd=${1:-usage}
case "$cmd" in
  seeds)
    ./run_seeds.sh
    ;;
  plots)
    python3 generate_plots.py --log-dir logs --require-real
    ;;
  scaling)
    ./run_scaling.sh
    python3 scripts/plot_scaling_bandwidth.py --log-dir logs
    ;;
  smoke)
    ./run_dist_erl.sh --env LunarLanderContinuous-v3 --mode dist_erl \
      --exp-name smoke_lunar_fixed --max-generations 30 --num-workers 2 --population-size 20 --seed 42
    ;;
  all)
    echo "=== smoke (optional quick check) ==="
    "$0" smoke || true
    echo "=== plots ==="
    python3 generate_plots.py --log-dir logs --require-real || python3 generate_plots.py --log-dir logs --allow-synthetic
    ;;
  *)
    usage
    exit 1
    ;;
esac

#!/bin/bash
# Starts the requested long comparison and ablation jobs in the background.
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p logs/background

STAMP=$(date +"%Y%m%d_%H%M%S")
SUITE_LOG="logs/background/mujoco_suite_${STAMP}.out"

nohup bash -c './run_mujoco_long_compare.sh; compare_rc=$?; ./run_fed_evo_ablation_mujoco.sh; ablation_rc=$?; exit $((compare_rc + ablation_rc))' > "$SUITE_LOG" 2>&1 &
SUITE_PID=$!

cat > "logs/background/latest_suite_${STAMP}.txt" <<EOF
suite_pid=$SUITE_PID
suite_log=$SUITE_LOG
started_at=$STAMP
EOF

ln -sfn "latest_suite_${STAMP}.txt" logs/background/latest_suite.txt

echo "Started MuJoCo suite: pid=$SUITE_PID log=$SUITE_LOG"
echo "Status file: logs/background/latest_suite.txt"

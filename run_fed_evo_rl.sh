#!/bin/bash
# FedEvoRL main launcher.
set -e

cd "$(dirname "$0")"
MODE=${MODE:-"fed_evo_rl"} ./run_dist_erl.sh "$@"

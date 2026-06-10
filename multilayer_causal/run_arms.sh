#!/bin/bash
# Shard arms across visible GPUs, sequential per GPU, resume-safe.
# Usage: bash run_arms.sh <NGPUS> <arm1> <arm2> ...
set -uo pipefail
NGPUS=$1; shift
declare -a Q
i=0
for arm in "$@"; do
  g=$((i % NGPUS)); Q[$g]="${Q[$g]:-} $arm"; i=$((i+1))
done
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for g in $(seq 0 $((NGPUS-1))); do
  (
    for arm in ${Q[$g]:-}; do
      echo "[gpu$g] start $arm $(date -Is)"
      python "$SCRIPT_DIR/run_experiment.py" --arm "$arm" --gpu "$g" \
        || echo "[gpu$g] $arm FAILED (continuing)"
    done
  ) &
done
wait
echo "ALL ARMS DONE"

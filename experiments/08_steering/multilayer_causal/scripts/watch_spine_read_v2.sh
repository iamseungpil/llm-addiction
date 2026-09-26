#!/bin/bash
set -uo pipefail
export PATH=~/miniconda3/envs/amlt/bin:$PATH
EXP=spine_read_v2
cd /home/v-seungplee/llm-addiction
while true; do
  L=$(amlt status "$EXP" 2>&1 | grep -E "^:?spine_read" | head -1)
  echo "[$(date +%H:%M:%S)] $L"
  case "$L" in
    *pass*|*completed*) echo "JOB_PASS"; exit 0 ;;
    *failed*)    echo "JOB_FAILED"; exit 2 ;;
    *preempted*) echo "JOB_PREEMPTED"; exit 2 ;;
    *killed*)    echo "JOB_KILLED"; exit 2 ;;
  esac
  sleep 180
done

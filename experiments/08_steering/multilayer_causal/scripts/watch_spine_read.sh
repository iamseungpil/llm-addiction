#!/bin/bash
# Poll amlt status for the spine_read sweep until it leaves the running set.
set -uo pipefail
export PATH=~/miniconda3/envs/amlt/bin:$PATH
EXP=spine_read
cd /home/v-seungplee/llm-addiction
while true; do
  OUT=$(amlt status "$EXP" 2>&1)
  LINE=$(echo "$OUT" | grep -E '^:?spine_read' | head -1)
  echo "[$(date +%H:%M:%S)] $LINE"
  case "$LINE" in
    *pass*)      echo "JOB_PASS";   exit 0 ;;
    *completed*) echo "JOB_PASS";   exit 0 ;;
    *failed*)    echo "JOB_FAILED"; exit 2 ;;
    *preempted*) echo "JOB_PREEMPTED"; exit 2 ;;
    *killed*)    echo "JOB_KILLED"; exit 2 ;;
  esac
  sleep 180
done

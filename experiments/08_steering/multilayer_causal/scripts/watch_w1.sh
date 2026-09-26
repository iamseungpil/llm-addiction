#!/bin/bash
# Poll amlt status for mlc-w1-0611 until the job leaves the running set.
# Exits 0 on pass, 2 on preempted/failed/killed (caller decides resubmit).
set -uo pipefail
export PATH=~/miniconda3/envs/amlt/bin:$PATH
EXP=mlc-w1-0611
cd /home/v-seungplee/llm-addiction
while true; do
  OUT=$(amlt status "$EXP" 2>&1)
  LINE=$(echo "$OUT" | grep -E '^:?mlc_w1' | head -1)
  echo "[$(date +%H:%M:%S)] $LINE"
  case "$LINE" in
    *pass*)      echo "JOB_PASS";   exit 0 ;;
    *failed*)    echo "JOB_FAILED"; exit 2 ;;
    *preempted*) echo "JOB_PREEMPTED"; exit 2 ;;
    *killed*)    echo "JOB_KILLED"; exit 2 ;;
  esac
  sleep 180
done

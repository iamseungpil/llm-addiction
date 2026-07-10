#!/bin/bash
set -uo pipefail
export PATH=~/miniconda3/envs/amlt/bin:$PATH
cd /home/v-seungplee/llm-addiction
declare -A DONE; for e in spine_read_v4 spine_crosstask; do DONE[$e]=""; done
while true; do
  ALL=1
  for e in spine_read_v4 spine_crosstask; do
    [ -n "${DONE[$e]}" ] && continue
    L=$(amlt status "$e" 2>&1 | grep -E "^:?spine" | head -1)
    echo "[$(date +%H:%M:%S)] $e | $L"
    case "$L" in
      *pass*|*completed*) DONE[$e]="pass" ;;
      *failed*) DONE[$e]="failed" ;;
      *killed*) DONE[$e]="killed" ;;
      *) ALL=0 ;;
    esac
  done
  [ "$ALL" = "1" ] && { echo "ALL_TERMINAL read=${DONE[spine_read_v4]} crosstask=${DONE[spine_crosstask]}"; exit 0; }
  sleep 180
done

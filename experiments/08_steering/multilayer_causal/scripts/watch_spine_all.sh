#!/bin/bash
# Poll all three spine jobs until every one reaches a terminal state.
set -uo pipefail
export PATH=~/miniconda3/envs/amlt/bin:$PATH
cd /home/v-seungplee/llm-addiction
declare -A DONE
for e in spine_read spine_write spine_extract; do DONE[$e]=""; done
while true; do
  ALL=1
  for e in spine_read spine_write spine_extract; do
    [ -n "${DONE[$e]}" ] && continue
    L=$(amlt status "$e" 2>&1 | grep -E "^:?$e" | head -1)
    echo "[$(date +%H:%M:%S)] $e | $L"
    case "$L" in
      *pass*|*completed*) DONE[$e]="pass" ;;
      *failed*)    DONE[$e]="failed" ;;
      *preempted*) DONE[$e]="preempted" ;;
      *killed*)    DONE[$e]="killed" ;;
      *) ALL=0 ;;
    esac
  done
  if [ "$ALL" = "1" ]; then
    echo "ALL_TERMINAL read=${DONE[spine_read]} write=${DONE[spine_write]} extract=${DONE[spine_extract]}"
    exit 0
  fi
  sleep 180
done

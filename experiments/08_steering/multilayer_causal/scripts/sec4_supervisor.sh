#!/bin/bash
# SEC4 auto-resume supervisor: watches W2+W3, resubmits on pause / lost-cells,
# alerts (exit!=0) only on build failure or resubmit-cap. Resume-safe by design
# (jobs pull per-arm HF checkpoints and skip completed seeds).
set -uo pipefail
AMLT=~/miniconda3/envs/amlt/bin/amlt
PY=~/miniconda3/envs/metavllm/bin/python
cd "$(dirname "$0")/.."

W2_EXPS=(mlc-sec4-w2-0706); W2_YAML=amlt/sec4_w2.rendered.yaml; W2_TARGET=69
W3_EXPS=(mlc-sec4-w3-0706c); W3_YAML=amlt/sec4_w3.rendered.yaml; W3_TARGET=38
W2_RESUB=0; W3_RESUB=0; MAX_RESUB=3

cells() { $PY - <<PY 2>/dev/null
from huggingface_hub import HfApi
f=HfApi().list_repo_files("llm-addiction-research/llm-addiction",repo_type="dataset")
print(sum(1 for x in f if "/sec4_$1_" in x and x.endswith(".jsonl")))
PY
}
status_of() { $AMLT status "$1" 2>/dev/null | grep -iE "mlc_sec4" | head -1; }

for i in $(seq 1 200); do
  sleep 180
  w2n=$(cells w2); w3n=$(cells w3)
  w2exp=${W2_EXPS[-1]}; w3exp=${W3_EXPS[-1]}
  st2=$(status_of "$w2exp"); st3=$(status_of "$w3exp")
  echo "[$(date -Is)] $i | w2=$w2n/$W2_TARGET($w2exp) w3=$w3n/$W3_TARGET($w3exp)"
  echo "    W2:$st2"; echo "    W3:$st3"

  # W3 build failure = code problem: alert, never auto-resubmit
  log3=$($AMLT log view "$w3exp" 2>/dev/null | tail -50)
  if echo "$log3" | grep -qE "SEC4_BUILD_FAILED|SEC4_AXIS_VERIFY_FAILED|PIP_INSTALL_FAILED"; then
    echo "W3 BUILD FAILED — needs code fix"; echo "$log3" | grep -E "Error|Traceback|ValueError" | tail -6; exit 2
  fi

  w2done=0; [ "${w2n:-0}" -ge $W2_TARGET ] && w2done=1
  w3done=0; [ "${w3n:-0}" -ge $W3_TARGET ] && w3done=1

  resubmit() { # $1 wave-tag $2 yaml $3 new-name
    echo "    -> AUTO-RESUME: submitting $3"
    $AMLT run "$2" "$3" -y 2>&1 | grep -E ":mlc_sec4|Experiment" | head -2
  }
  # W2: pause or terminal-with-missing-cells -> resubmit
  if [ $w2done -eq 0 ]; then
    case "$st2" in
      *aused*|*ailed*|*illed*|*pass*)
        if [ $W2_RESUB -lt $MAX_RESUB ]; then
          W2_RESUB=$((W2_RESUB+1)); new="mlc-sec4-w2-r$W2_RESUB"
          resubmit w2 "$W2_YAML" "$new"; W2_EXPS+=("$new")
        else echo "W2 resubmit cap"; exit 2; fi;;
    esac
  fi
  if [ $w3done -eq 0 ]; then
    case "$st3" in
      *aused*|*ailed*|*illed*|*pass*)
        if [ $W3_RESUB -lt $MAX_RESUB ]; then
          W3_RESUB=$((W3_RESUB+1)); new="mlc-sec4-w3-r$W3_RESUB"
          resubmit w3 "$W3_YAML" "$new"; W3_EXPS+=("$new")
        else echo "W3 resubmit cap"; exit 2; fi;;
    esac
  fi
  [ $w2done -eq 1 ] && [ $w3done -eq 1 ] && { echo "BOTH WAVES COMPLETE (w2=$w2n w3=$w3n)"; exit 0; }
done
echo "supervisor horizon reached"; exit 1

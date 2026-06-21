#!/bin/bash
# Poll the mlc-xtask-0616 wave; exit when terminal (done/failed/killed/paused) or
# when >=15 result cells have landed on HF. Exit 0 = results in; 2 = needs action.
set -uo pipefail
AMLT=~/miniconda3/envs/amlt/bin/amlt
PY=~/miniconda3/envs/metavllm/bin/python
EXP=mlc-xtask-0616
HF=llm-addiction-research/llm-addiction
PREFIX=experiments/xtask/checkpoints/xtask
for i in $(seq 1 400); do
  n=$($PY - <<PY 2>/dev/null
from huggingface_hub import HfApi
f=HfApi().list_repo_files("$HF",repo_type="dataset")
print(sum(1 for x in f if x.startswith("$PREFIX/") and x.endswith(".jsonl")))
PY
)
  st=$($AMLT status $EXP 2>/dev/null | grep -iE "mlc_xtask" | head -1)
  echo "[$(date -Is)] poll $i: HF_cells=${n:-?} | $st"
  case "$st" in
    *ompleted*|*ailed*|*illed*) echo "TERMINAL"; exit 0;;
    *aused*) echo "PAUSED — needs resubmit"; exit 2;;
  esac
  if [ "${n:-0}" -ge 15 ]; then echo "ALL 15 CELLS ON HF"; exit 0; fi
  sleep 180
done
echo "watch timed out"; exit 1

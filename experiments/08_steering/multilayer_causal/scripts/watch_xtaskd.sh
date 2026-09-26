#!/bin/bash
# Poll mlc-xtaskd-0618; exit when terminal/paused or >=22 result cells on HF.
set -uo pipefail
AMLT=~/miniconda3/envs/amlt/bin/amlt
PY=~/miniconda3/envs/metavllm/bin/python
EXP=mlc-xtaskd-0618
HF=llm-addiction-research/llm-addiction
for i in $(seq 1 400); do
  n=$($PY - <<PY 2>/dev/null
from huggingface_hub import HfApi
f=HfApi().list_repo_files("$HF",repo_type="dataset")
print(sum(1 for x in f if x.startswith("experiments/xtaskd/") and x.endswith(".jsonl")))
PY
)
  st=$($AMLT status $EXP 2>/dev/null | grep -iE "mlc_xtaskd" | head -1)
  echo "[$(date -Is)] poll $i: HF_cells=${n:-?} | $st"
  case "$st" in
    *ompleted*|*ailed*|*illed*|*pass*) echo "TERMINAL"; exit 0;;
    *aused*) echo "PAUSED"; exit 2;;
  esac
  if [ "${n:-0}" -ge 22 ]; then echo "ALL 22 CELLS ON HF"; exit 0; fi
  sleep 180
done
echo "watch timed out"; exit 1

#!/bin/bash
set -uo pipefail
cd /scratch/llm-addiction/sae_v3_analysis
export PYTHONPATH=src:/scratch/llm-addiction/paper_experiments/slot_machine_6models/src:/scratch/llm-addiction/exploratory_experiments/alternative_paradigms/src:${PYTHONPATH:-}
export LLM_ADDICTION_BEHAVIORAL_ROOT=/scratch/llm-addiction-data/behavioral
export LLM_ADDICTION_DATA_ROOT=/scratch/llm-addiction-data/sae_features_v3
export LLM_ADDICTION_ANALYSIS_ROOT=/scratch/llm-addiction/sae_v3_analysis
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
ALPHAS="-2.0 -1.0 -0.5 0.0 0.5 1.0 2.0"

# HF sync daemon
cat > /tmp/_l22_sync.py <<'PY'
import os,sys,glob
from huggingface_hub import HfApi
api=HfApi(token=os.environ["HF_TOKEN"])
prefix=sys.argv[1]
for p in glob.glob("/scratch/l22_runs/*.json"):
    try: api.upload_file(path_or_fileobj=p, path_in_repo=f"{prefix}/{os.path.basename(p)}", repo_id="iamseungpil/metacot", repo_type="dataset")
    except Exception as e: print(f"sync_err:{e}")
PY
nohup bash -c 'while true; do python3 /tmp/_l22_sync.py llm_addiction_results/steering_l22_held_out_v5/continue_0428_node >> /scratch/l22_logs/hf_sync.log 2>&1; sleep 300; done' > /scratch/l22_logs/hf_sync.outer 2>&1 &
echo "[sync] daemon pid=$!"

launch () {
  local GPU="$1" CELL="$2" LAYER="$3"
  local OUT="/scratch/l22_runs/${CELL,,}_mw_L${LAYER}.json"
  local LOG="/scratch/l22_logs/${CELL,,}_g${GPU}.log"
  echo "[launch] GPU=$GPU CELL=$CELL → $OUT"
  CUDA_VISIBLE_DEVICES=$GPU python -u src/run_l22_held_out_steering.py \
    --cell $CELL --task mw --model llama --layer $LAYER \
    --n-games 100 --g-offset 4000 \
    --alphas $ALPHAS \
    --output "$OUT" \
    > "$LOG" 2>&1 &
  echo "  pid=$!"
}

launch 0 H13 25
launch 1 H14 30
launch 2 H2  22

sleep 8
echo "--- nvidia-smi ---"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "--- procs ---"
pgrep -af 'run_l22_held_out' | head

# Block until all 3 background processes finish
wait
echo "[done] all 3 cells completed"
python3 /tmp/_l22_sync.py llm_addiction_results/steering_l22_held_out_v5/continue_0428_node 2>&1 | tail -5

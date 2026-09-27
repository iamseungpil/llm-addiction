#!/bin/bash
# Launch L22 held-out cells on idle GPUs of a running AMLT node.
# Usage: NODE=<node_name> bash launch_l22_on_node.sh <ROLE>
#   ROLE in: ic_h1, mw_h2_h5true, adapted_h3, natural_h5rand
# Run via: amlt ssh <node> -c "bash -s" < /tmp/launch_l22_on_node.sh

set -uo pipefail

ROLE="${1:?need ROLE: ic_h1 | mw_h2_h5true | adapted_h3 | natural_h5rand}"

WORK=/scratch/llm-addiction/sae_v3_analysis
SRC=$WORK/src
LOGDIR=/scratch/l22_logs
OUTDIR=/scratch/l22_runs
mkdir -p "$LOGDIR" "$OUTDIR"

# 1) Pull new script from HF (single file, no tarball)
HF_TOKEN_FILE=/scratch/.hf_token
if [ ! -f "$HF_TOKEN_FILE" ] && [ -n "${HF_TOKEN:-}" ]; then
  echo -n "$HF_TOKEN" > "$HF_TOKEN_FILE"
fi

python3 - <<'PY'
import os
from huggingface_hub import hf_hub_download
tok = os.environ.get("HF_TOKEN") or open("/scratch/.hf_token").read().strip()
p = hf_hub_download(
    repo_id="iamseungpil/metacot",
    filename="code_snapshots/l22_held_out/run_l22_held_out_steering.py",
    repo_type="dataset",
    token=tok,
)
import shutil, os
dst = "/scratch/llm-addiction/sae_v3_analysis/src/run_l22_held_out_steering.py"
shutil.copy2(p, dst)
print("script staged:", dst)
PY

cd "$WORK"
export PYTHONPATH=$SRC:/scratch/llm-addiction/paper_experiments/slot_machine_6models/src:/scratch/llm-addiction/exploratory_experiments/alternative_paradigms/src:${PYTHONPATH:-}
export LLM_ADDICTION_BEHAVIORAL_ROOT=/scratch/llm-addiction-data/behavioral
export LLM_ADDICTION_DATA_ROOT=/scratch/llm-addiction-data/sae_features_v3
export LLM_ADDICTION_ANALYSIS_ROOT=/scratch/llm-addiction/sae_v3_analysis
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

launch_cell () {
  local TAG="$1" GPU="$2" CMD_ARGS="$3"
  local LOG="$LOGDIR/${TAG}.log"
  if [ -f "$LOG" ] && grep -q "WROTE " "$LOG" 2>/dev/null; then
    echo "[skip] $TAG already complete"
    return
  fi
  echo "[launch] $TAG on GPU $GPU"
  CUDA_VISIBLE_DEVICES=$GPU nohup python -u "$SRC/run_l22_held_out_steering.py" \
    $CMD_ARGS \
    > "$LOG" 2>&1 &
  echo "  pid=$! tag=$TAG log=$LOG"
}

case "$ROLE" in
  ic_h1)
    launch_cell h1_a1 1 "--cell H1 --task sm --model llama --layer 22 --n-games 200 --g-offset 4000 --alphas -2.0 -1.0 -0.5 --output $OUTDIR/h1_a1.json"
    launch_cell h1_a2 2 "--cell H1 --task sm --model llama --layer 22 --n-games 200 --g-offset 4000 --alphas 0.0 0.5 --output $OUTDIR/h1_a2.json"
    launch_cell h1_a3 3 "--cell H1 --task sm --model llama --layer 22 --n-games 200 --g-offset 4000 --alphas 1.0 2.0 --output $OUTDIR/h1_a3.json"
    ;;
  mw_h2_h5true)
    launch_cell h2_a1 1 "--cell H2 --task mw --model llama --layer 22 --n-games 100 --g-offset 4000 --alphas -2.0 -1.0 -0.5 0.0 --output $OUTDIR/h2_a1.json"
    launch_cell h2_a2 2 "--cell H2 --task mw --model llama --layer 22 --n-games 100 --g-offset 4000 --alphas 0.5 1.0 2.0 --output $OUTDIR/h2_a2.json"
    launch_cell h5_bk 3 "--cell H5 --task sm --model llama --layer 22 --n-games 20 --g-offset 5000 --alphas 1.0 --output $OUTDIR/h5_bk.json"
    ;;
  natural_h5rand)
    # H5 random axes split across 3 GPUs: 10 indices each, run SEQUENTIALLY per
    # GPU (one model load per GPU, indices iterated within the same process).
    seq_runner () {
      local GPU="$1"; shift
      local INDICES="$@"
      local TAG="h5_seq_g${GPU}"
      local LOG="$LOGDIR/${TAG}.log"
      echo "[seq-launch] $TAG on GPU $GPU indices=$INDICES"
      (
        for i in $INDICES; do
          local OUT="$OUTDIR/h5_r${i}.json"
          if [ -f "$OUT" ] && python3 -c "import json; d=json.load(open('$OUT')); raise SystemExit(0 if d.get('results_by_alpha') else 1)" 2>/dev/null; then
            echo "[skip] h5_r${i} (already complete)"
            continue
          fi
          echo "[seq] starting h5_r${i} on GPU $GPU"
          CUDA_VISIBLE_DEVICES=$GPU python -u "$SRC/run_l22_held_out_steering.py" \
            --cell H5 --task sm --model llama --layer 22 \
            --n-games 20 --g-offset 5000 --alphas 1.0 \
            --random-axis --random-axis-index $i \
            --output "$OUT" >> "$LOG" 2>&1 || echo "[seq] h5_r${i} returned $?"
        done
        echo "[seq] GPU $GPU done"
      ) > "$LOG.outer" 2>&1 &
      disown
      echo "  outer pid=$! log=$LOG"
    }
    seq_runner 1 0 1 2 3 4 5 6 7 8 9
    seq_runner 2 10 11 12 13 14 15 16 17 18 19
    seq_runner 3 20 21 22 23 24 25 26 27 28 29
    ;;
  *)
    echo "unknown ROLE: $ROLE"; exit 2
    ;;
esac

echo "[done] launched cells for ROLE=$ROLE; logs in $LOGDIR; outputs in $OUTDIR"
sleep 2
echo "--- nvidia-smi ---"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "--- pgrep ---"
pgrep -af "run_l22_held_out_steering" | head -20

#!/usr/bin/env bash
set -euo pipefail

# Activate conda environment
source /data/miniforge3/etc/profile.d/conda.sh
conda activate llama_sae_env

ROOT="/home/ubuntu/llm_addiction"
SCRIPT="$ROOT/causal_feature_discovery/src/experiment_2_final_correct.py"
BATCH_DIR="$ROOT/analysis/missing_feature_batches"
LOG_DIR="$ROOT/analysis/missing_feature_runs"
mkdir -p "$LOG_DIR"

run_batch() {
  local gpu="$1"
  local process_id="$2"
  local batch_file="$3"
  echo "[INFO] Launching $process_id on GPU $gpu with $batch_file (n_trials=30)"
  CUDA_VISIBLE_DEVICES="$gpu" python "$SCRIPT" \
    --gpu "$gpu" \
    --process_id "$process_id" \
    --feature_indices_file "$batch_file" \
    --start_idx 0 \
    --end_idx 0 \
    --n_trials 30 \
    > "$LOG_DIR/${process_id}.log" 2>&1 & disown
}

run_batch 4 g4_batch1 "$BATCH_DIR/batch_1.csv"
run_batch 4 g4_batch2 "$BATCH_DIR/batch_2.csv"
run_batch 4 g4_batch3 "$BATCH_DIR/batch_3.csv"

run_batch 5 g5_batch4 "$BATCH_DIR/batch_4.csv"
run_batch 5 g5_batch5 "$BATCH_DIR/batch_5.csv"
run_batch 5 g5_batch6 "$BATCH_DIR/batch_6.csv"

wait

echo "All batches completed."

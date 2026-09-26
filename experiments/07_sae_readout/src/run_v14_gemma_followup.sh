#!/bin/bash
# Queue Gemma V14 follow-up runs after current V14 jobs release the GPU.

set -euo pipefail

BASE_DIR="/home/v-seungplee/llm-addiction/sae_v3_analysis"
SRC_DIR="${BASE_DIR}/src"
RESULTS_DIR="${BASE_DIR}/results"
PYTHON="/home/v-seungplee/miniconda3/envs/llm-addiction/bin/python"
QUEUE_LOG="${RESULTS_DIR}/v14_gemma_followup_queue.log"

mkdir -p "${RESULTS_DIR}"

{
  echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] queued Gemma follow-up runner"

  while pgrep -af "run_v14_(experiments|parallel)\\.py" >/dev/null; do
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] waiting for current V14 processes to finish"
    sleep 300
  done

  cd "${SRC_DIR}"

  echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] starting exp5 (Gemma SM)"
  "${PYTHON}" -u run_v14_parallel.py --exp exp5 --gpu 0 2>&1 | tee "${RESULTS_DIR}/v14_exp5_log.txt"

  echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] starting exp6 (Gemma IC)"
  "${PYTHON}" -u run_v14_parallel.py --exp exp6 --gpu 0 2>&1 | tee "${RESULTS_DIR}/v14_exp6_log.txt"

  echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] Gemma follow-up queue complete"
} >> "${QUEUE_LOG}" 2>&1

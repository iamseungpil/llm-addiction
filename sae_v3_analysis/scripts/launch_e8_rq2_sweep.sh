#!/usr/bin/env bash
set -euo pipefail

AZ_PYTHON="${AZ_PYTHON:-/opt/az/bin/python3}"
CONNECTOR="${CONNECTOR:-/home/v-seungplee/.azure/cliextensions/ml/azext_mlv2/manual/custom/_ssh_connector.py}"
SSH_KEY="${SSH_KEY:-/home/v-seungplee/.ssh/id_rsa}"
URL_E8="${URL_E8:-wss://ssh-2etszrmvdrq4cwqdql4al50f38gyq2afb9nhuq49bngbf1buj3c.westus2.nodes.azureml.ms}"

LOCAL_ANALYSIS_ROOT="${LOCAL_ANALYSIS_ROOT:-/home/v-seungplee/llm-addiction/sae_v3_analysis}"
REMOTE_ROOT="${REMOTE_ROOT:-/scratch/llm_addiction/sae_v3_analysis}"
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/scratch/llm_addiction/data/sae_features_v3}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/opt/conda/envs/ptca/bin/python}"

GPU_ID="${GPU_ID:-1}"
MODEL="${MODEL:-gemma}"
BASIS_METHOD="${BASIS_METHOD:-centroid_pca}"
LAYERS="${LAYERS:-8,12,22,25,30}"
RANKS="${RANKS:-1,2,3}"
N_SPLITS="${N_SPLITS:-5}"
PCA_DIM="${PCA_DIM:-64}"
TASKS="${TASKS:-}"
TAG_PREFIX="${TAG_PREFIX:-e8}"
JOB_NAME="${JOB_NAME:-rq2_sweep_${MODEL}_${BASIS_METHOD}}"

proxy_ssh() {
  ssh -T -o LogLevel=ERROR \
      -o StrictHostKeyChecking=no \
      -o UserKnownHostsFile=/dev/null \
      -o "ProxyCommand=$AZ_PYTHON $CONNECTOR $URL_E8" \
      -i "$SSH_KEY" \
      azureuser@placeholder "$@"
}

proxy_rsync() {
  rsync -az \
    -e "ssh -T -o LogLevel=ERROR -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ProxyCommand='$AZ_PYTHON $CONNECTOR $URL_E8' -i $SSH_KEY" \
    "$@"
}

sync_code() {
  proxy_ssh "mkdir -p $REMOTE_ROOT/src $REMOTE_ROOT/scripts $REMOTE_ROOT/results/logs $REMOTE_ROOT/results/robustness"
  proxy_rsync "$LOCAL_ANALYSIS_ROOT/src/run_rq2_aligned_hidden_transfer.py" "azureuser@placeholder:$REMOTE_ROOT/src/"
  proxy_rsync "$LOCAL_ANALYSIS_ROOT/src/run_rq2_aligned_hidden_transfer_sweep.py" "azureuser@placeholder:$REMOTE_ROOT/src/"
  proxy_rsync "$LOCAL_ANALYSIS_ROOT/scripts/launch_e8_rq2_sweep.sh" "azureuser@placeholder:$REMOTE_ROOT/scripts/"
}

launch() {
  proxy_ssh "bash --noprofile --norc -lc '
    set -euo pipefail
    mkdir -p $REMOTE_ROOT/results/logs $REMOTE_ROOT/results/robustness
    export PYTHONPATH=$REMOTE_ROOT/src
    export LLM_ADDICTION_ANALYSIS_ROOT=$REMOTE_ROOT
    export LLM_ADDICTION_DATA_ROOT=$REMOTE_DATA_ROOT
    LOG=$REMOTE_ROOT/results/logs/${JOB_NAME}.log
    PIDFILE=$REMOTE_ROOT/results/logs/${JOB_NAME}.pid
    nohup env CUDA_VISIBLE_DEVICES=$GPU_ID \
      $REMOTE_PYTHON $REMOTE_ROOT/src/run_rq2_aligned_hidden_transfer_sweep.py \
      --model $MODEL \
      --basis-method $BASIS_METHOD \
      --layers $LAYERS \
      --ranks $RANKS \
      --n-splits $N_SPLITS \
      --pca-dim $PCA_DIM \
      --tag-prefix $TAG_PREFIX \
      ${TASKS:+--tasks $TASKS} > \$LOG 2>&1 < /dev/null &
    echo \$! > \$PIDFILE
    echo launched:\$(cat \$PIDFILE)
  '"
}

status() {
  proxy_ssh "bash --noprofile --norc -lc '
    set -euo pipefail
    echo __GPU__
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
    echo __PROC__
    ps -eo pid,etimes,cmd | grep -E \"$JOB_NAME|run_rq2_aligned_hidden_transfer.py\" | grep -v grep || true
    echo __LOG_TAIL__
    tail -n 60 $REMOTE_ROOT/results/logs/${JOB_NAME}.log 2>/dev/null || echo NO_LOG
  '"
}

cmd="${1:-all}"
case "$cmd" in
  sync) sync_code ;;
  launch) launch ;;
  status) status ;;
  all) sync_code; launch; status ;;
  *) echo "Usage: $0 [sync|launch|status|all]" >&2; exit 1 ;;
esac

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

proxy_ssh() {
  ssh -o StrictHostKeyChecking=no \
      -o UserKnownHostsFile=/dev/null \
      -o "ProxyCommand=$AZ_PYTHON $CONNECTOR $URL_E8" \
      -i "$SSH_KEY" \
      azureuser@placeholder "$@"
}

proxy_rsync() {
  rsync -az \
    --delete \
    -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ProxyCommand='$AZ_PYTHON $CONNECTOR $URL_E8' -i $SSH_KEY" \
    "$@"
}

mkdir_remote() {
  proxy_ssh "mkdir -p $REMOTE_ROOT/src $REMOTE_ROOT/scripts $REMOTE_ROOT/results/json $REMOTE_ROOT/results/logs $REMOTE_ROOT/results/figures $REMOTE_DATA_ROOT/slot_machine $REMOTE_DATA_ROOT/investment_choice $REMOTE_DATA_ROOT/mystery_wheel"
}

sync_code() {
  proxy_rsync "$LOCAL_ANALYSIS_ROOT/src/" "azureuser@placeholder:$REMOTE_ROOT/src/"
  proxy_rsync "$LOCAL_ANALYSIS_ROOT/scripts/" "azureuser@placeholder:$REMOTE_ROOT/scripts/"
}

sync_data() {
  local base="/home/v-seungplee/data/llm-addiction/sae_features_v3"
  proxy_rsync "$base/slot_machine/gemma/" "azureuser@placeholder:$REMOTE_DATA_ROOT/slot_machine/gemma/"
  proxy_rsync "$base/investment_choice/gemma/" "azureuser@placeholder:$REMOTE_DATA_ROOT/investment_choice/gemma/"
  proxy_rsync "$base/mystery_wheel/gemma/" "azureuser@placeholder:$REMOTE_DATA_ROOT/mystery_wheel/gemma/"
  proxy_rsync "$base/slot_machine/llama/hidden_states_dp.npz" "azureuser@placeholder:$REMOTE_DATA_ROOT/slot_machine/llama/hidden_states_dp.npz"
  proxy_rsync "$base/investment_choice/llama/hidden_states_dp.npz" "azureuser@placeholder:$REMOTE_DATA_ROOT/investment_choice/llama/hidden_states_dp.npz"
  proxy_rsync "$base/mystery_wheel/llama/hidden_states_dp.npz" "azureuser@placeholder:$REMOTE_DATA_ROOT/mystery_wheel/llama/hidden_states_dp.npz"
}

launch_jobs() {
  proxy_ssh "bash -lc '
    set -euo pipefail
    mkdir -p $REMOTE_ROOT/results/logs
    export PYTHONPATH=$REMOTE_ROOT/src
    export LLM_ADDICTION_ANALYSIS_ROOT=$REMOTE_ROOT
    export LLM_ADDICTION_DATA_ROOT=$REMOTE_DATA_ROOT
    nohup env CUDA_VISIBLE_DEVICES=0 $REMOTE_PYTHON $REMOTE_ROOT/src/run_v16_multilayer_steering.py --model gemma --n-bk-games 200 --n-rand-games 100 --n-random-dirs 20 --layers all > $REMOTE_ROOT/results/logs/e8_v16_gemma.log 2>&1 < /dev/null & echo \$! > $REMOTE_ROOT/results/logs/e8_v16_gemma.pid
    nohup env CUDA_VISIBLE_DEVICES=1 $REMOTE_PYTHON $REMOTE_ROOT/src/run_v12_crossdomain_steering.py --n 3 --combos sm-ic,ic-sm > $REMOTE_ROOT/results/logs/e8_v12_crossdomain_smoke.log 2>&1 < /dev/null & echo \$! > $REMOTE_ROOT/results/logs/e8_v12_crossdomain_smoke.pid
    echo launched
  '"
}

status() {
  proxy_ssh "bash -lc '
    set -euo pipefail
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
    echo
    ps -eo pid,etimes,cmd | grep -E \"run_v16_multilayer_steering|run_v12_crossdomain_steering\" | grep -v grep || true
    echo
    tail -n 20 $REMOTE_ROOT/results/logs/e8_v16_gemma.log 2>/dev/null || true
    echo
    tail -n 20 $REMOTE_ROOT/results/logs/e8_v12_crossdomain_smoke.log 2>/dev/null || true
  '"
}

cmd="${1:-all}"
case "$cmd" in
  mkdir) mkdir_remote ;;
  sync) mkdir_remote; sync_code; sync_data ;;
  launch) launch_jobs ;;
  status) status ;;
  all) mkdir_remote; sync_code; sync_data; launch_jobs; status ;;
  *) echo "Usage: $0 [mkdir|sync|launch|status|all]" >&2; exit 1 ;;
esac

#!/usr/bin/env bash
# Remote AMLT launcher for M3' indicator-direction steering + Track B work.
# Dispatches by JOB_ROLE env var:
#   gemma_sm_iba  : Gemma-2-9B SM, I_BA, full dose ladder + specificity
#   llama_sm_iba  : LLaMA-3.1-8B SM, I_BA, prep Ridge/d_unit + dose ladder + specificity
#   gemma_xtask   : Gemma MW + IC, focused 3-alpha ladder cross-task
#   llama_xtask   : LLaMA MW + IC, mirror of gemma_xtask
#
# Required env: HF_TOKEN, JOB_ROLE
set -euo pipefail

ROLE=${JOB_ROLE:-gemma_sm_iba}
N=${N_PER_COND:-80}
LOG=/scratch/logs/m3prime_${ROLE}.log
mkdir -p /scratch/logs
exec > >(tee -a "$LOG") 2>&1

echo "=== [start $(date '+%Y-%m-%dT%H:%M:%S')] role=$ROLE n=$N ==="
nvidia-smi -L || true

# --- 1) Install minimal deps (user-site to avoid /opt/conda permission denial) ---
export PYTHONUSERBASE=/scratch/pyuserbase
mkdir -p "$PYTHONUSERBASE"
export PATH="$PYTHONUSERBASE/bin:$PATH"
python -m pip install --user --quiet --no-cache-dir --no-warn-script-location \
    huggingface_hub hf_transfer scikit-learn statsmodels pandas sae_lens \
    || { echo "[fatal] pip install --user failed (deps)"; exit 1; }
# transformers / torch are typically already in the ptca image; only install if missing
python -c "import transformers, sae_lens; print('transformers', transformers.__version__)" \
    || python -m pip install --user --quiet --no-cache-dir --no-warn-script-location \
        "transformers>=4.45,<4.50" "torch>=2.5" sae_lens \
    || { echo "[fatal] pip install --user failed (transformers/torch)"; exit 1; }
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false

# --- 2) Fetch latest code from GitHub ---
mkdir -p /scratch/code && cd /scratch/code
if [ ! -d llm-addiction ]; then
    git clone --depth 1 https://github.com/iamseungpil/llm-addiction.git
fi
cd /scratch/code/llm-addiction
git pull --rebase || true
echo "[code] commit: $(git log -1 --oneline)"

# --- 3) Fetch behavioral + SAE feature data from HF ---
DATA_ROOT=/scratch/data/llm-addiction
mkdir -p "$DATA_ROOT"

python <<'PYEOF'
import os
from pathlib import Path
from huggingface_hub import snapshot_download
token = os.environ['HF_TOKEN']
repo = 'llm-addiction-research/llm-addiction'

# Behavioral game JSONs (small, fast)
print(f'[data] downloading behavioral game JSONs')
snapshot_download(
    repo_id=repo, repo_type='dataset', token=token,
    local_dir='/scratch/data/llm-addiction',
    allow_patterns=['behavioral/**/*.json'],
)
# SAE features (large NPZ)
print(f'[data] downloading sae_features_v3 NPZs')
snapshot_download(
    repo_id=repo, repo_type='dataset', token=token,
    local_dir='/scratch/data/llm-addiction',
    allow_patterns=['sae_features_v3/**/*.npz'],
)
# Direction metadata for steering
print(f'[data] downloading M3prime direction_metadata')
snapshot_download(
    repo_id=repo, repo_type='dataset', token=token,
    local_dir='/scratch/code/llm-addiction',
    allow_patterns=['sae_v3_analysis/results/v19_multi_patching/M3prime_indicator_steering/direction_metadata/*.json'],
)
print('[data] done')
PYEOF

# --- 4) Symlink data into the canonical local paths the scripts expect ---
mkdir -p /home/v-seungplee/data
ln -sfn "$DATA_ROOT" /home/v-seungplee/data/llm-addiction
mkdir -p /home/v-seungplee/llm-addiction
ln -sfn /scratch/code/llm-addiction/sae_v3_analysis /home/v-seungplee/llm-addiction/sae_v3_analysis

cd /scratch/code/llm-addiction/sae_v3_analysis

# --- 5) Background HF push scheduler ---
nohup python scripts/m3_push_scheduler.py --interval 600 \
    > /scratch/logs/push_scheduler.log 2>&1 &
PUSH_PID=$!
echo "[push] scheduler PID=$PUSH_PID"

# --- 6) Dispatch by role ---
PY=python
GPU=0

run_dose_ladder() {
    local model=$1 task=$2 dir=$3
    for cond_alpha in "alpha-2:-2.0" "alpha-1:-1.0" "alpha+0:0.0" "alpha+1:1.0" "alpha+2:2.0" "alpha+3:3.0"; do
        IFS=':' read -r cond alpha <<< "$cond_alpha"
        echo "=== [$(date '+%H:%M:%S')] $model/$task/$dir $cond α=$alpha ==="
        $PY src/run_m3prime_indicator_steering.py \
            --model "$model" --task "$task" \
            --condition "$cond" --alpha "$alpha" --direction "$dir" --layer 22 \
            --n $N --gpu $GPU
    done
}

run_specificity() {
    local model=$1 task=$2
    echo "=== [$(date '+%H:%M:%S')] $model/$task random direction (α=+2σ, L22) ==="
    $PY src/run_m3prime_indicator_steering.py --model "$model" --task "$task" \
        --condition random --alpha 2.0 --direction random --layer 22 --n $N --gpu $GPU
    echo "=== [$(date '+%H:%M:%S')] $model/$task L8 layer specificity ==="
    $PY src/run_m3prime_indicator_steering.py --model "$model" --task "$task" \
        --condition L8 --alpha 2.0 --direction i_ba --layer 8 --n $N --gpu $GPU
    echo "=== [$(date '+%H:%M:%S')] $model/$task ILC indicator specificity ==="
    $PY src/run_m3prime_indicator_steering.py --model "$model" --task "$task" \
        --condition ILC --alpha 2.0 --direction i_lc --layer 22 --n $N --gpu $GPU
}

prep_direction_if_missing() {
    local model=$1 task=$2 ind=$3
    local steer_file="results/v19_multi_patching/M3prime_indicator_steering/direction_metadata/${model}_${task}_${ind}_L22_steering.json"
    if [ -f "$steer_file" ]; then
        echo "[prep] $steer_file already present"
        return 0
    fi
    echo "=== [prep] computing Ridge weights + d_unit for $model/$task/$ind ==="
    $PY src/extract_section4_ridge_weights.py --model "$model" --task "$task" --indicator "$ind"
    $PY src/compute_section4_steering_directions.py --model "$model" --task "$task" --indicator "$ind"
}

case "$ROLE" in
    gemma_sm_iba)
        run_dose_ladder gemma sm i_ba
        run_specificity gemma sm
        $PY src/aggregate_m3prime_dose_response.py --model gemma --task sm
        ;;
    llama_sm_iba)
        prep_direction_if_missing llama sm i_ba
        prep_direction_if_missing llama sm i_lc
        run_dose_ladder llama sm i_ba
        run_specificity llama sm
        $PY src/aggregate_m3prime_dose_response.py --model llama --task sm
        ;;
    gemma_xtask)
        for task in mw ic; do
            for ind in i_ba i_lc; do
                prep_direction_if_missing gemma "$task" "$ind"
            done
            for cond_alpha in "alpha-1:-1.0" "alpha+0:0.0" "alpha+2:2.0"; do
                IFS=':' read -r cond alpha <<< "$cond_alpha"
                $PY src/run_m3prime_indicator_steering.py \
                    --model gemma --task "$task" \
                    --condition "$cond" --alpha "$alpha" --direction i_ba --layer 22 \
                    --n $N --gpu $GPU
            done
            $PY src/aggregate_m3prime_dose_response.py --model gemma --task "$task" || true
        done
        ;;
    llama_xtask)
        for task in mw ic; do
            for ind in i_ba i_lc; do
                prep_direction_if_missing llama "$task" "$ind"
            done
            for cond_alpha in "alpha-1:-1.0" "alpha+0:0.0" "alpha+2:2.0"; do
                IFS=':' read -r cond alpha <<< "$cond_alpha"
                $PY src/run_m3prime_indicator_steering.py \
                    --model llama --task "$task" \
                    --condition "$cond" --alpha "$alpha" --direction i_ba --layer 22 \
                    --n $N --gpu $GPU
            done
            $PY src/aggregate_m3prime_dose_response.py --model llama --task "$task" || true
        done
        ;;
    *)
        echo "[fatal] unknown JOB_ROLE: $ROLE"
        exit 1
        ;;
esac

# --- 7) Final HF push ---
echo "=== [final-push $(date '+%H:%M:%S')] ==="
$PY scripts/m3_push_scheduler.py --once || true

echo "=== [done $(date '+%Y-%m-%dT%H:%M:%S')] role=$ROLE ==="
sleep 300  # leave time for last push to flush

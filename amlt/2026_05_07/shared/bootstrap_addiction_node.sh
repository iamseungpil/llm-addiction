#!/usr/bin/env bash
# Drop -e so individual failures don't abort the entire bootstrap.
set -uo pipefail

mkdir -p /scratch/logs /scratch/code

# --- conda activate (read-only ptca env on AMLT) -----------------------------
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
if [ -d /opt/conda/envs/ptca ]; then
    conda activate ptca
elif [ -d /opt/conda/envs/llm-addiction ]; then
    conda activate llm-addiction
fi

echo "[bootstrap] python=$(which python) version=$(python --version 2>&1)"

# --- pip deps with --user (ptca env is read-only) ----------------------------
PIP_DEPS="huggingface_hub transformers accelerate pandas numpy scipy scikit-learn pyyaml bambi pymc anthropic openai google-generativeai sae_lens"
echo "[bootstrap] installing pip deps with --user: $PIP_DEPS"
pip install --user --quiet --upgrade $PIP_DEPS 2>&1 | tail -10 || echo "[bootstrap] pip --user had issues, continuing"

# Make sure user-site is on Python path for any spawned subprocess.
USER_SITE="$(python -c 'import site; print(site.getusersitepackages())' 2>/dev/null || echo /home/aiscuser/.local/lib/python3.10/site-packages)"
export PYTHONPATH="${USER_SITE}:${PYTHONPATH:-}"
export PATH="$(python -c 'import site; import os; print(os.path.dirname(site.getusersitepackages())+\"/../bin\")' 2>/dev/null):$PATH"
echo "[bootstrap] USER_SITE=$USER_SITE"

# Verify imports
python -c "import huggingface_hub, transformers, pandas, numpy, scipy, sklearn, yaml, openai, anthropic; print('[bootstrap] core deps OK')" || \
    echo "[bootstrap] WARNING: core deps import failed — re-trying without --quiet"

# Re-try pip install verbosely if anything is missing
python -c "import huggingface_hub" 2>/dev/null || \
    pip install --user huggingface_hub 2>&1 | tail -5

# --- Pull and extract code tarball (ALWAYS fresh; data on /scratch is preserved) -
cd /scratch/code
rm -rf /scratch/code/llm-addiction
python - <<'PYEOF'
import os, tarfile, shutil
from huggingface_hub import hf_hub_download
p = hf_hub_download(
    repo_id='iamseungpil/llm-addiction-rebuttal-2026-05',
    filename='code_snapshots/2026_05_07/code.tar.gz',
    repo_type='dataset',
    token=os.environ['HF_TOKEN'],
    force_download=True,
)
shutil.copy(p, '/scratch/code/code.tar.gz')
t = tarfile.open('/scratch/code/code.tar.gz')
t.extractall('/scratch/code')
t.close()
print('[bootstrap] fresh tarball extracted to /scratch/code/llm-addiction')
PYEOF

# --- Output dirs -------------------------------------------------------------
mkdir -p /scratch/x3415a02/data/llm-addiction/track0_w3
mkdir -p /scratch/x3415a02/data/llm-addiction/m2_persona
mkdir -p /scratch/x3415a02/data/llm-addiction/m5_residualisation
mkdir -p /scratch/x3415a02/data/llm-addiction/m1_portfolio
mkdir -p /scratch/x3415a02/data/llm-addiction/d_robustness

# --- Background helpers ------------------------------------------------------
if [ ! -f /scratch/logs/gpu_keeper.pid ] || ! kill -0 "$(cat /scratch/logs/gpu_keeper.pid 2>/dev/null)" 2>/dev/null; then
    nohup python /scratch/code/llm-addiction/amlt/2026_05_07/shared/gpu_keeper.py \
        > /scratch/logs/gpu_keeper.log 2>&1 &
    echo $! > /scratch/logs/gpu_keeper.pid
fi

if [ ! -f /scratch/logs/push_ckpts.pid ] || ! kill -0 "$(cat /scratch/logs/push_ckpts.pid 2>/dev/null)" 2>/dev/null; then
    nohup python /scratch/code/llm-addiction/amlt/2026_05_07/shared/push_ckpts_to_hf.py \
        --interval 600 \
        --base_dir /scratch/x3415a02/data/llm-addiction \
        --hf_repo iamseungpil/llm-addiction-rebuttal-2026-05 \
        > /scratch/logs/push_ckpts.log 2>&1 &
    echo $! > /scratch/logs/push_ckpts.pid
fi

cd /scratch/code/llm-addiction
echo "bootstrap done at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "cwd: $(pwd)"
ls -la paper_experiments/track0_w3_replication/src/ 2>&1 | head -5

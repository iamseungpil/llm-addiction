#!/bin/bash
# Remote launch for llm-addiction aligned factor steering on H100 node.
#
# Features:
#   - Resume from HF: downloads existing checkpoints at start
#   - Periodic HF sync: pushes checkpoints + partial results every 5 min
#   - GPU keeper: prevents BSC idle-suspend
#   - Final result push on completion
#
# Required env (injected by AMLT or passed on SSH):
#   HF_TOKEN    -- HuggingFace token
#   NODE_ROLE   -- steering_expa_ic | steering_expa_mw

set -uo pipefail

NODE_ROLE=${NODE_ROLE:-steering_expa_ic}
SCRATCH=/scratch
WORK=$SCRATCH/llm-addiction
DATA=$SCRATCH/llm-addiction-data
LOG=$SCRATCH/logs
CKPT=$SCRATCH/llm-addiction/sae_v3_analysis/results/checkpoints
RESULTS=$SCRATCH/llm-addiction/sae_v3_analysis/results/json
mkdir -p $LOG $CKPT $RESULTS

HF_REPO=iamseungpil/metacot
HF_CKPT_PREFIX="llm_addiction_results/${NODE_ROLE}/checkpoints"
HF_RESULTS_PREFIX="llm_addiction_results/${NODE_ROLE}/results"
HF_LOG_PREFIX="llm_addiction_results/${NODE_ROLE}/logs"

echo "[$(date)] === llm-addiction launcher start (role=$NODE_ROLE) ==="
nvidia-smi | head -20

# --- GPU keeper ---
cat > $SCRATCH/gpu_keeper.py <<'PY'
import os, time, torch
STOP = "/scratch/gpu_keeper.stop"
DONE = "/scratch/bootstrap.done"
ng = torch.cuda.device_count()
print(f"[gpu_keeper] starting on {ng} GPUs", flush=True)
while not os.path.exists(STOP):
    try:
        for d in range(ng):
            with torch.cuda.device(d):
                a = torch.randn(256, 256, device=f"cuda:{d}", dtype=torch.float16)
                _ = (a @ a).sum().item()
        time.sleep(60 if not os.path.exists(DONE) else 180)
    except Exception as e:
        print(f"[gpu_keeper] warn {e}", flush=True); time.sleep(30)
PY
rm -f $SCRATCH/gpu_keeper.stop
nohup python $SCRATCH/gpu_keeper.py > $LOG/gpu_keeper.log 2>&1 &
echo $! > $LOG/gpu_keeper.pid
echo "[$(date)] gpu_keeper pid=$(cat $LOG/gpu_keeper.pid)"

# --- Fetch code + data ---
pip install -q huggingface_hub hf_transfer 2>/dev/null || true
export HF_HUB_ENABLE_HF_TRANSFER=1

if [ ! -d "$WORK/sae_v3_analysis/src" ]; then
  python - <<PY
import os, tarfile
from huggingface_hub import hf_hub_download
token = os.environ["HF_TOKEN"]
code = hf_hub_download(repo_id="iamseungpil/metacot", filename="code_snapshots/llm_addiction_code.tar.gz", repo_type="dataset", token=token)
os.makedirs("/scratch/llm-addiction", exist_ok=True)
tarfile.open(code).extractall("/scratch/llm-addiction")
print(f"code extracted")
PY
fi

if [ ! -d "$DATA/behavioral" ]; then
  python - <<PY
import os, tarfile
from huggingface_hub import hf_hub_download
token = os.environ["HF_TOKEN"]
data = hf_hub_download(repo_id="iamseungpil/metacot", filename="llm_addiction_data/llama_ic_mw_bundle.tar.gz", repo_type="dataset", token=token)
os.makedirs("/scratch/llm-addiction-data", exist_ok=True)
tarfile.open(data).extractall("/scratch/llm-addiction-data")
print(f"data extracted")
PY
fi

# --- Resume: pull existing checkpoints from HF ---
echo "[$(date)] resuming from HF checkpoints..."
python - <<PY
import os
from huggingface_hub import HfApi, hf_hub_download
api = HfApi(token=os.environ["HF_TOKEN"])
role = os.environ["NODE_ROLE"]
ckpt_dst = "/scratch/llm-addiction/sae_v3_analysis/results/checkpoints"
os.makedirs(ckpt_dst, exist_ok=True)
try:
    files = api.list_repo_files(repo_id="iamseungpil/metacot", repo_type="dataset")
    pref = f"llm_addiction_results/{role}/checkpoints/"
    restored = 0
    for f in files:
        if f.startswith(pref) and f.endswith(".json"):
            local = hf_hub_download(repo_id="iamseungpil/metacot", filename=f, repo_type="dataset", token=os.environ["HF_TOKEN"])
            import shutil
            tgt = os.path.join(ckpt_dst, os.path.basename(f))
            shutil.copy2(local, tgt)
            restored += 1
    print(f"restored {restored} checkpoint files")
except Exception as e:
    print(f"no previous checkpoints or err: {e}")
PY

# --- Bootstrap python deps ---
pip install -q transformers==4.44.2 accelerate==0.33.0 safetensors==0.4.4 numpy scipy pandas scikit-learn einops statsmodels 2>&1 | tail -3

touch $SCRATCH/bootstrap.done

# --- Periodic HF sync daemon (every 5 min) ---
cat > $SCRATCH/hf_sync.py <<'PY'
import os, time, glob, tarfile, traceback
from huggingface_hub import HfApi
STOP = "/scratch/hf_sync.stop"
INTERVAL = 300  # 5 min
api = HfApi(token=os.environ["HF_TOKEN"])
role = os.environ["NODE_ROLE"]
ckpt_dir = "/scratch/llm-addiction/sae_v3_analysis/results/checkpoints"
results_dir = "/scratch/llm-addiction/sae_v3_analysis/results/json"
log_dir = "/scratch/logs"
print(f"[hf_sync] daemon started (interval={INTERVAL}s, role={role})", flush=True)
while not os.path.exists(STOP):
    try:
        # Push each checkpoint JSON individually
        for f in sorted(glob.glob(f"{ckpt_dir}/ckpt_*.json")):
            rel = os.path.basename(f)
            tgt = f"llm_addiction_results/{role}/checkpoints/{rel}"
            try:
                api.upload_file(path_or_fileobj=f, path_in_repo=tgt, repo_id="iamseungpil/metacot", repo_type="dataset")
                print(f"[hf_sync] pushed {rel}", flush=True)
            except Exception as e:
                print(f"[hf_sync] WARN {rel}: {e}", flush=True)
        # Push partial results JSONs
        for f in sorted(glob.glob(f"{results_dir}/aligned_steering_*.json")):
            rel = os.path.basename(f)
            tgt = f"llm_addiction_results/{role}/results/{rel}"
            try:
                api.upload_file(path_or_fileobj=f, path_in_repo=tgt, repo_id="iamseungpil/metacot", repo_type="dataset")
                print(f"[hf_sync] pushed result {rel}", flush=True)
            except Exception as e:
                print(f"[hf_sync] WARN result {rel}: {e}", flush=True)
        # Push tail of main log
        main_log = f"{log_dir}/steering.log"
        if os.path.exists(main_log):
            try:
                api.upload_file(path_or_fileobj=main_log, path_in_repo=f"llm_addiction_results/{role}/logs/steering.log", repo_id="iamseungpil/metacot", repo_type="dataset")
            except Exception as e:
                pass
    except Exception as e:
        print(f"[hf_sync] outer err: {e}", flush=True)
        traceback.print_exc()
    time.sleep(INTERVAL)
print("[hf_sync] stopping", flush=True)
PY

rm -f $SCRATCH/hf_sync.stop
nohup python $SCRATCH/hf_sync.py > $LOG/hf_sync.log 2>&1 &
echo $! > $LOG/hf_sync.pid
echo "[$(date)] hf_sync pid=$(cat $LOG/hf_sync.pid)"

# --- Run the steering experiment ---
cd /scratch/llm-addiction/sae_v3_analysis
export LLM_ADDICTION_BEHAVIORAL_ROOT=/scratch/llm-addiction-data/behavioral
export LLM_ADDICTION_DATA_ROOT=/scratch/llm-addiction-data/sae_features_v3
export LLM_ADDICTION_ANALYSIS_ROOT=/scratch/llm-addiction/sae_v3_analysis
export PYTHONPATH=/scratch/llm-addiction/sae_v3_analysis/src:/scratch/llm-addiction/paper_experiments/slot_machine_6models/src:/scratch/llm-addiction/exploratory_experiments/alternative_paradigms/src:${PYTHONPATH:-}

case "$NODE_ROLE" in
  steering_expa_ic)
    TASK_ARG="--experiment a --model llama --task-filter ic --gpu 0"
    ;;
  steering_expa_mw)
    TASK_ARG="--experiment a --model llama --task-filter mw --gpu 0"
    ;;
  *)
    echo "unknown NODE_ROLE=$NODE_ROLE"; exit 1 ;;
esac

echo "[$(date)] launching: python src/run_aligned_factor_steering.py $TASK_ARG"
python src/run_aligned_factor_steering.py $TASK_ARG 2>&1 | tee $LOG/steering.log

RC=$?
echo "[$(date)] steering finished (rc=$RC)"

# --- Final HF push (force fresh copy of everything) ---
python - <<PY
import os, glob
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
role = os.environ["NODE_ROLE"]
for base in ("results/json", "results/checkpoints"):
    d = f"/scratch/llm-addiction/sae_v3_analysis/{base}"
    for f in sorted(glob.glob(f"{d}/**/*.json", recursive=True)):
        rel = os.path.relpath(f, d)
        sub = "results" if "json" in base else "checkpoints"
        tgt = f"llm_addiction_results/{role}/final_{sub}/{rel}"
        try:
            api.upload_file(path_or_fileobj=f, path_in_repo=tgt, repo_id="iamseungpil/metacot", repo_type="dataset")
            print(f"final pushed {rel}", flush=True)
        except Exception as e:
            print(f"final WARN {rel}: {e}", flush=True)
for f in ("/scratch/logs/steering.log", "/scratch/logs/hf_sync.log", "/scratch/logs/gpu_keeper.log"):
    if os.path.exists(f):
        try:
            api.upload_file(path_or_fileobj=f, path_in_repo=f"llm_addiction_results/{role}/final_logs/{os.path.basename(f)}", repo_id="iamseungpil/metacot", repo_type="dataset")
        except: pass
PY

touch $SCRATCH/hf_sync.stop
touch $SCRATCH/gpu_keeper.stop
echo "[$(date)] === launcher complete (rc=$RC) ==="

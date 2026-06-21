#!/bin/bash
# Remote launch for LLaMA SM + MW all-layer (L0-31) SAE feature extraction (C2).
#
# Reuses extract_llama_sm.py / extract_llama_mw.py UNCHANGED. They already:
#   - default to --layers all (0-31), matching the canonical llama-IC schema,
#   - checkpoint Phase A hidden states and skip per-layer NPZs that already exist
#     (so a preempted job resumes on plain resubmission),
#   - emit sae_features_L{0-31}.npz with the same sparse-COO + metadata schema as
#     the existing llama-IC features.
#
# Required env (injected by AMLT):
#   HF_TOKEN   -- HuggingFace token (for code/data pull + result push)
#   NODE_ROLE  -- extract_llama_sm | extract_llama_mw

set -uo pipefail

NODE_ROLE=${NODE_ROLE:-extract_llama_sm}
SCRATCH=/scratch
WORK=$SCRATCH/llm-addiction
DATA=$SCRATCH/llm-addiction-data
LOG=$SCRATCH/logs
mkdir -p "$LOG"

HF_REPO=iamseungpil/metacot

echo "[$(date)] === llama SM/MW extraction launcher (role=$NODE_ROLE) ==="
nvidia-smi | head -20

# --- GPU keeper: prevents BSC idle-suspend during the long Phase A pass ---
cat > "$SCRATCH/gpu_keeper.py" <<'PY'
import os, time, torch
STOP = "/scratch/gpu_keeper.stop"
ng = torch.cuda.device_count()
print(f"[gpu_keeper] starting on {ng} GPUs", flush=True)
while not os.path.exists(STOP):
    try:
        for d in range(ng):
            with torch.cuda.device(d):
                a = torch.randn(256, 256, device=f"cuda:{d}", dtype=torch.float16)
                _ = (a @ a).sum().item()
        time.sleep(180)
    except Exception as e:
        print(f"[gpu_keeper] warn {e}", flush=True); time.sleep(30)
PY
rm -f "$SCRATCH/gpu_keeper.stop"
nohup python "$SCRATCH/gpu_keeper.py" > "$LOG/gpu_keeper.log" 2>&1 &
echo $! > "$LOG/gpu_keeper.pid"

# --- Fetch code + behavioral data (skip if already present from a prior attempt) ---
pip install -q huggingface_hub hf_transfer 2>/dev/null || true
export HF_HUB_ENABLE_HF_TRANSFER=1

if [ ! -d "$WORK/sae_v3_analysis/src" ]; then
  python - <<PY
import os, tarfile
from huggingface_hub import hf_hub_download
code = hf_hub_download(repo_id="$HF_REPO", filename="code_snapshots/llm_addiction_code.tar.gz", repo_type="dataset", token=os.environ["HF_TOKEN"])
os.makedirs("/scratch/llm-addiction", exist_ok=True)
tarfile.open(code).extractall("/scratch/llm-addiction")
print("code extracted")
PY
fi

if [ ! -d "$DATA/behavioral" ]; then
  python - <<PY
import os, tarfile
from huggingface_hub import hf_hub_download
data = hf_hub_download(repo_id="$HF_REPO", filename="llm_addiction_data/llama_ic_mw_bundle.tar.gz", repo_type="dataset", token=os.environ["HF_TOKEN"])
os.makedirs("/scratch/llm-addiction-data", exist_ok=True)
tarfile.open(data).extractall("/scratch/llm-addiction-data")
print("data extracted")
PY
fi

# --- Resume: pull any sae_features_L*.npz already on HF so finished layers are skipped ---
python - <<PY
import os, shutil
from huggingface_hub import HfApi, hf_hub_download
role = os.environ["NODE_ROLE"]
task = "slot_machine" if role.endswith("sm") else "mystery_wheel"
dst = f"/scratch/llm-addiction-data/sae_features_v3/{task}/llama"
os.makedirs(dst, exist_ok=True)
api = HfApi(token=os.environ["HF_TOKEN"])
pref = f"llm_addiction_results/{role}/sae_features/"
restored = 0
try:
    for f in api.list_repo_files(repo_id="$HF_REPO", repo_type="dataset"):
        if f.startswith(pref):
            local = hf_hub_download(repo_id="$HF_REPO", filename=f, repo_type="dataset", token=os.environ["HF_TOKEN"])
            shutil.copy2(local, os.path.join(dst, os.path.basename(f)))
            restored += 1
    print(f"restored {restored} prior feature files")
except Exception as e:
    print(f"no prior features or err: {e}")
PY

pip install -q transformers==4.44.2 accelerate==0.33.0 safetensors==0.4.4 numpy scipy tqdm 2>&1 | tail -3

# --- Run the matching extractor (Phase A hidden states -> Phase B fnlp SAE, all 32 layers) ---
cd "$WORK/sae_v3_analysis"
export PYTHONUNBUFFERED=1
DATA_ROOT=/scratch/llm-addiction-data
case "$NODE_ROLE" in
  extract_llama_sm)
    SCRIPT=src/extract_llama_sm.py
    DATA_DIR=$DATA_ROOT/behavioral/slot_machine/llama_v4_role
    OUT_DIR=$DATA_ROOT/sae_features_v3/slot_machine/llama
    ;;
  extract_llama_mw)
    SCRIPT=src/extract_llama_mw.py
    DATA_DIR=$DATA_ROOT/behavioral/mystery_wheel/llama_v2_role
    OUT_DIR=$DATA_ROOT/sae_features_v3/mystery_wheel/llama
    ;;
  *)
    echo "unknown NODE_ROLE=$NODE_ROLE"; exit 1 ;;
esac

echo "[$(date)] launching: python $SCRIPT --gpu 0 --data-dir $DATA_DIR --output-dir $OUT_DIR --layers all"
python "$SCRIPT" --gpu 0 --data-dir "$DATA_DIR" --output-dir "$OUT_DIR" --layers all 2>&1 | tee "$LOG/extract.log"
RC=$?
echo "[$(date)] extraction finished (rc=$RC)"

# --- Push all emitted sae_features_L*.npz (+ sidecar JSON, summary) to HF ---
python - <<PY
import os, glob
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
role = os.environ["NODE_ROLE"]
out = "$OUT_DIR"
for f in sorted(glob.glob(f"{out}/sae_features_L*.npz")) + sorted(glob.glob(f"{out}/sae_features_L*.json")) + glob.glob(f"{out}/extraction_summary.json"):
    rel = os.path.basename(f)
    try:
        api.upload_file(path_or_fileobj=f, path_in_repo=f"llm_addiction_results/{role}/sae_features/{rel}", repo_id="$HF_REPO", repo_type="dataset")
        print(f"pushed {rel}", flush=True)
    except Exception as e:
        print(f"WARN {rel}: {e}", flush=True)
PY

touch "$SCRATCH/gpu_keeper.stop"
echo "[$(date)] === extraction launcher complete (rc=$RC) ==="

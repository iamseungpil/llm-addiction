"""
Permutation Null for I_LC using HIDDEN STATES (matches Table 1's actual method)
================================================================================
The paper's I_LC R² values come from run_irrationality_probe.py which uses
raw hidden states + Ridge with covariates (round, balance, prompt_combo).
This script validates those values with game-level block permutation.
"""
import numpy as np
import json
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

RNG = np.random.RandomState(42)
DATA_ROOT = Path("/home/v-seungplee/data/llm-addiction/sae_features_v3")
BEHAVIORAL_ROOT = Path("/home/v-seungplee/data/llm-addiction/behavioral")
RESULTS_DIR = Path("/home/v-seungplee/llm-addiction/sae_v3_analysis/results/robustness")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
N_PERM = 200


def load_hs_and_labels(model, paradigm, layer_idx):
    """Load hidden states and compute I_LC labels (matching run_irrationality_probe.py)."""
    task_dirs = {"sm": "slot_machine", "mw": "mystery_wheel"}

    # Load SAE meta for game_ids, round_nums etc
    # Use all-layers hidden states
    sae_path = DATA_ROOT / task_dirs[paradigm] / model / f"sae_features_L{layer_idx}.npz"
    sae_data = np.load(sae_path, allow_pickle=False)
    gids = sae_data["game_ids"]
    rn = sae_data["round_nums"].astype(float)
    bt = sae_data["bet_types"]
    bal = sae_data["balances"].astype(float) if "balances" in sae_data else np.full(len(gids), np.nan)
    conditions = sae_data["prompt_conditions"] if "prompt_conditions" in sae_data else np.zeros(len(gids))

    # Load hidden states at the corresponding layer from DP file
    dp_path = DATA_ROOT / task_dirs[paradigm] / model / "hidden_states_dp.npz"
    if not dp_path.exists():
        # Use per-round hidden states if available
        # For now, use SAE features as proxy - but this isn't right
        print(f"  WARNING: No hidden states file, cannot validate HS I_LC")
        return None

    # Actually, DP hidden states are per-GAME (decision point), not per-round
    # The irrationality probe uses per-ROUND hidden states from a different extraction
    # We need to find the per-round hidden state file
    all_hs_path = DATA_ROOT / task_dirs[paradigm] / model / "hidden_states_all_rounds.npz"
    if not all_hs_path.exists():
        # Try alternative path
        print(f"  Per-round hidden states not found at {all_hs_path}")
        print(f"  Looking for alternatives...")
        # The irrationality probe loaded from a specific path
        # Let's check what files exist
        hs_dir = DATA_ROOT / task_dirs[paradigm] / model
        hs_files = list(hs_dir.glob("hidden_states*.npz"))
        print(f"  Available HS files: {[f.name for f in hs_files]}")
        return None

    return None  # placeholder


def load_behavioral_data(model, paradigm):
    """Load behavioral data for I_LC computation."""
    if paradigm == "sm":
        if model == "gemma":
            gpath = BEHAVIORAL_ROOT / "slot_machine/gemma_v4_role/final_gemma_20260227_002507.json"
        else:
            gpath = BEHAVIORAL_ROOT / "slot_machine/llama_v4_role/final_llama_20260315_062428.json"
        with open(gpath) as f:
            raw = json.load(f)
        return raw.get("results", raw.get("games", []))
    elif paradigm == "mw":
        mw_dir = BEHAVIORAL_ROOT / f"mystery_wheel/{model}_v2_role"
        games = []
        for f in sorted(mw_dir.glob(f"{model}_mysterywheel_*.json")):
            d = json.load(open(f))
            r = d.get("results", d.get("games", []))
            games.extend(r.values() if isinstance(r, dict) else r)
        return games


if __name__ == "__main__":
    # First, check what per-round hidden state files exist
    for model in ["gemma", "llama"]:
        for task in ["slot_machine", "mystery_wheel"]:
            hs_dir = DATA_ROOT / task / model
            hs_files = list(hs_dir.glob("hidden_states*.npz"))
            print(f"{model} {task}: {[f.name for f in hs_files]}")

    # Check the irrationality probe's data loading
    print("\n--- Checking irrationality probe data source ---")
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    # Try to load the same way as run_irrationality_probe
    for model in ["gemma", "llama"]:
        for para_short, para_full in [("sm", "slot_machine")]:
            sae_dir = DATA_ROOT / para_full / model
            # Check for round-level hidden states
            for f in sorted(sae_dir.glob("*.npz")):
                d = np.load(f, allow_pickle=True)
                keys = list(d.keys())
                if 'hidden_states' in keys:
                    hs = d['hidden_states']
                    print(f"  {f.name}: hs shape={hs.shape}, keys={keys[:5]}...")

    print("\nDONE - diagnostic complete")

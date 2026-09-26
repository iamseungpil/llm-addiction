#!/usr/bin/env python3
"""
V14 Causal Validation Experiments
- Exp1: Permutation test (100 random dirs, LLaMA SM, n=200)
- Exp2: LLaMA IC/MW n=200 + 10 random controls
- Exp3: Cross-domain steering random controls (MW→IC, MW→SM, IC→SM)
- Exp4: Gemma MW n=200 + 10 random controls
"""

import os
import sys
import json
import time
import numpy as np
import torch
from scipy import stats
from datetime import datetime
from pathlib import Path

# Paths
DATA_ROOT = Path("/home/v-seungplee/data/llm-addiction/sae_features_v3")
RESULTS_DIR = Path("/home/v-seungplee/llm-addiction/sae_v3_analysis/results/json")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Global settings
LAYER = 22
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def load_hidden_states(model_name, task_name):
    """Load hidden_states_dp.npz for a model-task combo."""
    task_dirs = {"sm": "slot_machine", "ic": "investment_choice", "mw": "mystery_wheel"}
    path = DATA_ROOT / task_dirs[task_name] / model_name / "hidden_states_dp.npz"
    print(f"  Loading {path}...")
    data = np.load(path, allow_pickle=True)
    return data


def extract_bk_direction(hidden_states, outcomes, layer):
    """Extract BK direction vector: mean(BK) - mean(Safe) at given layer."""
    # hidden_states shape: (n_games, n_layers, hidden_dim) or keyed by layer
    if isinstance(hidden_states, np.lib.npyio.NpzFile):
        # Try different key formats
        for key_fmt in [f"layer_{layer}", f"L{layer}", f"hidden_states"]:
            if key_fmt in hidden_states:
                hs = hidden_states[key_fmt]
                if hs.ndim == 3:  # (n_games, n_layers, hidden_dim)
                    hs = hs[:, layer, :]
                break
        else:
            # Try loading all and indexing
            keys = list(hidden_states.keys())
            print(f"    Available keys: {keys[:10]}")
            hs = hidden_states[keys[0]]
            if hs.ndim == 3:
                hs = hs[:, layer, :]
    else:
        hs = hidden_states

    bk_mask = (outcomes == 1) | (outcomes == True)
    safe_mask = ~bk_mask

    bk_mean = hs[bk_mask].mean(axis=0)
    safe_mean = hs[safe_mask].mean(axis=0)

    direction = bk_mean - safe_mean
    return direction, hs


def generate_random_directions(dim, n_dirs, seed=42):
    """Generate random unit-norm direction vectors."""
    rng = np.random.RandomState(seed)
    dirs = rng.randn(n_dirs, dim)
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    return dirs / norms


def run_steering_experiment(model, tokenizer, task_name, direction, alpha_values, n_games, model_name="llama"):
    """Run steering experiment: add direction to residual stream, measure BK rate."""
    from run_v12_all_steering import (
        build_sm_prompt, build_ic_prompt, build_mw_prompt,
        parse_sm_response, parse_ic_response, parse_mw_response,
        play_sm_game, play_ic_game, play_mw_game
    )

    results = {}
    direction_tensor = torch.tensor(direction, dtype=torch.float16).to(DEVICE)

    for alpha in alpha_values:
        # Register hook
        hooks = []
        def make_hook(a, v):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    modified = output[0] + a * v.unsqueeze(0).unsqueeze(0)
                    return (modified,) + output[1:]
                return output + a * v.unsqueeze(0).unsqueeze(0)
            return hook_fn

        # Find the target layer
        if "llama" in model_name.lower():
            layer_module = model.model.layers[LAYER]
        else:  # gemma
            layer_module = model.model.layers[LAYER]

        hook = layer_module.register_forward_hook(make_hook(alpha, direction_tensor))
        hooks.append(hook)

        # Play games
        bk_count = 0
        for game_idx in range(n_games):
            try:
                if task_name == "sm":
                    result = play_sm_game(model, tokenizer, seed=game_idx + alpha * 10000)
                elif task_name == "ic":
                    result = play_ic_game(model, tokenizer, seed=game_idx + alpha * 10000)
                elif task_name == "mw":
                    result = play_mw_game(model, tokenizer, seed=game_idx + alpha * 10000)

                if result.get("bankrupt", False) or result.get("outcome", "") == "bankrupt":
                    bk_count += 1
            except Exception as e:
                print(f"    Game {game_idx} failed: {e}")

        # Remove hooks
        for h in hooks:
            h.remove()

        bk_rate = bk_count / n_games
        results[str(alpha)] = {"bk_count": bk_count, "n": n_games, "bk_rate": bk_rate}
        print(f"    alpha={alpha:+.1f}: BK={bk_count}/{n_games} = {bk_rate:.3f}")

    return results


def compute_steering_stats(results, alpha_values):
    """Compute Spearman rho and p-value from steering results."""
    bk_rates = [results[str(a)]["bk_rate"] for a in alpha_values]
    if len(set(bk_rates)) <= 1:
        return float('nan'), float('nan'), bk_rates
    rho, p = stats.spearmanr(alpha_values, bk_rates)
    return rho, p, bk_rates


def run_experiment_set(model, tokenizer, model_name, task_name,
                       bk_direction, hidden_dim, alpha_values,
                       n_games, n_random, exp_name, source_task=None):
    """Run BK direction + N random controls."""
    print(f"\n{'='*60}")
    print(f"  {exp_name}: {model_name} {task_name} (n={n_games}, {n_random} randoms)")
    print(f"{'='*60}")

    # Normalize BK direction to unit norm (same scale as randoms)
    bk_norm = np.linalg.norm(bk_direction)
    bk_unit = bk_direction / bk_norm

    # BK direction steering
    print(f"\n  [BK Direction] (norm={bk_norm:.2f})")
    bk_results = run_steering_experiment(
        model, tokenizer, task_name, bk_direction, alpha_values, n_games, model_name
    )
    bk_rho, bk_p, bk_rates = compute_steering_stats(bk_results, alpha_values)
    print(f"  BK Direction: rho={bk_rho:.4f}, p={bk_p:.6f}")

    # Random controls
    random_dirs = generate_random_directions(hidden_dim, n_random)
    # Scale randoms to same norm as BK direction
    random_dirs_scaled = random_dirs * bk_norm

    random_results = []
    for i in range(n_random):
        print(f"\n  [Random {i+1}/{n_random}]")
        r_results = run_steering_experiment(
            model, tokenizer, task_name, random_dirs_scaled[i], alpha_values, n_games, model_name
        )
        r_rho, r_p, r_rates = compute_steering_stats(r_results, alpha_values)
        print(f"  Random {i}: rho={r_rho:.4f}, p={r_p:.6f}")
        random_results.append({
            "rho": float(r_rho) if not np.isnan(r_rho) else None,
            "p": float(r_p) if not np.isnan(r_p) else None,
            "bk_rates": {str(a): r_results[str(a)]["bk_rate"] for a in alpha_values}
        })

    # Compute permutation p-value
    random_abs_rhos = [abs(r["rho"]) for r in random_results if r["rho"] is not None]
    if random_abs_rhos and not np.isnan(bk_rho):
        perm_p = (sum(1 for r in random_abs_rhos if r >= abs(bk_rho)) + 1) / (len(random_abs_rhos) + 1)
        n_random_sig = sum(1 for r in random_results if r["p"] is not None and r["p"] < 0.05)
    else:
        perm_p = float('nan')
        n_random_sig = 0

    # Verdict
    bk_sig = not np.isnan(bk_p) and bk_p < 0.05
    direction_specific = bk_sig and (perm_p < 0.05)

    if direction_specific:
        verdict = "BK_SPECIFIC_CONFIRMED"
    elif bk_sig:
        verdict = "BK_SIGNIFICANT_BUT_NOT_SPECIFIC"
    else:
        verdict = "NOT_SIGNIFICANT"

    result = {
        "experiment": exp_name,
        "timestamp": TIMESTAMP,
        "model": model_name,
        "task": task_name,
        "source_task": source_task,
        "layer": LAYER,
        "n_games": n_games,
        "n_random": n_random,
        "alpha_values": alpha_values,
        "bk_direction": {
            "rho": float(bk_rho) if not np.isnan(bk_rho) else None,
            "p": float(bk_p) if not np.isnan(bk_p) else None,
            "bk_rates": {str(a): bk_results[str(a)]["bk_rate"] for a in alpha_values},
            "norm": float(bk_norm)
        },
        "random_controls": random_results,
        "permutation_p": float(perm_p) if not np.isnan(perm_p) else None,
        "n_random_significant": n_random_sig,
        "verdict": verdict
    }

    # Save
    filename = f"v14_{exp_name}_{TIMESTAMP}.json"
    with open(RESULTS_DIR / filename, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n  VERDICT: {verdict}")
    print(f"  Permutation p-value: {perm_p:.4f}")
    print(f"  Random significant: {n_random_sig}/{n_random}")
    print(f"  Saved to {filename}")

    return result


def main():
    print(f"V14 Causal Validation - {TIMESTAMP}")
    print(f"Device: {DEVICE}")

    # Add source directory to path
    sys.path.insert(0, "/home/v-seungplee/llm-addiction/sae_v3_analysis/src")

    alpha_values = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    all_results = {}

    # ============================================================
    # PHASE 1: LLaMA experiments (SM permutation, IC/MW replication)
    # ============================================================
    print("\n" + "=" * 60)
    print("  PHASE 1: Loading LLaMA-3.1-8B-Instruct")
    print("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    llama_model_id = "meta-llama/Llama-3.1-8B-Instruct"
    print(f"  Loading {llama_model_id}...")
    tokenizer_llama = AutoTokenizer.from_pretrained(llama_model_id)
    model_llama = AutoModelForCausalLM.from_pretrained(
        llama_model_id, torch_dtype=torch.float16, device_map="auto"
    )
    model_llama.eval()

    # --- Exp1: LLaMA SM permutation test (n=200, 100 random dirs) ---
    print("\n>>> Exp1: LLaMA SM Permutation Test")
    sm_data = load_hidden_states("llama", "sm")
    sm_keys = list(sm_data.keys())
    print(f"    Keys: {sm_keys}")

    # Extract hidden states and outcomes
    if 'hidden_states' in sm_data and 'outcomes' in sm_data:
        sm_hs = sm_data['hidden_states']
        sm_outcomes = sm_data['outcomes']
    elif 'dp_hidden_states' in sm_data:
        sm_hs = sm_data['dp_hidden_states']
        sm_outcomes = sm_data.get('dp_outcomes', sm_data.get('outcomes', None))
    else:
        # Try first key
        sm_hs = sm_data[sm_keys[0]]
        sm_outcomes = sm_data[sm_keys[1]] if len(sm_keys) > 1 else None

    print(f"    Hidden states shape: {sm_hs.shape}")
    print(f"    Outcomes shape: {sm_outcomes.shape if sm_outcomes is not None else 'N/A'}")

    if sm_hs.ndim == 3:
        sm_hs_l22 = sm_hs[:, LAYER, :]
    else:
        sm_hs_l22 = sm_hs

    bk_mask = (sm_outcomes == 1) | (sm_outcomes == True)
    sm_bk_dir = sm_hs_l22[bk_mask].mean(axis=0) - sm_hs_l22[~bk_mask].mean(axis=0)
    hidden_dim = sm_hs_l22.shape[1]

    print(f"    BK direction norm: {np.linalg.norm(sm_bk_dir):.4f}")
    print(f"    Hidden dim: {hidden_dim}")
    print(f"    BK count: {bk_mask.sum()}, Safe count: {(~bk_mask).sum()}")

    # Run Exp1: 100 random controls
    all_results["exp1_llama_sm_permutation"] = run_experiment_set(
        model_llama, tokenizer_llama, "llama", "sm",
        sm_bk_dir, hidden_dim, alpha_values,
        n_games=200, n_random=100, exp_name="exp1_llama_sm_permutation"
    )

    # --- Exp2a: LLaMA IC n=200 + 10 randoms ---
    print("\n>>> Exp2a: LLaMA IC (n=200, 10 randoms)")
    ic_data = load_hidden_states("llama", "ic")
    ic_keys = list(ic_data.keys())
    if 'hidden_states' in ic_data and 'outcomes' in ic_data:
        ic_hs = ic_data['hidden_states']
        ic_outcomes = ic_data['outcomes']
    elif 'dp_hidden_states' in ic_data:
        ic_hs = ic_data['dp_hidden_states']
        ic_outcomes = ic_data.get('dp_outcomes', ic_data.get('outcomes', None))
    else:
        ic_hs = ic_data[ic_keys[0]]
        ic_outcomes = ic_data[ic_keys[1]] if len(ic_keys) > 1 else None

    if ic_hs.ndim == 3:
        ic_hs_l22 = ic_hs[:, LAYER, :]
    else:
        ic_hs_l22 = ic_hs

    ic_bk_mask = (ic_outcomes == 1) | (ic_outcomes == True)
    ic_bk_dir = ic_hs_l22[ic_bk_mask].mean(axis=0) - ic_hs_l22[~ic_bk_mask].mean(axis=0)

    all_results["exp2a_llama_ic"] = run_experiment_set(
        model_llama, tokenizer_llama, "llama", "ic",
        ic_bk_dir, hidden_dim, alpha_values,
        n_games=200, n_random=10, exp_name="exp2a_llama_ic"
    )

    # --- Exp2b: LLaMA MW n=200 + 10 randoms ---
    print("\n>>> Exp2b: LLaMA MW (n=200, 10 randoms)")
    mw_data = load_hidden_states("llama", "mw")
    mw_keys = list(mw_data.keys())
    if 'hidden_states' in mw_data and 'outcomes' in mw_data:
        mw_hs = mw_data['hidden_states']
        mw_outcomes = mw_data['outcomes']
    elif 'dp_hidden_states' in mw_data:
        mw_hs = mw_data['dp_hidden_states']
        mw_outcomes = mw_data.get('dp_outcomes', mw_data.get('outcomes', None))
    else:
        mw_hs = mw_data[mw_keys[0]]
        mw_outcomes = mw_data[mw_keys[1]] if len(mw_keys) > 1 else None

    if mw_hs.ndim == 3:
        mw_hs_l22 = mw_hs[:, LAYER, :]
    else:
        mw_hs_l22 = mw_hs

    mw_bk_mask = (mw_outcomes == 1) | (mw_outcomes == True)
    mw_bk_dir = mw_hs_l22[mw_bk_mask].mean(axis=0) - mw_hs_l22[~mw_bk_mask].mean(axis=0)

    all_results["exp2b_llama_mw"] = run_experiment_set(
        model_llama, tokenizer_llama, "llama", "mw",
        mw_bk_dir, hidden_dim, alpha_values,
        n_games=200, n_random=10, exp_name="exp2b_llama_mw"
    )

    # --- Exp3: Cross-domain steering with random controls ---
    print("\n>>> Exp3: Cross-domain steering + random controls")

    cross_combos = [
        ("mw", "ic", mw_bk_dir, "MW→IC"),
        ("mw", "sm", mw_bk_dir, "MW→SM"),
        ("ic", "sm", ic_bk_dir, "IC→SM"),
    ]

    for source_task, target_task, direction, label in cross_combos:
        print(f"\n  --- {label} ---")
        all_results[f"exp3_{source_task}_to_{target_task}"] = run_experiment_set(
            model_llama, tokenizer_llama, "llama", target_task,
            direction, hidden_dim, alpha_values,
            n_games=50, n_random=5, exp_name=f"exp3_{source_task}_to_{target_task}",
            source_task=source_task
        )

    # Free LLaMA memory
    del model_llama
    del tokenizer_llama
    torch.cuda.empty_cache()

    # ============================================================
    # PHASE 2: Gemma experiments
    # ============================================================
    print("\n" + "=" * 60)
    print("  PHASE 2: Loading Gemma-2-9B-IT")
    print("=" * 60)

    gemma_model_id = "google/gemma-2-9b-it"
    print(f"  Loading {gemma_model_id}...")
    tokenizer_gemma = AutoTokenizer.from_pretrained(gemma_model_id)
    model_gemma = AutoModelForCausalLM.from_pretrained(
        gemma_model_id, torch_dtype=torch.float16, device_map="auto"
    )
    model_gemma.eval()

    # --- Exp4: Gemma MW n=200 + 10 randoms ---
    print("\n>>> Exp4: Gemma MW (n=200, 10 randoms)")
    gemma_mw_data = load_hidden_states("gemma", "mw")
    gemma_mw_keys = list(gemma_mw_data.keys())
    if 'hidden_states' in gemma_mw_data and 'outcomes' in gemma_mw_data:
        gemma_mw_hs = gemma_mw_data['hidden_states']
        gemma_mw_outcomes = gemma_mw_data['outcomes']
    elif 'dp_hidden_states' in gemma_mw_data:
        gemma_mw_hs = gemma_mw_data['dp_hidden_states']
        gemma_mw_outcomes = gemma_mw_data.get('dp_outcomes', gemma_mw_data.get('outcomes', None))
    else:
        gemma_mw_hs = gemma_mw_data[gemma_mw_keys[0]]
        gemma_mw_outcomes = gemma_mw_data[gemma_mw_keys[1]] if len(gemma_mw_keys) > 1 else None

    if gemma_mw_hs.ndim == 3:
        gemma_mw_hs_l22 = gemma_mw_hs[:, LAYER, :]
    else:
        gemma_mw_hs_l22 = gemma_mw_hs

    gemma_mw_bk_mask = (gemma_mw_outcomes == 1) | (gemma_mw_outcomes == True)
    gemma_mw_bk_dir = gemma_mw_hs_l22[gemma_mw_bk_mask].mean(axis=0) - gemma_mw_hs_l22[~gemma_mw_bk_mask].mean(axis=0)
    gemma_hidden_dim = gemma_mw_hs_l22.shape[1]

    all_results["exp4_gemma_mw"] = run_experiment_set(
        model_gemma, tokenizer_gemma, "gemma", "mw",
        gemma_mw_bk_dir, gemma_hidden_dim, alpha_values,
        n_games=200, n_random=10, exp_name="exp4_gemma_mw"
    )

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 60)
    print("  V14 SUMMARY")
    print("=" * 60)

    for name, result in all_results.items():
        v = result.get("verdict", "N/A")
        bd = result.get("bk_direction", {})
        rho = bd.get("rho", "N/A")
        pp = result.get("permutation_p", "N/A")
        n_rs = result.get("n_random_significant", "N/A")
        n_r = result.get("n_random", "N/A")
        print(f"  {name}: verdict={v}, rho={rho}, perm_p={pp}, random_sig={n_rs}/{n_r}")

    # Save combined summary
    summary_path = RESULTS_DIR / f"v14_summary_{TIMESTAMP}.json"
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()

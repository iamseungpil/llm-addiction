#!/usr/bin/env python3
"""
V15 Activation Steering: Gemma SM + LLaMA SM Cross-Model Causal Validation
===========================================================================
Plan (Codex-approved, 2 rounds):
  Pilot: Gemma SM, L25, BK direction only, 50 games, 9 alphas → calibrate range
  Exp A: Gemma SM, L25, 200 BK / 100 rand games, 50 random dirs
  Exp B: LLaMA SM, L22, 200 BK / 100 rand games, 50 random dirs

Key methodological choices:
  - Direction: mean(BK DP) - mean(Safe DP) at decision point (last round)
  - Alpha=0 included in rho; also report rho-without-0 as sensitivity
  - Random dirs: isotropic Gaussian → unit-norm → scaled to BK direction norm
  - PCA sanity check before steering
"""

import os, sys, json, time, re, random
import numpy as np
import torch
from scipy import stats
from sklearn.decomposition import PCA
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

DATA_ROOT = Path("/home/v-seungplee/data/llm-addiction/sae_features_v3")
RESULTS_DIR = Path("/home/v-seungplee/llm-addiction/sae_v3_analysis/results/json")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# ── Import game functions ──────────────────────────────────────────
from run_v12_all_steering import (
    build_sm_prompt, parse_sm_response, play_game
)


# ── Direction Extraction ───────────────────────────────────────────

def extract_bk_direction(model_name, task="sm", layer_idx=None):
    """Extract BK direction from DP hidden states.

    Returns: direction vector, hidden_dim, metadata dict
    """
    task_dirs = {"sm": "slot_machine", "ic": "investment_choice", "mw": "mystery_wheel"}
    dp_path = DATA_ROOT / task_dirs[task] / model_name / "hidden_states_dp.npz"

    # For LLaMA, use all-layers file if layer not in standard set
    all_layers_path = DATA_ROOT / task_dirs[task] / model_name / "hidden_states_dp_all_layers.npz"

    print(f"Loading DP hidden states from {dp_path}...")
    d = np.load(dp_path, allow_pickle=True)

    layers = d["layers"]
    outcomes = d["game_outcomes"]
    valid = d["valid_mask"].astype(bool)
    hs = d["hidden_states"]  # (n_games, n_layers, hidden_dim)

    # Find layer index
    if layer_idx is None:
        layer_idx = len(layers) - 2  # second to last

    target_layer = layers[layer_idx] if layer_idx < len(layers) else layer_idx

    # If target layer not in stored layers, try all-layers file
    if target_layer not in layers and all_layers_path.exists():
        print(f"  Layer {target_layer} not in {list(layers)}, using all-layers file...")
        d_all = np.load(all_layers_path, allow_pickle=True)
        hs_all = d_all["hidden_states"]  # (n_games, 32, hidden_dim)
        hs_layer = hs_all[valid, target_layer, :]
        outcomes_valid = d_all["game_outcomes"][valid] if "game_outcomes" in d_all else outcomes[valid]
    else:
        li = np.where(layers == target_layer)[0][0]
        hs_layer = hs[valid, li, :]
        outcomes_valid = outcomes[valid]

    bk_mask = outcomes_valid == "bankruptcy"
    safe_mask = ~bk_mask

    n_bk = bk_mask.sum()
    n_safe = safe_mask.sum()

    bk_mean = hs_layer[bk_mask].mean(axis=0)
    safe_mean = hs_layer[safe_mask].mean(axis=0)
    direction = bk_mean - safe_mean

    # PCA sanity check
    pca = PCA(n_components=5)
    pca.fit(hs_layer)
    proj = pca.transform(direction.reshape(1, -1))[0]
    pca_variance_explained = sum(proj**2) / np.linalg.norm(direction)**2

    meta = {
        "model": model_name,
        "task": task,
        "layer": int(target_layer),
        "n_bk": int(n_bk),
        "n_safe": int(n_safe),
        "direction_norm": float(np.linalg.norm(direction)),
        "pca_top5_variance_ratio": float(pca_variance_explained),
        "pca_explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
    }

    print(f"  Layer {target_layer}: n_BK={n_bk}, n_Safe={n_safe}")
    print(f"  Direction norm: {meta['direction_norm']:.4f}")
    print(f"  PCA top-5 variance ratio: {pca_variance_explained:.4f}")
    if pca_variance_explained > 0.5:
        print(f"  ⚠ WARNING: >50% of direction variance in top-5 PCs")

    return direction, hs_layer.shape[1], meta


def generate_random_directions(dim, n_dirs, seed=42):
    """Isotropic Gaussian → unit-norm random directions."""
    rng = np.random.RandomState(seed)
    dirs = rng.randn(n_dirs, dim)
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    return dirs / norms


# ── Steering Engine ────────────────────────────────────────────────

def make_hook(alpha, direction_tensor, device):
    """Create a forward hook that adds alpha * direction to the layer output."""
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            modified = output[0] + alpha * direction_tensor.unsqueeze(0).unsqueeze(0)
            return (modified,) + output[1:]
        return output + alpha * direction_tensor.unsqueeze(0).unsqueeze(0)
    return hook_fn


def load_model(model_name):
    """Load model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if model_name == "gemma":
        model_id = "google/gemma-2-9b-it"
    elif model_name == "llama":
        model_id = "meta-llama/Llama-3.1-8B-Instruct"
    else:
        raise ValueError(f"Unknown model: {model_name}")

    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    return model, tokenizer


def get_layer_module(model, model_name, layer):
    """Get the transformer layer module for hook registration."""
    return model.model.layers[layer]


def run_steering(model, tokenizer, model_name, layer, direction, alpha_values, n_games, seed_offset=0):
    """Run steering for one direction across all alphas. Returns {alpha: bk_rate}."""
    direction_tensor = torch.tensor(direction, dtype=torch.bfloat16).to(DEVICE)
    layer_module = get_layer_module(model, model_name, layer)

    results = {}
    for alpha in alpha_values:
        hook_fn = make_hook(alpha, direction_tensor, DEVICE) if alpha != 0 else None
        bk_count = 0
        for g in range(n_games):
            seed = int(g + seed_offset + abs(alpha) * 100000)
            try:
                import signal
                def timeout_handler(signum, frame):
                    raise TimeoutError("Game timed out")
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(120)  # 2 min timeout per game
                r = play_game(model, tokenizer, DEVICE, hook_fn, layer_module, seed, "sm")
                signal.alarm(0)
                if r.get("bk", False) or r.get("bankrupt", False):
                    bk_count += 1
            except (TimeoutError, Exception) as e:
                signal.alarm(0)
                print(f"      Game {g} failed: {e}")

        bk_rate = bk_count / n_games if n_games > 0 else 0
        results[str(alpha)] = {"bk_count": bk_count, "n": n_games, "bk_rate": bk_rate}
        print(f"    α={alpha:+.1f}: BK={bk_count}/{n_games} ({bk_rate:.1%})")

    return results


def compute_rho(results, alpha_values, exclude_zero=False):
    """Compute Spearman rho between alpha and BK rate."""
    if exclude_zero:
        alphas = [a for a in alpha_values if a != 0]
    else:
        alphas = list(alpha_values)

    rates = [results[str(a)]["bk_rate"] for a in alphas]
    if len(set(rates)) <= 1:
        return float("nan"), float("nan")
    rho, p = stats.spearmanr(alphas, rates)
    return float(rho), float(p)


# ── Experiment Runner ──────────────────────────────────────────────

def run_pilot(model, tokenizer, model_name, layer, direction):
    """Pilot: BK direction only, 50 games, wide alpha range to calibrate."""
    print("\n" + "=" * 60)
    print("  PILOT: Calibrating alpha range")
    print("=" * 60)

    pilot_alphas = [0.0, 0.5, 1.0, 2.0, -0.5, -1.0, -2.0]
    results = run_steering(model, tokenizer, model_name, layer, direction, pilot_alphas, n_games=50)

    rho, p = compute_rho(results, pilot_alphas)
    print(f"\n  Pilot rho={rho:.3f}, p={p:.4f}")
    print(f"  BK rates: {[(a, results[str(a)]['bk_rate']) for a in pilot_alphas]}")

    # Suggest alpha range: find range where BK rate varies meaningfully
    rates = [results[str(a)]["bk_rate"] for a in pilot_alphas]
    rate_range = max(rates) - min(rates)
    print(f"  BK rate range: {rate_range:.3f}")

    return results, pilot_alphas, rho


def run_full_experiment(model, tokenizer, model_name, layer, direction, hidden_dim,
                        alpha_values, n_bk_games, n_rand_games, n_random_dirs, exp_name, direction_meta):
    """Full experiment: BK direction + random controls."""
    print(f"\n{'=' * 60}")
    print(f"  {exp_name}: {model_name} SM L{layer}")
    print(f"  n_bk={n_bk_games}, n_rand={n_rand_games}, n_dirs={n_random_dirs}")
    print(f"{'=' * 60}")

    bk_norm = np.linalg.norm(direction)

    # 1. BK direction
    print(f"\n  [BK Direction] norm={bk_norm:.2f}")
    bk_results = run_steering(model, tokenizer, model_name, layer, direction, alpha_values, n_bk_games)
    bk_rho, bk_p = compute_rho(bk_results, alpha_values, exclude_zero=False)
    bk_rho_no0, bk_p_no0 = compute_rho(bk_results, alpha_values, exclude_zero=True)
    print(f"  BK: rho={bk_rho:.3f} (p={bk_p:.4f}), rho_no0={bk_rho_no0:.3f} (p={bk_p_no0:.4f})")

    # 2. Random controls
    random_dirs = generate_random_directions(hidden_dim, n_random_dirs)
    random_dirs_scaled = random_dirs * bk_norm  # same norm

    random_results = []
    for i in range(n_random_dirs):
        print(f"\n  [Random {i+1}/{n_random_dirs}]")
        r_results = run_steering(
            model, tokenizer, model_name, layer,
            random_dirs_scaled[i], alpha_values, n_rand_games,
            seed_offset=(i + 1) * 1000000
        )
        r_rho, r_p = compute_rho(r_results, alpha_values, exclude_zero=False)
        r_rho_no0, _ = compute_rho(r_results, alpha_values, exclude_zero=True)
        print(f"  Random {i+1}: rho={r_rho:.3f}, rho_no0={r_rho_no0:.3f}")
        random_results.append({
            "rho": r_rho if not np.isnan(r_rho) else None,
            "rho_no0": r_rho_no0 if not np.isnan(r_rho_no0) else None,
            "bk_rates": {str(a): r_results[str(a)]["bk_rate"] for a in alpha_values}
        })

    # 3. Permutation test
    rand_abs_rhos = [abs(r["rho"]) for r in random_results if r["rho"] is not None]
    if rand_abs_rhos and not np.isnan(bk_rho):
        perm_p = (sum(1 for r in rand_abs_rhos if r >= abs(bk_rho)) + 1) / (len(rand_abs_rhos) + 1)
    else:
        perm_p = float("nan")

    # Also for rho_no0
    rand_abs_rhos_no0 = [abs(r["rho_no0"]) for r in random_results if r["rho_no0"] is not None]
    if rand_abs_rhos_no0 and not np.isnan(bk_rho_no0):
        perm_p_no0 = (sum(1 for r in rand_abs_rhos_no0 if r >= abs(bk_rho_no0)) + 1) / (len(rand_abs_rhos_no0) + 1)
    else:
        perm_p_no0 = float("nan")

    n_rand_sig = sum(1 for r in random_results if r["rho"] is not None and abs(r["rho"]) > 0.5)

    # 4. Verdict
    bk_sig = not np.isnan(bk_rho) and abs(bk_rho) > 0.5
    specific = bk_sig and (perm_p < 0.05)

    if specific:
        verdict = "BK_SPECIFIC_CONFIRMED"
    elif bk_sig:
        verdict = "BK_SIGNIFICANT_NOT_SPECIFIC"
    else:
        verdict = "NOT_SIGNIFICANT"

    result = {
        "experiment": exp_name,
        "timestamp": TIMESTAMP,
        "model": model_name,
        "task": "sm",
        "layer": layer,
        "alpha_values": alpha_values,
        "n_bk_games": n_bk_games,
        "n_rand_games": n_rand_games,
        "n_random_dirs": n_random_dirs,
        "direction_meta": direction_meta,
        "bk_direction": {
            "rho": bk_rho,
            "rho_no0": bk_rho_no0,
            "p": bk_p,
            "p_no0": bk_p_no0,
            "bk_rates": {str(a): bk_results[str(a)]["bk_rate"] for a in alpha_values},
            "norm": float(bk_norm),
        },
        "random_controls": random_results,
        "permutation_p": float(perm_p) if not np.isnan(perm_p) else None,
        "permutation_p_no0": float(perm_p_no0) if not np.isnan(perm_p_no0) else None,
        "n_random_with_rho_gt_05": n_rand_sig,
        "random_abs_rhos_sorted": sorted(rand_abs_rhos, reverse=True) if rand_abs_rhos else [],
        "baseline_bk": bk_results["0.0"]["bk_rate"] if "0.0" in bk_results else None,
        "verdict": verdict,
    }

    filename = f"v15_{exp_name}_{TIMESTAMP}.json"
    filepath = RESULTS_DIR / filename
    with open(filepath, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  ═══════════════════════════════════")
    print(f"  VERDICT: {verdict}")
    print(f"  rho={bk_rho:.3f} (perm_p={perm_p:.4f})")
    print(f"  rho_no0={bk_rho_no0:.3f} (perm_p_no0={perm_p_no0:.4f})")
    print(f"  Random |rho| > 0.5: {n_rand_sig}/{n_random_dirs}")
    print(f"  Saved: {filepath}")
    print(f"  ═══════════════════════════════════")

    return result


# ── Main ───────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pilot", "gemma", "llama", "all"], default="all")
    parser.add_argument("--n-bk-games", type=int, default=200)
    parser.add_argument("--n-rand-games", type=int, default=100)
    parser.add_argument("--n-random-dirs", type=int, default=50)
    parser.add_argument("--smoke", action="store_true", help="Quick smoke test (5 games, 3 randoms)")
    args = parser.parse_args()

    if args.smoke:
        args.n_bk_games = 5
        args.n_rand_games = 3
        args.n_random_dirs = 3

    print(f"V15 Steering Experiment — {TIMESTAMP}")
    print(f"Mode: {args.mode}, Device: {DEVICE}")

    # Alpha values (will be updated after pilot)
    alpha_values = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]

    results = {}

    # ── GEMMA ─────────────────────────────────────
    if args.mode in ("pilot", "gemma", "all"):
        gemma_layer = 25  # Closest to L24 probe peak available in DP data
        print(f"\n{'#' * 60}")
        print(f"  GEMMA-2-9B-IT  |  SM  |  L{gemma_layer}")
        print(f"{'#' * 60}")

        # Extract direction
        direction_gemma, hidden_dim_gemma, meta_gemma = extract_bk_direction("gemma", "sm", layer_idx=3)  # layers[3] = 25

        model_gemma, tok_gemma = load_model("gemma")

        if args.mode == "pilot":
            pilot_results, pilot_alphas, pilot_rho = run_pilot(
                model_gemma, tok_gemma, "gemma", gemma_layer, direction_gemma
            )
            # Save pilot
            pilot_out = {
                "experiment": "v15_pilot_gemma_sm",
                "timestamp": TIMESTAMP,
                "results": {str(a): pilot_results[str(a)] for a in pilot_alphas},
                "rho": pilot_rho,
                "direction_meta": meta_gemma,
            }
            with open(RESULTS_DIR / f"v15_pilot_gemma_{TIMESTAMP}.json", "w") as f:
                json.dump(pilot_out, f, indent=2)
            print("Pilot complete. Check results and adjust alpha_values before full run.")
            results["pilot"] = pilot_out
        else:
            # Full Exp A
            results["gemma"] = run_full_experiment(
                model_gemma, tok_gemma, "gemma", gemma_layer,
                direction_gemma, hidden_dim_gemma,
                alpha_values, args.n_bk_games, args.n_rand_games, args.n_random_dirs,
                "expA_gemma_sm", meta_gemma
            )

        # Free GPU memory
        del model_gemma, tok_gemma
        torch.cuda.empty_cache()

    # ── LLAMA ─────────────────────────────────────
    if args.mode in ("llama", "all"):
        llama_layer = 22
        print(f"\n{'#' * 60}")
        print(f"  LLAMA-3.1-8B-Instruct  |  SM  |  L{llama_layer}")
        print(f"{'#' * 60}")

        # Extract direction (use all-layers file for L22)
        direction_llama, hidden_dim_llama, meta_llama = extract_bk_direction("llama", "sm", layer_idx=2)  # layers[2] = 22

        model_llama, tok_llama = load_model("llama")

        results["llama"] = run_full_experiment(
            model_llama, tok_llama, "llama", llama_layer,
            direction_llama, hidden_dim_llama,
            alpha_values, args.n_bk_games, args.n_rand_games, args.n_random_dirs,
            "expB_llama_sm", meta_llama
        )

        del model_llama, tok_llama
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'=' * 60}")
    print("  V15 SUMMARY")
    print(f"{'=' * 60}")
    for name, r in results.items():
        if "verdict" in r:
            print(f"  {name}: {r['verdict']} (rho={r['bk_direction']['rho']:.3f}, perm_p={r['permutation_p']})")

    print("\nDone.")


if __name__ == "__main__":
    main()

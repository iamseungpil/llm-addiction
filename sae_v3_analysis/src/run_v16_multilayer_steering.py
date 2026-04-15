#!/usr/bin/env python3
"""
V16 Multi-Layer Activation Steering
====================================
Gemma: L8, L12, L22, L25, L30 (5 layers, norm varies 3-81)
LLaMA: L8, L12, L22, L25, L30 (5 layers, norm varies 0.1-3.0)

Key improvement over V15:
  - Per-layer alpha scaling based on direction/HS norm ratio
  - Target: α*direction = ~2-5% of HS norm (sweet spot from LLaMA success)
  - All layers tested, best layer identified by rho
  - Logs saved to results/ not /tmp
"""

import os, sys, json, time
import numpy as np
import torch
from scipy import stats
from sklearn.decomposition import PCA
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

ANALYSIS_ROOT = Path(
    os.environ.get("LLM_ADDICTION_ANALYSIS_ROOT", "/home/v-seungplee/llm-addiction/sae_v3_analysis")
)
DATA_ROOT = Path(
    os.environ.get("LLM_ADDICTION_DATA_ROOT", "/home/v-seungplee/data/llm-addiction/sae_features_v3")
)
RESULTS_DIR = ANALYSIS_ROOT / "results" / "json"
LOG_DIR = ANALYSIS_ROOT / "results" / "logs"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

from exact_behavioral_replay import play_exact_behavioral_game, validate_behavioral_catalog


def extract_bk_direction(model_name, task="sm", layer_idx=None):
    task_dirs = {"sm": "slot_machine", "ic": "investment_choice", "mw": "mystery_wheel"}
    dp_path = DATA_ROOT / task_dirs[task] / model_name / "hidden_states_dp.npz"
    d = np.load(dp_path, allow_pickle=True)
    layers = d["layers"]
    outcomes = d["game_outcomes"]
    valid = d["valid_mask"].astype(bool)
    hs = d["hidden_states"]

    if layer_idx is None:
        layer_idx = len(layers) - 2

    target_layer = layers[layer_idx]
    li = np.where(layers == target_layer)[0][0]
    hs_layer = hs[valid, li, :]
    outcomes_valid = outcomes[valid]

    bk_mask = outcomes_valid == "bankruptcy"
    direction = hs_layer[bk_mask].mean(0) - hs_layer[~bk_mask].mean(0)
    hs_norm = np.linalg.norm(hs_layer, axis=1).mean()

    meta = {
        "model": model_name, "task": task, "layer": int(target_layer),
        "n_bk": int(bk_mask.sum()), "n_safe": int((~bk_mask).sum()),
        "direction_norm": float(np.linalg.norm(direction)),
        "hs_mean_norm": float(hs_norm),
        "direction_hs_ratio": float(np.linalg.norm(direction) / hs_norm),
    }
    return direction, hs_layer.shape[1], meta


def generate_random_directions(dim, n_dirs, seed=42):
    rng = np.random.RandomState(seed)
    dirs = rng.randn(n_dirs, dim)
    return dirs / np.linalg.norm(dirs, axis=1, keepdims=True)


def make_hook(alpha, direction_tensor, device):
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            modified = output[0] + alpha * direction_tensor.unsqueeze(0).unsqueeze(0)
            return (modified,) + output[1:]
        return output + alpha * direction_tensor.unsqueeze(0).unsqueeze(0)
    return hook_fn


def load_model(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if model_name == "gemma":
        mid = "google/gemma-2-9b-it"
    else:
        mid = "meta-llama/Llama-3.1-8B-Instruct"
    print(f"Loading {mid}...")
    tokenizer = AutoTokenizer.from_pretrained(mid)
    model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    return model, tokenizer


def get_layer_module(model, model_name, layer):
    if model_name == "gemma":
        return model.model.layers[layer]
    else:
        return model.model.layers[layer]


def run_steering(model, tokenizer, model_name, layer, direction, alpha_values, n_games, seed_offset=0):
    import signal
    direction_tensor = torch.tensor(direction, dtype=torch.bfloat16).to(DEVICE)
    layer_module = get_layer_module(model, model_name, layer)
    results = {}
    for alpha in alpha_values:
        hook_fn = make_hook(alpha, direction_tensor, DEVICE) if alpha != 0 else None
        bk_count = 0
        for g in range(n_games):
            seed = int(g + seed_offset)  # alpha-independent for paired design
            try:
                def timeout_handler(signum, frame):
                    raise TimeoutError("Game timed out")
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(600)
                r = play_exact_behavioral_game(
                    model=model,
                    tokenizer=tokenizer,
                    device=DEVICE,
                    hook_fn=hook_fn,
                    layer_module=layer_module,
                    model_name=model_name,
                    task="sm",
                    game_index=g,
                    seed=seed,
                )
                signal.alarm(0)
                if r.get("bk", False) or r.get("bankrupt", False):
                    bk_count += 1
            except Exception:
                signal.alarm(0)
        bk_rate = bk_count / n_games if n_games > 0 else 0
        results[str(alpha)] = {"bk": bk_count, "total": n_games, "bk_rate": bk_rate}
        print(f"    α={alpha:+.2f}: BK={bk_count}/{n_games} ({bk_rate*100:.1f}%)")
    return results


def compute_rho(results, alpha_values):
    rates = [results[str(a)]["bk_rate"] for a in alpha_values]
    rho, p = stats.spearmanr(alpha_values, rates)
    # Without alpha=0
    no0_alphas = [a for a in alpha_values if a != 0.0]
    no0_rates = [results[str(a)]["bk_rate"] for a in no0_alphas]
    rho_no0, p_no0 = stats.spearmanr(no0_alphas, no0_rates) if len(no0_alphas) > 2 else (np.nan, np.nan)
    return rho, p, rho_no0, p_no0


def compute_alpha_range(direction_norm, hs_norm, target_pct=0.03,
                        mode="pct", absolute_base=None, override=None):
    """Compute alpha values.

    mode='pct': adaptive, target HS displacement fraction (legacy V16 default).
    mode='absolute': scale V14-style range [-2,-1,-0.5,0,+0.5,+1,+2] by absolute_base
                     so α*direction_norm ≈ absolute_base·(V14 scale). If
                     absolute_base is None we reuse legacy V14 LLaMA baseline=1.0,
                     i.e. the raw V14 range.
    mode='override': use exactly the alphas in `override`.
    """
    if mode == "override" and override is not None:
        return [round(float(a), 3) for a in override]

    if mode == "absolute":
        scale = 1.0 if absolute_base is None else float(absolute_base)
        alphas = [-2.0*scale, -1.0*scale, -0.5*scale, 0.0,
                  0.5*scale, 1.0*scale, 2.0*scale]
        return [round(a, 3) for a in alphas]

    base_alpha = target_pct * hs_norm / direction_norm
    alphas = [-2*base_alpha, -base_alpha, -0.5*base_alpha, 0.0,
              0.5*base_alpha, base_alpha, 2*base_alpha]
    return [round(a, 3) for a in alphas]


def run_layer_experiment(model, tokenizer, model_name, layer_idx, n_bk_games, n_rand_games, n_random_dirs, log_f,
                          alpha_mode="pct", alpha_override=None, alpha_absolute_base=None):
    direction, hidden_dim, meta = extract_bk_direction(model_name, "sm", layer_idx)
    layer = meta["layer"]
    direction_norm = meta["direction_norm"]
    hs_norm = meta["hs_mean_norm"]

    alpha_values = compute_alpha_range(
        direction_norm, hs_norm,
        target_pct=0.03,
        mode=alpha_mode,
        absolute_base=alpha_absolute_base,
        override=alpha_override,
    )
    msg = f"\n  Layer {layer}: norm={direction_norm:.2f}, HS_norm={hs_norm:.1f}, " \
          f"ratio={direction_norm/hs_norm:.4f}, n_bk={meta['n_bk']}"
    print(msg); log_f.write(msg + "\n")
    msg = f"  Alpha range (3% HS): {alpha_values}"
    print(msg); log_f.write(msg + "\n")

    # BK direction
    msg = f"  [BK Direction]"
    print(msg); log_f.write(msg + "\n")
    bk_results = run_steering(model, tokenizer, model_name, layer, direction,
                               alpha_values, n_bk_games)
    for a in alpha_values:
        r = bk_results[str(a)]
        msg = f"    α={a:+.2f}: BK={r['bk']}/{r['total']} ({r['bk_rate']*100:.1f}%)"
        log_f.write(msg + "\n")

    bk_rho, bk_p, bk_rho_no0, bk_p_no0 = compute_rho(bk_results, alpha_values)
    msg = f"  BK: rho={bk_rho:.3f} (p={bk_p:.4f}), rho_no0={bk_rho_no0:.3f} (p={bk_p_no0:.4f})"
    print(msg); log_f.write(msg + "\n")

    # Random directions
    rand_dirs = generate_random_directions(hidden_dim, n_random_dirs, seed=42+layer)
    rand_rhos = []
    for ri in range(n_random_dirs):
        scaled_dir = rand_dirs[ri] * direction_norm  # scale to same norm
        msg = f"  [Random {ri+1}/{n_random_dirs}]"
        print(msg); log_f.write(msg + "\n")
        rand_results = run_steering(model, tokenizer, model_name, layer, scaled_dir,
                                     alpha_values, n_rand_games, seed_offset=ri+1)
        for a in alpha_values:
            r = rand_results[str(a)]
            log_f.write(f"    α={a:+.2f}: BK={r['bk']}/{r['total']} ({r['bk_rate']*100:.1f}%)\n")
        r_rho, _, r_rho_no0, _ = compute_rho(rand_results, alpha_values)
        rand_rhos.append(abs(r_rho))
        msg = f"  Random {ri+1}: rho={r_rho:.3f}"
        print(msg); log_f.write(msg + "\n")

    # Specificity
    perm_p = (1 + sum(r >= abs(bk_rho) for r in rand_rhos)) / (1 + n_random_dirs)

    result = {
        "layer": layer, "direction_meta": meta,
        "alpha_values": alpha_values,
        "bk_rho": float(bk_rho), "bk_p": float(bk_p),
        "bk_rho_no0": float(bk_rho_no0), "bk_p_no0": float(bk_p_no0),
        "bk_rates": {str(a): bk_results[str(a)]["bk_rate"] for a in alpha_values},
        "perm_p": float(perm_p),
        "n_rand_above": int(sum(r >= abs(bk_rho) for r in rand_rhos)),
        "rand_rhos": [float(r) for r in rand_rhos],
        "effect_size_pp": float(
            max(bk_results[str(a)]["bk_rate"] for a in alpha_values) -
            min(bk_results[str(a)]["bk_rate"] for a in alpha_values)
        ) * 100,
    }

    msg = f"\n  *** L{layer}: rho={bk_rho:.3f}, perm_p={perm_p:.3f}, effect={result['effect_size_pp']:.1f}pp ***"
    print(msg); log_f.write(msg + "\n")
    log_f.flush()
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["gemma", "llama"], required=True)
    parser.add_argument("--n-bk-games", type=int, default=200)
    parser.add_argument("--n-rand-games", type=int, default=100)
    parser.add_argument("--n-random-dirs", type=int, default=20)
    parser.add_argument("--layers", type=str, default="all",
                        help="Comma-separated layer indices (0-4) or 'all'")
    parser.add_argument("--alpha-mode", choices=["pct", "absolute", "override"], default="pct",
                        help="pct: adaptive 3% HS; absolute: V14-style [-2..+2] · base; override: explicit list")
    parser.add_argument("--alpha-absolute-base", type=float, default=1.0,
                        help="Scale factor when alpha-mode=absolute (α=base·[-2,-1,-0.5,0,+0.5,+1,+2])")
    parser.add_argument("--alpha-values", type=str, default=None,
                        help="Comma-separated α list; implies alpha-mode=override")
    parser.add_argument("--tag", type=str, default="",
                        help="Extra tag appended to output filename")
    args = parser.parse_args()

    layer_indices = list(range(5)) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    alpha_override = None
    if args.alpha_values:
        alpha_override = [float(x) for x in args.alpha_values.split(",")]
        args.alpha_mode = "override"

    log_path = LOG_DIR / f"v16_{args.model}_{TIMESTAMP}.log"
    print(f"V16 Multi-Layer Steering — {args.model.upper()}")
    print(f"Layers: {layer_indices}, Device: {DEVICE}")
    print(f"Log: {log_path}")
    summary = validate_behavioral_catalog("sm", args.model)
    print(
        f"Behavioral catalog sm/{args.model}: n={summary['n_games']}, "
        f"prompt_conditions={len(summary['prompt_conditions'])}, bet_types={summary['bet_types']}"
    )

    model, tokenizer = load_model(args.model)

    all_results = {"model": args.model, "timestamp": TIMESTAMP, "layers": {}}

    with open(log_path, "w") as log_f:
        log_f.write(f"V16 Multi-Layer Steering — {args.model.upper()}\n")
        log_f.write(f"n_bk={args.n_bk_games}, n_rand={args.n_rand_games}, n_dirs={args.n_random_dirs}\n")

        for li in layer_indices:
            result = run_layer_experiment(
                model, tokenizer, args.model, li,
                args.n_bk_games, args.n_rand_games, args.n_random_dirs, log_f,
                alpha_mode=args.alpha_mode,
                alpha_override=alpha_override,
                alpha_absolute_base=args.alpha_absolute_base,
            )
            all_results["layers"][str(result["layer"])] = result

            tag_suffix = f"_{args.tag}" if args.tag else ""
            out_path = RESULTS_DIR / f"v16_{args.model}_{TIMESTAMP}{tag_suffix}.json"
            with open(out_path, "w") as f:
                json.dump(all_results, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print(f"  V16 SUMMARY — {args.model.upper()}")
    print(f"{'='*60}")
    for layer_str, r in sorted(all_results["layers"].items(), key=lambda x: int(x[0])):
        sig = "***" if r["perm_p"] < 0.05 else ""
        print(f"  L{layer_str}: rho={r['bk_rho']:.3f}, perm_p={r['perm_p']:.3f}, "
              f"effect={r['effect_size_pp']:.1f}pp {sig}")

    print(f"\nResults saved to {out_path}")
    print("DONE")


if __name__ == "__main__":
    main()

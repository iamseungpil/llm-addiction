#!/usr/bin/env python3
"""
V14: Rigorous causal validation with proper random controls.
Reuses game logic from run_v12_all_steering.py verbatim.

Experiments:
  Exp1: LLaMA SM - 100 random dirs (n=200) → permutation p-value
  Exp2a: LLaMA IC - 10 random dirs (n=200)
  Exp2b: LLaMA MW - 10 random dirs (n=200)
  Exp3: Cross-domain + 5 random controls each (n=50)
  Exp4: Gemma MW - 10 random dirs (n=200)
"""
import os, sys, json, torch, numpy as np, logging, gc, random
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from run_v12_all_steering import (
    play_game, run_condition, compute_bk_direction,
    build_sm_prompt, build_ic_prompt, build_mw_prompt,
    parse_sm_response, parse_ic_response, parse_mw_response,
    HS_DIR, OUT, TARGET_LAYER, MODELS, ALPHAS as V12_ALPHAS
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("v14")

ALPHAS = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
TS = datetime.now().strftime("%Y%m%d_%H%M%S")


def make_hook(alpha, direction_tensor):
    """Create steering hook: adds alpha * direction to residual stream output."""
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            return (output[0] + alpha * direction_tensor.unsqueeze(0).unsqueeze(0),) + output[1:]
        return output + alpha * direction_tensor.unsqueeze(0).unsqueeze(0)
    return hook_fn


def steering_sweep(model, tokenizer, device, layer_module, direction, task, n_games, alphas=ALPHAS):
    """Run steering for one direction across all alphas using V12's play_game."""
    d_tensor = torch.tensor(direction, dtype=torch.float16 if model.dtype == torch.float16 else torch.bfloat16).to(device)
    results = {}

    for a in alphas:
        if a == 0.0:
            hook_fn = None
        else:
            hook_fn = make_hook(a, d_tensor)

        res = run_condition(model, tokenizer, device, layer_module, hook_fn, f"alpha={a:.1f}", n_games, task)
        results[a] = res["bk_rate"]
        log.info(f"    alpha={a:+.1f}: BK={res['bks']}/{res['n']}={res['bk_rate']:.3f}")

    return results


def compute_rho(results, alphas=ALPHAS):
    """Compute Spearman rho between alpha and BK rate."""
    rates = [results[a] for a in alphas]
    if len(set(rates)) <= 1:
        return float('nan'), float('nan')
    rho, p = spearmanr(alphas, rates)
    return float(rho), float(p)


def gen_random_dirs(dim, n, bk_norm, seed=42):
    """Generate n random direction vectors with same norm as BK direction."""
    rng = np.random.RandomState(seed)
    dirs = rng.randn(n, dim).astype(np.float32)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs *= bk_norm  # same norm as BK direction
    return dirs


def run_experiment(model, tokenizer, device, model_name, task, bk_dir, bk_norm,
                   n_games, n_random, exp_name, source_task=None):
    """Full experiment: BK direction + N random controls → verdict."""
    log.info(f"\n{'='*60}")
    log.info(f"  {exp_name}: {model_name} {task} (n={n_games}, {n_random} randoms)")
    log.info(f"{'='*60}")

    hidden_dim = len(bk_dir)
    layer_module = model.model.layers[TARGET_LAYER]

    # BK direction sweep
    log.info("  [BK Direction]")
    bk_results = steering_sweep(model, tokenizer, device, layer_module, bk_dir, task, n_games)
    bk_rho, bk_p = compute_rho(bk_results)
    log.info(f"  >> BK Direction: rho={bk_rho:.4f}, p={bk_p:.6f}")

    # Random controls
    rand_dirs = gen_random_dirs(hidden_dim, n_random, bk_norm, seed=hash(exp_name) % (2**31))
    rand_results = []
    rand_abs_rhos = []

    for i in range(n_random):
        log.info(f"  [Random {i+1}/{n_random}]")
        r_res = steering_sweep(model, tokenizer, device, layer_module, rand_dirs[i], task, n_games)
        r_rho, r_p = compute_rho(r_res)
        log.info(f"  >> Random {i}: rho={r_rho:.4f}, p={r_p:.6f}")

        rand_results.append({
            "rho": None if np.isnan(r_rho) else round(r_rho, 4),
            "p": None if np.isnan(r_p) else round(r_p, 6),
            "bk_rates": {str(a): round(r_res[a], 4) for a in ALPHAS}
        })
        if not np.isnan(r_rho):
            rand_abs_rhos.append(abs(r_rho))

    # Permutation p-value
    if rand_abs_rhos and not np.isnan(bk_rho):
        n_exceed = sum(1 for r in rand_abs_rhos if r >= abs(bk_rho))
        perm_p = (n_exceed + 1) / (len(rand_abs_rhos) + 1)
    else:
        perm_p = float('nan')

    n_rand_sig = sum(1 for r in rand_results if r["p"] is not None and r["p"] < 0.05)
    bk_sig = not np.isnan(bk_p) and bk_p < 0.05

    # Verdict
    if bk_sig and not np.isnan(perm_p) and perm_p < 0.05:
        verdict = "BK_SPECIFIC_CONFIRMED"
    elif bk_sig:
        verdict = "BK_SIGNIFICANT_NOT_SPECIFIC"
    else:
        verdict = "NOT_SIGNIFICANT"

    result = {
        "experiment": exp_name,
        "timestamp": TS,
        "model": model_name,
        "task": task,
        "source_task": source_task,
        "layer": TARGET_LAYER,
        "n_games": n_games,
        "n_random": n_random,
        "baseline_bk": bk_results.get(0.0),
        "bk_direction": {
            "rho": None if np.isnan(bk_rho) else round(bk_rho, 4),
            "p": None if np.isnan(bk_p) else round(bk_p, 6),
            "bk_rates": {str(a): round(bk_results[a], 4) for a in ALPHAS},
            "norm": round(float(bk_norm), 4)
        },
        "random_controls": rand_results,
        "random_abs_rhos_sorted": [round(r, 4) for r in sorted(rand_abs_rhos, reverse=True)[:20]],
        "permutation_p": None if np.isnan(perm_p) else round(perm_p, 4),
        "n_random_significant": n_rand_sig,
        "verdict": verdict
    }

    fname = f"v14_{exp_name}_{TS}.json"
    with open(OUT / fname, "w") as f:
        json.dump(result, f, indent=2)

    log.info(f"\n  VERDICT: {verdict}")
    log.info(f"  BK rho={bk_rho:.4f}, perm_p={perm_p}, rand_sig={n_rand_sig}/{n_random}")
    log.info(f"  Saved: {fname}")
    return result


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info(f"V14 Causal Validation — {TS}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_results = {}

    # ── PHASE 1: LLaMA ────────────────────────────────────────
    log.info("Loading LLaMA-3.1-8B-Instruct...")
    model_id = MODELS["llama"]["name"]
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    mdl.eval()

    # BK directions for all LLaMA tasks
    sm_unit, sm_norm = compute_bk_direction("llama", "sm")
    ic_unit, ic_norm = compute_bk_direction("llama", "ic")
    mw_unit, mw_norm = compute_bk_direction("llama", "mw")

    sm_dir = sm_unit * sm_norm  # full (non-unit) direction
    ic_dir = ic_unit * ic_norm
    mw_dir = mw_unit * mw_norm

    # Exp1: LLaMA SM permutation (n=100, 20 randoms)
    # 20 randoms: if BK is rank 1, perm_p = 1/21 = 0.048 < 0.05
    all_results["exp1"] = run_experiment(
        mdl, tok, device, "llama", "sm", sm_dir, sm_norm,
        n_games=100, n_random=20, exp_name="exp1_llama_sm_perm20")

    # Exp2a: LLaMA IC (n=100, 10 randoms)
    all_results["exp2a"] = run_experiment(
        mdl, tok, device, "llama", "ic", ic_dir, ic_norm,
        n_games=100, n_random=10, exp_name="exp2a_llama_ic_n100")

    # Exp2b: LLaMA MW (n=100, 10 randoms)
    all_results["exp2b"] = run_experiment(
        mdl, tok, device, "llama", "mw", mw_dir, mw_norm,
        n_games=100, n_random=10, exp_name="exp2b_llama_mw_n100")

    # Exp3: Cross-domain steering + random controls
    for src, tgt, d, norm, label in [
        ("mw", "ic", mw_dir, mw_norm, "mw2ic"),
        ("mw", "sm", mw_dir, mw_norm, "mw2sm"),
        ("ic", "sm", ic_dir, ic_norm, "ic2sm"),
    ]:
        all_results[f"exp3_{label}"] = run_experiment(
            mdl, tok, device, "llama", tgt, d, norm,
            n_games=50, n_random=5, exp_name=f"exp3_{label}",
            source_task=src)

    del mdl, tok
    gc.collect()
    torch.cuda.empty_cache()

    # ── PHASE 2: Gemma ────────────────────────────────────────
    log.info("Loading Gemma-2-9B-IT...")
    model_id = MODELS["gemma"]["name"]
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    mdl.eval()

    gemma_mw_unit, gemma_mw_norm = compute_bk_direction("gemma", "mw")
    gemma_mw_dir = gemma_mw_unit * gemma_mw_norm

    # Exp4: Gemma MW (n=100, 10 randoms)
    all_results["exp4"] = run_experiment(
        mdl, tok, device, "gemma", "mw", gemma_mw_dir, gemma_mw_norm,
        n_games=100, n_random=10, exp_name="exp4_gemma_mw_n100")

    del mdl, tok
    gc.collect()
    torch.cuda.empty_cache()

    # ── Summary ───────────────────────────────────────────────
    log.info(f"\n{'='*60}")
    log.info("  V14 FINAL SUMMARY")
    log.info(f"{'='*60}")

    for k, r in all_results.items():
        bd = r["bk_direction"]
        log.info(f"  {k}: verdict={r['verdict']}, |rho|={abs(bd['rho']) if bd['rho'] else 'N/A':.4f}, "
                 f"perm_p={r['permutation_p']}, rand_sig={r['n_random_significant']}/{r['n_random']}")

    with open(OUT / f"v14_summary_{TS}.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    log.info("V14 complete.")


if __name__ == "__main__":
    main()
